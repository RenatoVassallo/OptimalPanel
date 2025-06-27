import numpy as np
import pandas as pd
import pickle
import os
import torch
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.ar_model import AutoReg
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

class ThompsonSamplerRL:
    def __init__(self, df: pd.DataFrame, unit_col: str, time_col: str, target_col: str,
                 feature_cols: List[str], target_unit: str, donor_units: List[str],
                 forecast_times: List[pd.Timestamp], similarities: Optional[torch.Tensor] = None,
                 seed: int = 42):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.df = df.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.target_unit = target_unit
        self.donor_units = donor_units
        self.forecast_times = forecast_times

        # Default to uniform prior if similarities not provided
        if similarities is None:
            similarities = torch.ones(len(donor_units), dtype=torch.float32)

        assert isinstance(similarities, torch.Tensor), "similarities must be a torch.Tensor"
        assert similarities.shape[0] == len(donor_units), "Length of similarities must match donor_units"

        # Initialize priors using similarity values as pseudo-counts
        self.alpha = similarities.clone()
        self.beta = torch.ones_like(self.alpha)

        self.avg_mse_per_epoch = []
        self.top_bundles = []
        self.inclusion_probs_by_donor = {iso: [] for iso in donor_units}
        self.benchmarks = {}

    def _forecast_for_time(self, t, included_isos):
        df_train = self.df[self.df[self.time_col] < t]
        train_rows = df_train[df_train[self.unit_col] == self.target_unit]
        for iso in included_isos:
            train_rows = pd.concat([train_rows, df_train[df_train[self.unit_col] == iso]])

        X_train = train_rows[self.feature_cols].values
        y_train = train_rows[self.target_col].values

        model = RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

        x_eval = self.df[(self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] == t)][self.feature_cols].values
        y_true = self.df[(self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] == t)][self.target_col].values[0]
        y_pred_val = model.predict(x_eval)[0]

        return (y_pred_val - y_true) ** 2

    def _update_top_bundles(self, mse: float, bundle: List[str], epoch: int):
        self.top_bundles.append((mse, bundle.copy(), epoch))
        self.top_bundles = sorted(self.top_bundles, key=lambda x: x[0])[:5]

    def train(self, n_epochs: int = 300, save: bool = True, save_path: str = None):
        for epoch in range(n_epochs):
            sampled_probs = np.random.beta(self.alpha, self.beta)
            inclusion_tensor = np.random.binomial(1, sampled_probs)
            included_isos = [self.donor_units[i] for i in range(len(self.donor_units)) if inclusion_tensor[i] == 1]
            if not included_isos:
                continue

            errors = Parallel(n_jobs=-1)(
                delayed(self._forecast_for_time)(t, included_isos) for t in self.forecast_times)
            mse = np.mean(errors)
            self.avg_mse_per_epoch.append(mse)

            self._update_top_bundles(mse, included_isos, epoch)

            # Bayesian update
            delta = (self.avg_mse_per_epoch[0] if epoch == 0 else self.avg_mse_per_epoch[epoch - 1]) - mse
            for i, included in enumerate(inclusion_tensor):
                if included:
                    self.alpha[i] += sigmoid(delta)
                    self.beta[i] += sigmoid(-delta)

            # Log posterior means
            for i, iso in enumerate(self.donor_units):
                self.inclusion_probs_by_donor[iso].append(self.alpha[i] / (self.alpha[i] + self.beta[i]))

            if epoch % 10 == 0:
                print(f"Epoch {epoch} — MSE: {mse:.4f}, Bundle Size: {int(np.sum(inclusion_tensor))}")

        if save:
            if save_path is None:
                save_path = f"results_thompson_{self.target_unit}.pkl"
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            results = {
                "avg_mse_per_epoch": self.avg_mse_per_epoch,
                "top_bundles": self.top_bundles,
                "inclusion_probs_by_donor": self.inclusion_probs_by_donor,
                "forecast_times": self.forecast_times,
                "donor_units": self.donor_units,
                "target_unit": self.target_unit
            }
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"✅ Results saved to: {save_path}")

    def plot_learning_curve(self):
        plt.plot(self.avg_mse_per_epoch, label="TS MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Curve (Thompson Sampling)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_donor_probs(self, top_n=10):
        final_scores = {iso: probs[-1] for iso, probs in self.inclusion_probs_by_donor.items()}
        top_donors = sorted(final_scores.items(), key=lambda x: -x[1])[:top_n]

        for iso, _ in top_donors:
            plt.plot(self.inclusion_probs_by_donor[iso], label=iso)

        plt.xlabel("Epoch")
        plt.ylabel("Posterior Inclusion Prob.")
        plt.title(f"Top {top_n} Donor Posterior Probabilities")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
