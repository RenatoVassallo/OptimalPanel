import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import List, Tuple, Optional
from joblib import Parallel, delayed
import pickle
import os
import time 

class PolicyNet(nn.Module):
    def __init__(self, input_dim):
        """
        Simple linear policy network initialized to identity for stable start.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.linear.weight.data = torch.eye(input_dim)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        return self.linear(x)

class OptimalBundleRL:
    """
    Reinforcement Learning routine to find optimal unit bundles for panel forecasting.
    """
    def __init__(self, df: pd.DataFrame, unit_col: str, time_col: str, target_col: str, feature_cols: List[str],
                 target_unit: str, donor_units: Optional[List[str]] = None, similarities: Optional[torch.Tensor] = None,
                 forecast_times: List[pd.Timestamp] = None, lr: float = 0.001, seed: int = 42):
        """
        Args:
            df: DataFrame containing panel data.
            unit_col: Column with unit identifiers (e.g., 'isocode').
            time_col: Column with time as datetime64[ns] (e.g., 'date').
            target_col: Column to be forecasted.
            feature_cols: Predictor feature columns.
            target_unit: The focal unit (e.g., 'USA').
            donor_units: (Optional) List of possible donor units.
            similarities: (Optional) Tensor of similarity scores between target and donors.
            forecast_times: List of pd.Timestamp values (rolling forecast steps).
            lr: Learning rate for policy network.
            seed: Random seed for reproducibility.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        assert pd.api.types.is_datetime64_any_dtype(df[time_col]), f"{time_col} must be a datetime column (e.g., pd.Timestamp)."

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = df.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.target_unit = target_unit
        self.forecast_times = forecast_times
        self.lr = lr  # Store for reuse

        if similarities is not None and donor_units is not None:
            self.similarities = similarities.to(self.device)
            self.donor_units = donor_units
            self.policy_net = PolicyNet(input_dim=len(self.donor_units)).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
            self.inclusion_probs_by_donor = {iso: [] for iso in self.donor_units}
        else:
            self.similarities = None
            self.donor_units = None
            self.policy_net = None
            self.optimizer = None
            self.inclusion_probs_by_donor = {}

        self.baseline = 0.0
        self.avg_mse_per_epoch = []
        self.top_bundles: List[Tuple[float, List[str], int]] = []
        self.benchmarks = {}
        
    
    def compute_similarities(self, test_start, test_end, rf_params=None):
        """
        Compute donor similarities for a forecasting task using out-of-sample RMSE.

        This method fits a RandomForest model on each potential donor unit's 
        data prior to `test_start` and evaluates its predictive accuracy 
        on the target unit over the specified test window.

        Donors must have at least 5 training observations before `test_start` 
        to be considered valid.
        
        Args:
            test_start: pd.Timestamp for testing start (e.g., pd.to_datetime("2001-01-01")).
            test_end: pd.Timestamp for testing end (e.g., pd.to_datetime("2009-12-01")).
            rf_params: Parameters for RandomForestRegressor (optional).
            
        Returns:
            similarities: torch.Tensor of similarity scores
            donor_units: List of donor unit identifiers (excluding target unit)
        """
        if rf_params is None:
            rf_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "n_jobs": -1,
                "random_state": 42
            }
        df = self.df
        unit_col = self.unit_col
        time_col = self.time_col
        target_col = self.target_col
        feature_cols = self.feature_cols
        target_unit = self.target_unit

        target_test_df = df[(df[unit_col] == target_unit) & (df[time_col].between(test_start, test_end))]
        X_target_test = target_test_df[feature_cols]
        y_target_test = target_test_df[target_col]

        country_rmse = {}

        for iso in df[unit_col].unique():
            if iso == target_unit:
                continue

            train_df = df[(df[unit_col] == iso) & (df[time_col] < test_start)]
            if len(train_df) < 5:
                print(f"  ⛔ Skipping donor {iso}: too little training data.")
                continue
            if len(X_target_test) == 0:
                print(f"  ⛔ Target test set is empty. Check feature columns or filtering.")
                continue

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]

            model = RandomForestRegressor(**rf_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_target_test)
            rmse = sqrt(mean_squared_error(y_target_test, y_pred))
            country_rmse[iso] = rmse

        if len(country_rmse) < 1:
            raise ValueError(f"⚠️ No valid donor similarities found for target {target_unit}.")

        similarity_df = pd.DataFrame.from_dict(country_rmse, orient='index', columns=['RMSE']).sort_values(by='RMSE')
        donor_isos = similarity_df.index.tolist()
        similarity_scores = 1 / (1 + similarity_df['RMSE'].values)
        sim_tensor = torch.tensor(similarity_scores, dtype=torch.float32).to(self.device)
        
        self.similarities = sim_tensor
        self.donor_units = donor_isos
        self.policy_net = PolicyNet(input_dim=len(self.donor_units)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.inclusion_probs_by_donor = {iso: [] for iso in self.donor_units}

        print(f"\n🔍 Target: {target_unit} — Initial Similarity Vector:")
        for iso, score in zip(donor_isos[:10], similarity_scores[:10]):
            print(f"  {iso}: {score:.4f}")

        return sim_tensor, donor_isos
            

    def _forecast_for_time(self, t, included_isos, rf_params=None):
        """
        Trains model using target + donor units until time t, forecasts t, and returns squared error.
        """
        if rf_params is None:
            rf_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "n_jobs": -1,
                "random_state": 42
            }

        df_train = self.df[self.df[self.time_col] < t]
        train_rows = df_train[df_train[self.unit_col] == self.target_unit]
        for iso in included_isos:
            train_rows = pd.concat([train_rows, df_train[df_train[self.unit_col] == iso]])

        X_train = train_rows[self.feature_cols].values
        y_train = train_rows[self.target_col].values

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)

        x_eval = self.df[(self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] == t)][self.feature_cols].values
        y_true = self.df[(self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] == t)][self.target_col].values[0]

        y_pred_val = model.predict(x_eval)[0]
        return (y_pred_val - y_true) ** 2

    def _update_top_bundles(self, mse: float, bundle: List[str], epoch: int):
        """
        Track top 5 bundles with lowest MSE over training.
        """
        self.top_bundles.append((mse, bundle.copy(), epoch))
        self.top_bundles = sorted(self.top_bundles, key=lambda x: x[0])[:5]

    def _update_policy(self, logits: torch.Tensor, inclusion_tensor: torch.Tensor, avg_mse: float):
        """
        Update the policy network using REINFORCE rule with entropy regularization (based on MSE).
        """
        reward = -avg_mse
        self.baseline = 0.95 * self.baseline + 0.05 * reward
        advantage = reward - self.baseline

        probs = torch.sigmoid(logits)
        log_probs = torch.log(probs + 1e-8) * inclusion_tensor
        policy_loss = -log_probs.sum() * advantage

        policy_loss += 0.01 * inclusion_tensor.sum()  # L1 regularization
        entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).sum()
        policy_loss -= 0.1 * entropy

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def _log_probs(self):
        """
        Log inclusion probabilities for each donor after epoch update.
        """
        with torch.no_grad():
            final_probs = torch.sigmoid(self.policy_net(self.similarities.unsqueeze(0).to(self.device)).squeeze())
            for i, iso in enumerate(self.donor_units):
                self.inclusion_probs_by_donor[iso].append(final_probs[i].item())
    
    def print_top_bundles(self, top_k: int = 5):
        """
        Print the top-k bundles with the lowest MSE.
        """
        print(f"\n🔹 Top {top_k} Bundles by MSE:")
        for rank, (mse, bundle, epoch) in enumerate(self.top_bundles[:top_k], 1):
            print(f"{rank}. Epoch {epoch:3d} — MSE: {mse:.4f}, Bundle: {bundle}")
            
    def _compute_benchmarks(self, ar_exo='c'):
        """
        Compute MSE for: AR(1), Solo RF, Full Panel RF, Best Bundle RF.
        """
        solo_errors, full_errors, bundle_errors, ar1_errors = [], [], [], []

        best_mse, best_bundle, _ = self.top_bundles[0]

        for t in self.forecast_times:
            # --- Solo RF ---
            solo_errors.append(self._forecast_for_time(t, included_isos=[]))  # only target used

            # --- Full Panel RF ---
            full_errors.append(self._forecast_for_time(t, included_isos=self.donor_units))

            # --- Best Bundle RF ---
            if best_bundle:
                bundle_errors.append(self._forecast_for_time(t, included_isos=best_bundle))

            # --- AR(1) ---
            df_train = self.df[(self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] < t)]
            y_ar = df_train[self.target_col].values
            if len(y_ar) >= 3:
                ar_model = AutoReg(y_ar, lags=1, trend=ar_exo, old_names=False).fit()
                y_pred = ar_model.predict(start=len(y_ar), end=len(y_ar))[0]
                y_true = self.df[
                    (self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] == t)
                ][self.target_col].values[0]
                ar1_errors.append((y_pred - y_true) ** 2)

        print("- Benchmark MSE (AR(1)):", np.mean(ar1_errors))
        print("- Benchmark MSE (Target only):", np.mean(solo_errors))
        print("- Benchmark MSE (Full panel):", np.mean(full_errors))
        print("- Benchmark MSE (Best bundle):", np.mean(bundle_errors))
        
        return {
            "ar1_mse": np.mean(ar1_errors),
            "solo_rf_mse": np.mean(solo_errors),
            "full_panel_rf_mse": np.mean(full_errors),
            "best_bundle_rf_mse": np.mean(bundle_errors)
        }
            
    def train(self, n_epochs: int = 300, rf_params=None, ar_exo='c', save: bool = True, save_path: str = None):
        """
        Run RL training to identify optimal donor bundles over epochs.
        
        Args:
            n_epochs (int): Number of training epochs.
            rf_params (dict, optional): Parameters for RandomForestRegressor.
            ar_exo (str): Exogenous variable for AR(1) model, default is 'c' (constant).
            save (bool): Whether to save training results.
            save_path (str, optional): Path to save results. Defaults to 'results_{target_unit}.pkl'.
        """
        # Validate that similarities and donor_units are set
        if self.similarities is None or self.donor_units is None:
            raise ValueError("Similarities and donor units must be computed before training. Call compute_similarities() first.")
        
        for epoch in range(n_epochs):
            #print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            logits = self.policy_net(self.similarities.unsqueeze(0).to(self.device)).squeeze()
            probs = torch.sigmoid(logits)
            inclusion_tensor = torch.bernoulli(probs)

            included_isos = [
                self.donor_units[i] for i in range(len(self.donor_units))
                if inclusion_tensor[i].item() == 1
            ]
            if not included_isos:
                continue

            errors = Parallel(n_jobs=-1)(delayed(self._forecast_for_time)(t, included_isos, rf_params) for t in self.forecast_times)
            mse = np.mean(errors)
            self.avg_mse_per_epoch.append(mse)

            self._update_top_bundles(mse, included_isos, epoch)
            self._update_policy(logits, inclusion_tensor, mse)
            self._log_probs()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} — Avg MSE: {mse:.4f}, Bundle Size: {int(inclusion_tensor.sum().item())}")

        self.benchmarks = self._compute_benchmarks(ar_exo=ar_exo)

        # Save results
        if save:
            if save_path is None:
                save_path = f"results_{self.target_unit}.pkl"
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            results = {
                "avg_mse_per_epoch": self.avg_mse_per_epoch,
                "top_bundles": self.top_bundles,
                "inclusion_probs_by_donor": self.inclusion_probs_by_donor,
                "policy_net_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "forecast_times": self.forecast_times,
                "donor_units": self.donor_units,
                "target_unit": self.target_unit,
                "benchmarks": self.benchmarks
            }

            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"- Training results saved to: {save_path}")
        else:
            print("- Training completed without saving results.")


    def plot_learning_curve(self):
        """Plot MSE over training epochs."""
        plt.plot(self.avg_mse_per_epoch, label="RL MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Curve")
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_donor_probs(self, top_n=10):
        """
        Plot evolution of inclusion probabilities for top-N donors.
        """
        final_scores = {iso: probs[-1] for iso, probs in self.inclusion_probs_by_donor.items()}
        top_donors = sorted(final_scores.items(), key=lambda x: -x[1])[:top_n]

        for iso, _ in top_donors:
            plt.plot(self.inclusion_probs_by_donor[iso], label=iso)

        plt.xlabel("Epoch")
        plt.ylabel("Inclusion Probability")
        plt.title(f"Top {top_n} Donor Inclusion Probabilities")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend outside
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to fit legend
        plt.show()
        
        
