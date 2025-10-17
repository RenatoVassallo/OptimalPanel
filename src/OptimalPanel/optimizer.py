import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Optional, Union
from joblib import Parallel, delayed
from OptimalPanel.summary import generate_rl_summary
from IPython.display import display 

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
    def __init__(self, df: pd.DataFrame, unit_col: str, time_col: str, target_col: str, feature_cols: List[str],
                 target_unit: str, donor_units: Optional[List[str]] = None, similarities: Optional[torch.Tensor] = None,
                 calibration_periods: Union[pd.DatetimeIndex, List[pd.Timestamp]] = None,
                 testing_periods: Union[pd.DatetimeIndex, List[pd.Timestamp]] = None,
                 lr: float = 0.001, seed: int = 42, print_summary: bool = True):
        """
        Reinforcement Learning (RL) framework for identifying optimal donor bundles 
        to enhance panel-based forecasting accuracy.

        This class implements an adaptive policy-learning routine that, using a rolling 
        forecast strategy, learns an optimal combination of donor units for predicting 
        the target unit’s outcome. It leverages cross-unit information, temporal 
        dependencies, and optionally pre-computed similarity structures to dynamically 
        update donor weights across calibration and testing horizons.

        Args:
            df (pd.DataFrame): Panel dataset containing time series for multiple units.
            unit_col (str): Name of the column identifying units (e.g., 'isocode').
            time_col (str): Name of the column identifying time (must be datetime64[ns]).
            target_col (str): Column name of the variable to be forecasted.
            feature_cols (List[str]): List of predictor (explanatory) feature columns.
            target_unit (str): The focal unit for which forecasts are generated (e.g., 'USA').
            donor_units (Optional[List[str]]): Subset of units available as potential donors.
                If None, all non-target units in `df` are considered.
            similarities (Optional[torch.Tensor]): Pre-computed tensor of similarity scores 
                between target and donor units (used for reward shaping or initialization).
            calibration_periods (Union[pd.DatetimeIndex, List[pd.Timestamp]], optional): 
                Time periods used for model training or calibration.
            testing_periods (Union[pd.DatetimeIndex, List[pd.Timestamp]], optional): 
                Time periods used for model evaluation.
            lr (float): Learning rate for the policy optimization algorithm.
            seed (int): Random seed for reproducibility.
            print_summary (bool): Whether to print initialization summary.
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
        self.lr = lr

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

        # after copying df and sorting by time/unit
        self.df_sorted = self.df.sort_values([self.time_col, self.unit_col]).reset_index(drop=True)
        self.units = self.df_sorted[self.unit_col].astype('category').cat.codes.to_numpy()
        self.times = self.df_sorted[self.time_col].to_numpy()
        self.X_all = self.df_sorted[self.feature_cols].to_numpy(dtype='float32')
        self.y_all = self.df_sorted[self.target_col].to_numpy(dtype='float32')

        # boolean masks per unit
        self.unit_masks = {}
        for u in self.df_sorted[self.unit_col].unique():
            self.unit_masks[u] = (self.df_sorted[self.unit_col].values == u)
        
        # build evaluation times from the provided collections ----
        assert calibration_periods is not None and testing_periods is not None, "Provide both calibration_periods and testing_periods."

        # Normalize to sorted unique timestamps
        cal_raw  = pd.to_datetime(pd.Index(calibration_periods)).unique()
        test_raw = pd.to_datetime(pd.Index(testing_periods)).unique()
        cal_raw  = pd.DatetimeIndex(cal_raw).sort_values()
        test_raw = pd.DatetimeIndex(test_raw).sort_values()

        # Keep only timestamps that actually exist for the target unit
        tgt_mask = self.unit_masks[self.target_unit]
        tgt_times = pd.DatetimeIndex(self.df_sorted.loc[tgt_mask, self.time_col].values)

        self.cal_times  = [t for t in cal_raw  if t in tgt_times]
        self.test_times = [t for t in test_raw if t in tgt_times]

        assert len(self.cal_times)  > 0, "No target observations found at calibration_periods."
        assert len(self.test_times) > 0, "No target observations found at testing_periods."
        assert set(self.cal_times).isdisjoint(self.test_times), "Calibration and testing periods must be disjoint."

        # Union of all evaluation times for shared caches
        self.all_eval_times = sorted(set(self.cal_times) | set(self.test_times))

        # ---- Precompute per-time caches ----
        # time < t masks for every eval time
        self.train_masks = {t: (self.times < t) for t in self.all_eval_times}

        # index of target at each eval time
        def _target_index_at(t):
            idxs = np.where((self.times == t) & self.unit_masks[self.target_unit])[0]
            if len(idxs) == 0:
                raise ValueError(f"No target row at time {t}.")
            return int(idxs[0])

        self.test_idx = {t: _target_index_at(t) for t in self.all_eval_times}

        # Speed caches
        self.train_masks_cal  = {t: self.train_masks[t] for t in self.cal_times}
        self.train_masks_test = {t: self.train_masks[t] for t in self.test_times}
        self.test_idx_cal     = {t: self.test_idx[t] for t in self.cal_times}
        self.test_idx_test    = {t: self.test_idx[t] for t in self.test_times}

        if print_summary:
            self._show_summary()

    def _show_summary(self):
        md = generate_rl_summary(self)
        try:
            display(md)  # pretty in notebooks
        except Exception:
            print(md.data)  # plain text fallback
        
    def compute_similarities(self,
                             similarity_periods: Optional[Union[pd.DatetimeIndex, List[pd.Timestamp]]] = None,
                             split_frac: float = 0.60,
                             rf_params: Optional[dict] = None,
                             min_train_obs: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """
        Compute donor similarities via out-of-sample RMSE using a TRAIN/VAL split.

        Default behavior:
            - Use the *training window* = all target timestamps strictly before the earliest
            calibration timestamp (min(self.cal_times)).
            - Split those timestamps into TRAIN (first split_frac) and VAL (remaining).
            - For each donor, train on donor rows with time < first VAL timestamp,
            then score on the TARGET rows at VAL timestamps.

        Custom behavior:
            - If `similarity_periods` is provided (e.g., a pd.date_range), intersect those
            timestamps with the target's available timestamps, then split them using split_frac.

        Args:
            similarity_periods: Optional custom timestamps to define the base window to split.
            split_frac: Fraction of base timestamps used for TRAIN (e.g., 0.60 train / 0.40 val).
            rf_params: RandomForestRegressor params (defaults provided).
            min_train_obs: Minimum donor training rows required.

        Returns:
            (similarities: torch.Tensor, donor_units: List[str])
        """
        # -----------------------------
        # 0) Defaults & guards
        # -----------------------------
        if rf_params is None:
            rf_params = {"n_estimators": 100, "max_depth": 4, "n_jobs": -1, "random_state": 42}
        assert 0.0 < split_frac < 1.0, "split_frac must be in (0,1)."

        forbidden = {'delta', 'rho_i', 'gamma_i'}  # explicit leak guards (keep lag-only features in self.feature_cols)
        leak_cols = forbidden & set(self.df.columns)
        assert not leak_cols, f"Remove forward/oracle columns from df: {sorted(leak_cols)}"

        df = self.df
        unit_col, time_col = self.unit_col, self.time_col
        target_col, feature_cols = self.target_col, self.feature_cols
        target_unit = self.target_unit

        # -----------------------------
        # 1) Build the base timestamp set
        # -----------------------------
        # All target timestamps available in the panel (sorted unique)
        tgt_all_times = pd.DatetimeIndex(
            df.loc[df[unit_col] == target_unit, time_col].drop_duplicates().sort_values()
        )

        if similarity_periods is None:
            if not hasattr(self, "cal_times") or len(self.cal_times) == 0:
                raise ValueError("No calibration periods available to define the training window.")
            cal_start = min(self.cal_times)
            base_times = tgt_all_times[tgt_all_times < cal_start]
            period_label = f"TRAIN window (< {cal_start.date()})"
        else:
            sim_times = pd.to_datetime(pd.Index(similarity_periods))
            base_times = tgt_all_times.intersection(sim_times)
            if len(base_times) == 0:
                raise ValueError("No overlap between target timestamps and similarity_periods.")
            period_label = f"CUSTOM window ({base_times.min().date()} → {base_times.max().date()})"

        if len(base_times) < 3:
            raise ValueError(f"Need at least 3 timestamps in base window; got {len(base_times)}.")

        # -----------------------------
        # 2) Split into TRAIN / VAL
        # -----------------------------
        cut = max(1, int(np.floor(len(base_times) * split_frac)))
        if cut >= len(base_times):  # ensure non-empty VAL
            cut = len(base_times) - 1
        train_times = base_times[:cut]
        val_times   = base_times[cut:]
        val_start   = val_times.min()

        # Target VAL set (features & labels at exact VAL timestamps)
        target_val_df = df[(df[unit_col] == target_unit) & (df[time_col].isin(val_times))]
        if target_val_df.empty:
            raise ValueError("Target VAL set is empty. Adjust split_frac or similarity_periods.")
        X_target_val = target_val_df[feature_cols]
        y_target_val = target_val_df[target_col]

        # -----------------------------
        # 3) Fit donor models on donor TRAIN (< first VAL time) and score on target VAL
        # -----------------------------
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from math import sqrt

        country_rmse = {}
        for iso in df[unit_col].unique():
            if iso == target_unit:
                continue

            donor_train_df = df[(df[unit_col] == iso) & (df[time_col] < val_start)]
            if len(donor_train_df) < min_train_obs:
                print(f"Skipping donor {iso}: too little training data (<{min_train_obs} rows).")
                continue

            X_train = donor_train_df[feature_cols]
            y_train = donor_train_df[target_col]

            try:
                model = RandomForestRegressor(**rf_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_target_val)
                rmse = sqrt(mean_squared_error(y_target_val, y_pred))
                country_rmse[iso] = rmse
            except Exception as e:
                print(f"Skipping donor {iso} due to error: {e}")

        if len(country_rmse) == 0:
            raise ValueError(f"No valid donor similarities found for target {target_unit}.")

        # -----------------------------
        # 4) Convert RMSE → similarity & init policy
        # -----------------------------
        similarity_df = pd.DataFrame.from_dict(country_rmse, orient='index', columns=['RMSE']).sort_values('RMSE')
        donor_isos = similarity_df.index.tolist()
        similarity_scores = 1 / (1 + similarity_df['RMSE'].values)
        sim_tensor = torch.tensor(similarity_scores, dtype=torch.float32).to(self.device)

        self.similarities = sim_tensor
        self.donor_units = donor_isos
        self.policy_net = PolicyNet(input_dim=len(self.donor_units)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.inclusion_probs_by_donor = {iso: [] for iso in self.donor_units}

        print(f"\nTarget: {target_unit} — Initial similarities from {period_label}")
        #print(f"    TRAIN timestamps: {train_times.min().date()} → {train_times.max().date()} ({len(train_times)} pts)")
        #print(f"    VAL   timestamps: {val_times.min().date()} → {val_times.max().date()} ({len(val_times)} pts)")
        for iso, score in zip(donor_isos[:10], similarity_scores[:10]):
            print(f"  {iso}: {score:.4f}")

        return sim_tensor, donor_isos          

    def _forecast_for_time(self, t, included_isos, rf_params=None):
        """
        Train model using target + donor units until time t, forecast t, and return squared error.
        Uses precomputed NumPy arrays for speed.
        """
        if rf_params is None:
            rf_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "n_jobs": 1,          # avoid nested parallelism
                "random_state": 42
            }

        # --- 0) Build training mask: time < t AND (target or donors) ---
        # mask for time < t (cached)
        time_mask = self.train_masks[t]  # boolean array precomputed in __init__

        # mask for target + donors
        unit_mask = self.unit_masks[self.target_unit].copy()
        if included_isos:
            for iso in included_isos:
                unit_mask |= self.unit_masks[iso]

        train_mask = time_mask & unit_mask

        # --- 1) Slice precomputed arrays ---
        X_train = self.X_all[train_mask]
        y_train = self.y_all[train_mask]
        if X_train.shape[0] == 0:
            raise ValueError(f"Empty training set at time {t} for bundle {included_isos}")

        # --- 2) Fit model on train only ---
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)

        # --- 3) Evaluate on exactly one target row at t ---
        # cached index of target at t
        test_idx = self.test_idx[t]  # int index precomputed in __init__
        x_eval = self.X_all[test_idx:test_idx+1]
        y_true = float(self.y_all[test_idx])

        y_pred = float(model.predict(x_eval)[0])
        return (y_pred - y_true) ** 2

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
        print(f"\nTop {top_k} Bundles by MSE:")
        for rank, (mse, bundle, epoch) in enumerate(self.top_bundles[:top_k], 1):
            print(f"{rank}. Epoch {epoch:3d} — MSE: {mse:.4f}, Bundle: {bundle}")
            
    def _compute_benchmarks(self, ar_exo='c'):
        """
        Compute benchmark MSEs on the TEST window:
        - AR(1) on target only
        - RF on target only
        - RF on full panel (target + all donors)
        - RF on fixed best bundle (from calibration)
        Args:
            ar_exo (str): Exogenous variable for AR(1) model, default is 'c' (constant).
        Returns:
            dict with benchmark MSEs
        """
        # Collect per-time errors keyed by t (TEST times only)
        solo_err, full_err, bundle_err, ar1_err = {}, {}, {}, {}

        # Fixed best bundle learned on calibration window
        best_bundle = None
        if getattr(self, "top_bundles", None):
            try:
                best_mse, best_bundle, _ = self.top_bundles[0]
            except Exception:
                best_bundle = None

        # Helper: (kept for safety if you still use df lookups anywhere)
        def _y_true_at(t):
            s = self.df.loc[
                (self.df[self.unit_col] == self.target_unit) &
                (self.df[self.time_col] == t),
                self.target_col
            ]
            return None if s.empty else float(s.iloc[0])

        for t in self.test_times:   # <-- use TEST window only
            # Target-only RF
            try:
                e = self._forecast_for_time(t, included_isos=[])
                if np.isfinite(e):
                    solo_err[t] = float(e)
            except Exception:
                pass

            # Full panel RF
            try:
                e = self._forecast_for_time(t, included_isos=self.donor_units or [])
                if np.isfinite(e):
                    full_err[t] = float(e)
            except Exception:
                pass

            # Fixed best-bundle RF (from calibration)
            if best_bundle:
                try:
                    e = self._forecast_for_time(t, included_isos=best_bundle)
                    if np.isfinite(e):
                        bundle_err[t] = float(e)
                except Exception:
                    pass

            # AR(1) (target only)
            try:
                # You can accelerate this with O(1) updates; keeping statsmodels for clarity
                # NOTE: AR(1) trains on all rows with time < t, which includes calibration
                df_train_t = self.df[(self.df[self.unit_col] == self.target_unit) & (self.df[self.time_col] < t)]
                y_ar = df_train_t[self.target_col].values
                if y_ar.shape[0] >= 3:
                    ar_model = AutoReg(y_ar, lags=1, trend=ar_exo, old_names=False).fit()
                    y_pred = float(ar_model.predict(start=len(y_ar), end=len(y_ar))[0])
                    y_true = _y_true_at(t)
                    if y_true is not None:
                        e = (y_pred - y_true) ** 2
                        if np.isfinite(e):
                            ar1_err[t] = float(e)
            except Exception:
                pass

        # Intersection of times across methods (fair MSEs)
        sets = []
        if ar1_err:   sets.append(set(ar1_err.keys()))
        if solo_err:  sets.append(set(solo_err.keys()))
        if full_err:  sets.append(set(full_err.keys()))
        if best_bundle and bundle_err:
            sets.append(set(bundle_err.keys()))
        common = set.intersection(*sets) if sets else set()

        def mse_at_common(d):
            return float(np.mean([d[k] for k in sorted(common)])) if common else np.nan

        #print(f"# test points used (common): {len(common)} / {len(self.test_times)}")
        print("\nBenchmark MSEs on TEST window:")
        print("- Benchmark MSE (AR(1))       :", mse_at_common(ar1_err))
        print("- Benchmark MSE (Target only) :", mse_at_common(solo_err))
        print("- Benchmark MSE (Full panel)  :", mse_at_common(full_err))
        if best_bundle and bundle_err:
            print("- Benchmark MSE (Fixed bundle):", mse_at_common(bundle_err))
        else:
            print("- Benchmark MSE (Fixed bundle): N/A")

        return {
            "ar1_mse": mse_at_common(ar1_err),
            "solo_rf_mse": mse_at_common(solo_err),
            "full_panel_rf_mse": mse_at_common(full_err),
            "best_bundle_rf_mse": mse_at_common(bundle_err) if best_bundle and bundle_err else np.nan,
            "n_common_test": len(common)
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
        
        print("\nStarting RL training with rolling calibration windows...")

        for epoch in range(n_epochs):
            logits = self.policy_net(self.similarities.unsqueeze(0).to(self.device)).squeeze()
            probs  = torch.sigmoid(logits)
            # optional: floor/ceiling to avoid extremes early on
            probs  = torch.clamp(probs, 0.05, 0.95)

            inclusion_tensor = torch.bernoulli(probs)
            if inclusion_tensor.sum().item() == 0:
                # fallback: force top-1 donor so we don't skip the epoch
                inclusion_tensor[torch.argmax(probs)] = 1.

            included_isos = [self.donor_units[i] for i in range(len(self.donor_units)) if inclusion_tensor[i].item() == 1]

            # Use calibration times for policy learning
            try:
                errors = Parallel(n_jobs=-1)(
                    delayed(self._forecast_for_time)(t, included_isos, rf_params) for t in self.cal_times
                )
                mse = float(np.mean(errors))
            except Exception:
                mse = float('nan')  # never skip logging

            self.avg_mse_per_epoch.append(mse)
            self._update_top_bundles(mse, included_isos, epoch)
            self._update_policy(logits, inclusion_tensor, mse)
            self._log_probs()

            if epoch % 100 == 0:
                bundle_sz = int(inclusion_tensor.sum().item())
                print(f"Epoch {epoch} — Avg MSE (cal): {mse:.4f} | Bundle Size: {bundle_sz}")

        # After policy training on calibration only, evaluate on the TEST window
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
                "cal_times": self.cal_times,
                "test_times": self.test_times,
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
        
        
