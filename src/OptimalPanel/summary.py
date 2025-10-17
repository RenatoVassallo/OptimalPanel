from IPython.display import Markdown
import pandas as pd

def generate_rl_summary(model):
    """
    Generate a structured Markdown summary for an OptimalBundleRL instance.
    Displays configuration, data structure, and period setup.
    """
    linkedin_text = "Renato Vassallo"
    linkedin_url = "https://www.linkedin.com/in/renatovassallo"

    # === TIME FREQUENCY ===
    tgt_mask = model.unit_masks[model.target_unit]
    t = pd.DatetimeIndex(model.df_sorted.loc[tgt_mask, model.time_col].sort_values().unique())
    freq = pd.infer_freq(t)
    if freq is None:
        diffs = (t[1:] - t[:-1]).days if len(t) > 1 else []
        freq_str = f"~ every {pd.Series(diffs).mode().iloc[0]} days" if len(diffs) > 0 else "single timestamp"
    else:
        freq_str = freq

    # === TRAINING PERIOD ===
    # Define as all target observations strictly before the first calibration date
    train_mask = model.df_sorted[model.time_col] < min(model.cal_times)
    tgt_train_mask = train_mask & model.unit_masks[model.target_unit]
    if tgt_train_mask.any():
        train_dates = pd.DatetimeIndex(model.df_sorted.loc[tgt_train_mask, model.time_col].unique()).sort_values()
        train_start, train_end = train_dates.min().date(), train_dates.max().date()
        n_train = len(train_dates)
    else:
        train_start = train_end = "N/A"
        n_train = 0

    # === HEADER ===
    summary = f"""
**OptimalPanel Package — Reinforcement Learning for Panel Forecasting**  
Developed by [{linkedin_text}]({linkedin_url}) — Institute for Economic Analysis (IAE-CSIC)  
Version 0.1 — October 2025  

---

**Model Overview**  
- **Framework**: Reinforcement Learning (RL) for optimal donor bundle selection  
- **Device**: {model.device}  
- **Learning Rate**: {model.lr}  

---

**Panel Structure**  
- **Unit Column**: `{model.unit_col}`  
- **Time Column**: `{model.time_col}`  
- **Time Frequency**: {freq_str}  
- **Target Unit**: `{model.target_unit}`  
- **Outcome Variable**: `{model.target_col}`  
- **Feature Variables**: {', '.join(model.feature_cols)}  

---

**Temporal Setup**  
- **Train Period**: {train_start} → {train_end}  ({n_train} obs)  
- **Calib Period**: {min(model.cal_times).date()} → {max(model.cal_times).date()}  ({len(model.cal_times)} obs)  
- **Test Period**: {min(model.test_times).date()} → {max(model.test_times).date()}  ({len(model.test_times)} obs)  
---
"""

    return Markdown(summary)