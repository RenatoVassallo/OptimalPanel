# OptimalPanel

**OptimalPanel** is a reinforcement learning-based tool for selecting optimal donor units (bundles) to improve panel forecasting accuracy. It leverages a policy network and reward-driven learning to dynamically identify informative subsets of units, providing a data-driven alternative to manual donor selection in synthetic control and related models.

---

## 📚 Tutorials and Examples

For a detailed tutorial, refer to the following notebook:

* [Simulation Tutorial](./notebooks/Tutorial_Simulation.ipynb): demonstrates how to simulate synthetic panel data with known structural patterns and shocks, and then apply `OptimalPanel` to forecast outcomes for a target unit.

---

## 🚀 Quick Start

```python
from OptimalPanel.optimizer import OptimalBundleRL
import pandas as pd

# Define forecasting steps (e.g. yearly from 2005 to 2014)
forecast_times = list(pd.date_range(start="2005-01-01", end="2014-01-01", freq="YS"))
total_epochs = 2000

# Initialize optimizer
rl = OptimalBundleRL(
    df=df_sim,
    unit_col='unit',
    time_col='year',
    target_col='y',
    feature_cols=['y_lag', 'x1', 'x2', 'x3'],
    target_unit='A',
    forecast_times=forecast_times
)

# Compute donor similarities based on out-of-sample RMSE
rl.compute_similarities(
    test_start=pd.to_datetime("1998-01-01"),
    test_end=pd.to_datetime("2004-01-01")
)

# Train the reinforcement learning policy
rl.train(
    n_epochs=total_epochs,
    save=False,
    save_path="results/annual/results_A.pkl"
)

# View top bundles and learning dynamics
rl.print_top_bundles(top_k=2)
rl.plot_learning_curve()
rl.plot_donor_probs(top_n=2)
```

---

## 🔑 Key Methods

### `compute_similarities(test_start, test_end)`
> Computes donor similarities using out-of-sample RMSE from a Random Forest.

- Filters potential donor units
- Initializes policy network based on similarity weights

---

### `train(n_epochs=300, rf_params=None, save=True, save_path=None)`
> Trains a reinforcement learning policy to select optimal donor bundles.

- Uses REINFORCE with entropy regularization
- Tracks top-performing bundles
- Computes benchmarks for comparison

---

### `print_top_bundles(top_k=5)`
> Display the top-k lowest-MSE bundles across training.

---

### `plot_learning_curve()`
> (Add-on: plot average MSE over training epochs.)

---

### `plot_donor_probs(top_n=5)`
> (Add-on: plot donor unit inclusion probabilities over time.)

---

## 📊 Benchmarks Included

After training, the following models are compared:

- **AR(1)** model  
- **Target-only Random Forest**  
- **Full-panel Random Forest**  
- **Best bundle Random Forest (RL-selected)**  

---

## 📁 Outputs (when `save=True`)
A `.pkl` file with:

- MSEs, top bundles, donor inclusion probabilities  
- Trained policy network and optimizer state  
- Benchmark performance  

---

## 🧠 Motivation

Forecasting panel data often requires selecting informative donor units. Rather than relying on static heuristics or arbitrary similarity metrics, `OptimalPanel` learns an inclusion policy that adapts over time and is tuned for predictive accuracy.
