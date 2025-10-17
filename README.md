# OptimalPanel

**OptimalPanel** is a reinforcement learning-based framework for selecting optimal donor units (bundles) to enhance panel forecasting accuracy.
It learns, through a policy network and reward-driven optimization, to dynamically identify the most informative subset of donors for a given target unit, offering a flexible, data-driven alternative to static pooling or manual donor selection approaches used in synthetic control and related econometric methods.

---
## ⚙️ Installation

### 🧩 Requirements

**Python 3.11** is required. It’s strongly recommended to work within a dedicated **virtual environment**.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# .venv\Scripts\activate    # On Windows
```

### 📦 Install the package

Download and install the latest release wheel from GitHub:
```bash
pip install https://github.com/RenatoVassallo/OptimalPanel/releases/download/v0.1.0/OptimalPanel-0.1.0-py3-none-any.whl
```

### 🧑‍💻 Developer tools (recommended for notebooks & visualization)

To enable examples, plotting, and interactive tutorials, create a `requirements.txt` file with the following contents:

```text
pyarrow>=20.0.0
ipykernel>=6.29.5
ipywidgets>=8.1.6
matplotlib>=3.10.1
seaborn>=0.13.2
tqdm
```

Then install the developer dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📚 Tutorials and Examples

For a detailed tutorial, refer to the following notebook:

* [Simulation Tutorial](./notebooks/Tutorial_Simulation.ipynb): Demonstrates how to simulate panel data with known structural patterns and shocks, and how to apply `OptimalPanel` to forecast outcomes for a target unit using **reinforcement learning**.

---

## 🚀 Quick Start

```python
from OptimalPanel import OptimalBundleRL
import pandas as pd

# Define calibration (training) and testing periods
calibration_periods = pd.date_range(start="2000-01-01", end="2020-01-01", freq="YS")
testing_periods     = pd.date_range(start="2021-01-01", end="2035-01-01", freq="YS")

# Initialize RL optimizer
rl = OptimalBundleRL(
    df=df_sim[['unit', 'year', 'y', 'y_lag', 'x_lag']],
    unit_col='unit',
    time_col='year',
    target_col='y',
    feature_cols=['y_lag', 'x_lag'],
    target_unit='A',
    calibration_periods=calibration_periods,
    testing_periods=testing_periods,
)

# Compute initial donor similarities
initial_similarities = rl.compute_similarities()

# Train the reinforcement learning policy
rl.train(n_epochs=1000, save=False)

# Inspect results
rl.print_top_bundles(top_k=1)
rl.plot_donor_probs(top_n=2)
rl.plot_learning_curve()
```

---

## 🔑 Key Methods

### `compute_similarities(similarity_periods, split_frac=0.6)`
> Computes pairwise similarities between the target unit and potential donors, typically based on out-of-sample RMSE from a Random Forest.

- Filters available donor units
- Initializes the policy network using similarity weights
- Returns a normalized similarity tensor for RL initialization

---

### `train(n_epochs=300, rf_params=None, ar_exo='c', save=True, save_path=None)`
> Trains a reinforcement learning policy to select optimal donor bundles.

- Updates policy network after each rolling forecast
- Uses REINFORCE with entropy regularization
- Tracks MSE by epoch and updates inclusion probabilities
- Evaluates performance on the testing horizon after training

---

### `print_top_bundles(top_k=5)`
> Displays the top-k donor bundles with the lowest mean squared error (MSE).

---

### `plot_learning_curve()`
> Plots the evolution of average MSE over training epochs.

---

### `plot_donor_probs(top_n=5)`
> Visualizes the evolution of inclusion probabilities for the top-N donor units.

---

## 📊 Benchmarks Included

After training, the model evaluates the following benchmarks for comparison:

- **AR(1)** model  
- **Target-only Random Forest**  
- **Full-panel Random Forest**  
- **RL-selected donor bundle Random Forest**  

---

## 📁 Outputs (when `save=True`)
A `.pkl` file with:

- Average MSEs per epoch
- Top-performing donor bundles
- Inclusion probabilities for all donors
- Trained policy network and optimizer state
- Benchmark results across models

---

## 🧠 Motivation

Selecting informative donors is central to panel forecasting.
Instead of relying on fixed heuristics or static pooling, `OptimalPanel` learns a policy that adaptively **adjusts donor weights over time**, exploiting temporal dependencies, cross-unit relationships, and data-driven reward feedback to improve predictive accuracy in dynamic, heterogeneous environments.
