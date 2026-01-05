# PYE Market Formation Simulator

A discrete-time economic model for simulating a Principal-Yield Tokenized (PYE) staking marketplace. The simulator models validator accounts, stochastic reward environments, PT/YT tokenization, secondary market trading, and profit distribution over a configurable time horizon.

## Features

- **Validator Accounts**: Configurable commission structures for inflation, MEV, and fee rewards
- **Stochastic Rewards**: Regime-switching reward process with inflation, MEV, and protocol fees
- **PT/YT Tokenization**: Principal and Yield token minting upon deposit
- **Secondary Market**: Simulated trading with configurable velocity and fee rates
- **Profit Accounting**: Detailed profit breakdown for validators, stakers, and protocol
- **Profit Matrix**: Deposits × Velocity scenario analysis
- **Interactive Dashboard**: Streamlit-based UI for parameter exploration

## Installation

### Requirements

- Python 3.10+
- Conda (recommended) or pip

### Setup

```bash
# Create conda environment
conda create -n pye python=3.11
conda activate pye

# Install dependencies
pip install numpy pandas streamlit plotly openpyxl
```

## Usage

### Interactive Dashboard

```bash
streamlit run dashboard.py
```

Navigate to `http://localhost:8501` in your browser to access the interactive dashboard.

### Command Line

```bash
python pye_simulator.py
```

This runs the default simulation scenario and prints:
- Summary statistics
- Monthly metrics table
- Profit matrices
- Scenario comparison

### Programmatic Usage

```python
from pye_simulator import PYESimulator, SimulationConfig, StochasticRewardProcess

# Configure simulation
config = SimulationConfig(
    num_months=24,
    epochs_per_month=30,
    trading_fee_rate=0.003,  # 30 bps
    fee_share_validators=0.4,
    fee_share_stakers=0.3,
    fee_share_protocol=0.3,
)

# Create reward process
rewards = StochasticRewardProcess(
    base_inflation_rate=0.045,
    base_mev_rate=0.02,
    base_fee_rate=0.01,
)

# Run simulation
simulator = PYESimulator(config=config, reward_process=rewards)
simulator.setup_default_scenario(
    num_validators=5,
    initial_deposits=10_000_000,
)
simulator.run_simulation()

# Get results
stats = simulator.get_summary_statistics()
profit_matrix = simulator.generate_profit_matrix()
```

## Project Structure

```
pye/
├── pye_simulator.py    # Core simulation engine
├── dashboard.py        # Streamlit interactive dashboard
├── simulator.tex       # Mathematical model documentation
└── README.md
```

## Model Overview

The simulator implements the mathematical model described in `simulator.tex`:

### Key Equations

**Yield Accrual** (per epoch):
```
Y_gross = P × (r_inf + r_mev + r_fee)
Y_validator = P × (c_inf × r_inf + c_mev × r_mev + c_fee × r_fee)
Y_staker = Y_gross - Y_validator
```

**Trading Fees**:
```
Fees_m = f × Volume_m
```

**Profit Distribution**:
```
Π_validator = ValYield + π_V × Fees - C_V
Π_staker = StkYield + π_S × Fees - C_S
Π_protocol = π_P × Fees - C_P
```

**Profit Matrix** (24-month horizon):
```
Π_total(D, V) = D × y_24 + f × 24 × V × D - C_24
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_months` | Simulation horizon | 24 |
| `epochs_per_month` | Epochs per month | 30 |
| `trading_fee_rate` | Fee rate on volume | 0.3% |
| `fee_share_validators` | Validator fee share (π_V) | 40% |
| `fee_share_stakers` | Staker fee share (π_S) | 30% |
| `fee_share_protocol` | Protocol fee share (π_P) | 30% |
| `base_trading_velocity` | Monthly turnover rate | 15% |

## Dashboard Features

The Streamlit dashboard provides:

1. **Parameter Controls**: Adjust all simulation parameters via sidebar
2. **Key Metrics**: Summary statistics cards
3. **Trading Volume Chart**: Monthly PT/YT volume breakdown
4. **Trading Fees Chart**: Monthly and cumulative fees
5. **Profit Distribution Chart**: Stacked area chart by participant
6. **Profit Matrix**: Interactive heatmap with Deposits × Velocity
7. **Data Export**: Download CSV/Excel reports

## License

Copyright (c) 2024 Kosmos Ventures. All rights reserved.
