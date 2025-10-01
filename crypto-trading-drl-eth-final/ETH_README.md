# Ethereum Trading DRL Bot

A Deep Reinforcement Learning (DRL) cryptocurrency trading bot implemented using Proximal Policy Optimization (PPO) algorithm with CNN-LSTM neural networks, specifically optimized for Ethereum (ETH) trading.

## 📋 Overview

This project implements a complete DRL-based Ethereum trading system based on the research paper: "Automated Cryptocurrency Trading Bot Implementing DRL". The system uses:

- **CNN-LSTM Neural Networks** for processing time-series market data
- **PPO Algorithm** for stable policy optimization
- **Technical Indicators** (RSI, ATR, OBV) for feature engineering
- **Custom RL Environment** for Ethereum trading simulation

## 🚀 Features

- **Real-time Data Acquisition**: Fetches hourly Ethereum price data from CryptoCompare API
- **Technical Analysis**: Calculates RSI, ATR, and OBV indicators optimized for ETH
- **Reinforcement Learning Environment**: Custom Gym environment with Buy/Hold/Sell actions
- **Deep Neural Networks**: CNN-LSTM architecture for both actor and critic networks
- **PPO Implementation**: Complete PPO algorithm with clipping and Generalized Advantage Estimation
- **Risk Management**: Transaction fees, position sizing, and portfolio tracking
- **Full Training Mode**: 1000 episodes with 6 years of historical data

## 🏗️ Architecture

```
ETH Data Acquisition → Feature Engineering → RL Environment → PPO Agent → Trading Actions
        ↓                    ↓                      ↓              ↓              ↓
CryptoCompare API → RSI/ATR/OBV → Buy/Hold/Sell → CNN-LSTM → Buy/Sell/Hold
```

## 📊 Technical Specifications

- **Cryptocurrency**: Ethereum (ETH/USD)
- **Training Episodes**: 1000 (full training configuration)
- **State Space**: 100 timesteps × 4 features (close, rsi, atr, obv)
- **Action Space**: 3 discrete actions (Buy=1, Hold=0, Sell=2)
- **Reward Function**: Change in net worth
- **Neural Networks**: CNN-LSTM with 32 filters/units
- **Algorithm**: PPO with ε=0.2 clipping
- **Transaction Fees**: 0.1%
- **Initial Balance**: $10,000

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Alz-Android/crypto-trading-drl-eth.git
cd crypto-trading-drl-eth
```

2. **Create virtual environment**:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### 1. Data Preparation
First, ensure you have 6 years of Ethereum historical data in `data/ETH_6year_data.csv` with columns:
- `date`: Date timestamp
- `open_usd`: Opening price in USD
- `high_usd`: High price in USD
- `low_usd`: Low price in USD
- `close_usd`: Closing price in USD
- `volume`: Trading volume

### 2. Full Training
```bash
python train_ethereum_agent.py
```

This will:
- Load 6 years of ETH historical data
- Train for 1000 episodes with comprehensive monitoring
- Save models every 50 episodes
- Generate training progress plots
- Save the best performing model

### 3. Backtesting
```bash
python backtest_ethereum_agent.py
```

This will:
- Load the best trained model
- Test on validation, recent, and holdout periods
- Compare against buy-and-hold strategy
- Generate performance reports and plots

## 📁 Project Structure

```
crypto-trading-drl-eth/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py          # ETH data acquisition from CryptoCompare
│   ├── technical_indicators.py  # RSI, ATR, OBV calculations
│   ├── trading_env.py           # Custom RL environment for ETH
│   └── neural_networks.py       # CNN-LSTM networks and PPO
├── data/                        # ETH historical data (6 years)
│   └── ETH_6year_data.csv      # Required ETH data file
├── full_training_results/       # Training results and models
│   ├── models/                  # Saved trained models
│   ├── plots/                   # Training progress plots
│   └── backtest_plots/          # Backtest result plots
├── train_ethereum_agent.py     # Main training script
├── backtest_ethereum_agent.py  # Backtesting script
├── test_*.py                    # Test scripts
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration

### Training Configuration
In `train_ethereum_agent.py`, modify the `TrainingConfig` class:
```python
class TrainingConfig:
    CRYPTO_SYMBOL = 'ETH'
    DATA_FILE = 'data/ETH_6year_data.csv'
    TOTAL_EPISODES = 1000
    EPISODES_PER_UPDATE = 10
    MAX_EPISODE_LENGTH = 1000
    LOOKBACK_WINDOW = 100
    INITIAL_BALANCE = 10000
```

### Neural Network Hyperparameters
In `src/neural_networks.py`:
```python
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
```

## 📊 Expected Performance

Based on full training configuration (1000 episodes, 6-year data):
- **Comprehensive Learning**: Uses complete 6-year Ethereum price history
- **Stable Performance**: 1000 episodes provide robust policy learning
- **Market Adaptation**: Learns from bull/bear cycles and volatility
- **Risk Management**: Balanced Buy/Hold/Sell action selection

## 🧪 Testing

Run component tests:
```bash
# Test data fetching
python src/data_fetcher.py

# Test technical indicators
python src/technical_indicators.py

# Test trading environment
python src/trading_env.py

# Test neural networks
python src/neural_networks.py
```

## 📈 Training Progress

The training process includes:
1. **Data Loading**: 6 years of ETH historical data
2. **Feature Engineering**: RSI, ATR, OBV calculation
3. **Environment Setup**: Custom Gym environment
4. **PPO Training**: 1000 episodes with validation
5. **Model Saving**: Best model based on validation performance
6. **Progress Visualization**: Real-time training plots

## 🎯 Backtesting Results

The backtesting evaluates performance on:
- **Validation Period**: Last 20% of training data
- **Recent Period**: Last 1 year of data
- **Holdout Period**: Last 6 months (unseen data)

Metrics include:
- Total return percentage
- Comparison vs buy-and-hold
- Trading action distribution
- Price statistics and volatility handling

## 🔒 Security & Risk

- **Paper Trading**: All trading is simulated with historical data
- **Risk Management**: Built-in transaction fees and position limits
- **Data Integrity**: Comprehensive data validation
- **Model Validation**: Multiple test periods for robust evaluation

## 🚀 Getting Started

1. **Prepare Data**: Ensure `data/ETH_6year_data.csv` exists with proper format
2. **Start Training**: Run `python train_ethereum_agent.py`
3. **Monitor Progress**: Check `full_training_results/plots/training_progress.png`
4. **Evaluate Results**: Run `python backtest_ethereum_agent.py`
5. **Analyze Performance**: Review backtest plots and reports

## ⚠️ Disclaimer

This is a research implementation for educational purposes. Cryptocurrency trading involves significant risk of loss. This system:
- Uses only historical data for training and testing
- Does not provide financial advice
- Should not be used for actual trading without extensive additional validation
- Results are not guaranteed for future performance

## 📚 References

- Research Paper: "Automated Cryptocurrency Trading Bot Implementing DRL"
- PPO Algorithm: Schulman et al. (2017)
- Ethereum: ethereum.org
- OpenAI Gym: Brockman et al. (2016)
- PyTorch: Paszke et al. (2019)

## 🆘 Support

For questions or issues:
1. Check the component test scripts for examples
2. Review training logs in `full_training_results/`
3. Examine backtest reports for performance insights
4. Open an issue on GitHub

---

**Happy ETH Trading with DRL! 🚀📈**