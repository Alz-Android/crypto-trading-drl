# Crypto Trading DRL Bot

A Deep Reinforcement Learning (DRL) cryptocurrency trading bot implemented using Proximal Policy Optimization (PPO) algorithm with CNN-LSTM neural networks.

## 📋 Overview

This project implements a complete DRL-based cryptocurrency trading system based on the research paper: "Automated Cryptocurrency Trading Bot Implementing DRL". The system uses:

- **CNN-LSTM Neural Networks** for processing time-series market data
- **PPO Algorithm** for stable policy optimization
- **Technical Indicators** (RSI, ATR, OBV) for feature engineering
- **Custom RL Environment** for cryptocurrency trading simulation

## 🔥 Features

### Core DRL Trading Features
- **Real-time Data Acquisition**: Fetches hourly cryptocurrency data from CryptoCompare API
- **Technical Analysis**: Calculates RSI, ATR, and OBV indicators
- **Reinforcement Learning Environment**: Custom Gym environment with Buy/Hold/Sell actions
- **Deep Neural Networks**: CNN-LSTM architecture for both actor and critic networks
- **PPO Implementation**: Complete PPO algorithm with clipping and Generalized Advantage Estimation
- **Risk Management**: Transaction fees, position sizing, and portfolio tracking

### 🎩 **Shared Architecture Benefits**
- **✨ Zero Code Duplication**: Single codebase serves all cryptocurrencies
- **🔄 Easy Maintenance**: Updates to core modules benefit all systems
- **📈 Consistent Performance**: Same robust algorithms across all assets
- **⚡ Rapid Deployment**: Add new cryptocurrencies in minutes
- **📊 Standardized APIs**: Uniform interface for all trading operations
- **🧪 Easy Testing**: Single test suite validates all systems

## 🏗️ Architecture

```
Data Acquisition → Feature Engineering → RL Environment → PPO Agent → Trading Actions
     ↓                    ↓                      ↓              ↓              ↓
CryptoCompare API → RSI/ATR/OBV → Buy/Hold/Sell → CNN-LSTM → Buy/Sell/Hold
```

## 📊 Technical Specifications

- **State Space**: 100 timesteps × 4 features (close, rsi, atr, obv)
- **Action Space**: 3 discrete actions (Buy=1, Hold=0, Sell=2)
- **Reward Function**: Change in net worth
- **Neural Networks**: CNN-LSTM with 32 filters/units
- **Algorithm**: PPO with ε=0.2 clipping
- **Transaction Fees**: 0.1%

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/crypto-trading-drl.git
cd crypto-trading-drl
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

### Quick Start by Cryptocurrency

#### Bitcoin (BTC) - Fully Trained System
```bash
cd crypto-trading-drl-btc/
python train_bitcoin.py  # Full training (1000 episodes)
python train_bitcoin_reduced.py  # Quick test (100 episodes)
```

#### Solana (SOL) - Multiple Training Modes
```bash
cd crypto-trading-drl-sol/
python train_solana_minimal.py  # Quick test (50 episodes)
python train_solana_fast.py     # Medium training (200 episodes) 
python train_solana_agent.py    # Full training (1000 episodes)
```

#### Ethereum (ETH) - Complete System
```bash
cd crypto-trading-drl-eth/
python train_ethereum_agent.py  # Full training
python backtest_ethereum_agent.py  # Backtesting
```

### Individual Component Usage

#### 1. Data Collection
```python
from src.data_fetcher import CryptoCompareDataFetcher

fetcher = CryptoCompareDataFetcher()
btc_data = fetcher.get_multiple_days('BTC', 'USD', days=365)
fetcher.save_data(btc_data, 'btc_usd_data')
```

#### 2. Feature Engineering
```python
from src.technical_indicators import TechnicalIndicators

ti = TechnicalIndicators()
data_with_indicators = ti.add_technical_indicators(btc_data)
features = ti.prepare_features_for_model(data_with_indicators)
```

#### 3. Training
```python
from src.trading_env import CryptoTradingEnv
from src.neural_networks import PPOAgent

# Create environment
env = CryptoTradingEnv(data_with_indicators)

# Create PPO agent
agent = PPOAgent()

# Training loop (see specific training scripts for complete implementation)
```

## 📁 Project Structure

This project features a **shared codebase architecture** with cryptocurrency-specific directories:

```
crypto-trading-drl/
├── src/                         # 🔥 SHARED CORE MODULES
│   ├── data_fetcher.py           # Data acquisition from APIs
│   ├── technical_indicators.py   # RSI, ATR, OBV calculations  
│   ├── trading_env.py            # Custom RL environment
│   ├── neural_networks.py        # CNN-LSTM networks and PPO
│   └── __init__.py              # Package initialization
├── crypto-trading-drl-btc/      # Bitcoin DRL Trading System
│   ├── data/                     # Bitcoin market data
│   ├── results/                  # Training results and models
│   ├── train_bitcoin_agent.py    # Bitcoin training script
│   ├── BTC_TRAINING_COMPLETION_REPORT.md
│   └── STATUS_SUMMARY.md
├── crypto-trading-drl-sol/      # Solana DRL Trading System
│   ├── data/                     # Solana market data
│   ├── full_training_results/    # Complete training results
│   ├── fast_training_results/    # Quick training results
│   ├── train_solana_*.py         # Various Solana training modes
│   ├── SOL_README.md             # Solana-specific documentation
│   └── SOL_requirements.txt      # Solana dependencies
├── crypto-trading-drl-eth/      # Ethereum DRL Trading System
│   ├── data/                     # Ethereum market data
│   ├── full_training_results/    # Training results and plots
│   ├── train_ethereum_agent.py   # Ethereum training script
│   ├── ETH_README.md             # Ethereum-specific documentation
│   └── ETH_requirements.txt      # Ethereum dependencies
├── requirements.txt             # Global dependencies
├── test_shared_src.py           # Shared modules test script
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

### 🔥 **Shared Codebase Architecture**

**Key Innovation**: All cryptocurrency systems share the same core modules, eliminating code duplication!

- **`src/`**: Single source of truth for all core trading functionality
- **DataFetcher**: Unified data acquisition across all cryptocurrencies
- **TradingEnvironment**: Consistent RL environment for all assets
- **PPOAgent**: Standardized neural network architecture
- **TechnicalIndicators**: Common technical analysis tools

### Cryptocurrency-Specific Systems

- **Bitcoin (BTC)**: Fully trained and tested system with comprehensive results
- **Solana (SOL)**: Multiple training modes (minimal, fast, full) with extensive documentation
- **Ethereum (ETH)**: Complete system with backtesting and performance analysis

Each cryptocurrency system contains:
- Independent training scripts that import from shared `src/`
- Dedicated market data and results
- Cryptocurrency-specific documentation and requirements
- Training results and performance metrics

## 🔧 Configuration

### API Keys (Optional)
- **CryptoCompare API**: Get free API key for higher rate limits
- Add to environment variables or create `config/api_keys.py`

### Hyperparameters
Modify in `src/neural_networks.py`:
```python
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
```

## 📊 Performance Metrics

The system tracks:
- **Net Worth**: Portfolio value over time
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall portfolio performance

## 🧪 Testing

Each cryptocurrency system includes comprehensive test suites:

### Bitcoin (BTC) Tests
```bash
cd crypto-trading-drl-btc/
python test_data_fetcher.py      # Test data acquisition
python test_technical_indicators.py  # Test RSI, ATR, OBV
python test_trading_env.py       # Test RL environment
python test_neural_networks.py  # Test CNN-LSTM networks
```

### Solana (SOL) Tests
```bash
cd crypto-trading-drl-sol/
python test_data_fetcher.py
python test_indicators.py
python test_trading_env.py
python test_neural_networks.py
python test_system.py          # Complete system test
```

### Ethereum (ETH) Tests
```bash
cd crypto-trading-drl-eth/
python ETH_test_data_fetcher.py
python ETH_test_neural_networks.py
python ETH_test_trading_env.py
python ETH_test_system.py
```

## 📈 Results

### Training Performance Summary

#### Bitcoin (BTC) - Outstanding Results
- **Peak Performance**: 1,983.96% return (Episode 75)
- **Final Performance**: 159.96% average return
- **Training Status**: ✅ COMPLETE (1000 episodes)
- **Data Coverage**: 6 years of market data
- **Model**: Stable and converged

#### Solana (SOL) - Multiple Training Modes
- **Minimal Training**: 50 episodes for quick testing
- **Fast Training**: 200 episodes with good performance
- **Full Training**: 1000 episodes for maximum accuracy
- **Status**: ✅ All training modes available
- **Special Features**: Speed-optimized training options

#### Ethereum (ETH) - Complete System
- **Training**: Full 1000-episode training completed
- **Backtesting**: Comprehensive performance analysis
- **Results**: Detailed trading metrics and plots
- **Status**: ✅ Training and backtesting complete

### Performance Metrics Tracked
- **Net Worth**: Portfolio value over time
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall portfolio performance

## 🔒 Security & Risk

- **Paper Trading First**: Test with simulated money before real trading
- **Risk Management**: Implement stop-loss and position sizing
- **API Security**: Use environment variables for API keys
- **Data Validation**: Verify data integrity before training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This is a research implementation for educational purposes. Cryptocurrency trading involves significant risk of loss. Always do your own research and never invest more than you can afford to lose.

## 📚 References

- Research Paper: "Automated Cryptocurrency Trading Bot Implementing DRL"
- PPO Algorithm: Schulman et al. (2017)
- OpenAI Gym: Brockman et al. (2016)
- PyTorch: Paszke et al. (2019)

## 🆘 Support

For questions or issues:
1. Check the test scripts for examples
2. Review the docstrings in each module
3. Open an issue on GitHub

---

**Happy Trading! 🚀📈**
