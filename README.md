# Crypto Trading DRL Bot

A Deep Reinforcement Learning (DRL) cryptocurrency trading bot implemented using Proximal Policy Optimization (PPO) algorithm with CNN-LSTM neural networks.

## 📋 Overview

This project implements a complete DRL-based cryptocurrency trading system based on the research paper: "Automated Cryptocurrency Trading Bot Implementing DRL". The system uses:

- **CNN-LSTM Neural Networks** for processing time-series market data
- **PPO Algorithm** for stable policy optimization
- **Technical Indicators** (RSI, ATR, OBV) for feature engineering
- **Custom RL Environment** for cryptocurrency trading simulation

## 🚀 Features

- **Real-time Data Acquisition**: Fetches hourly cryptocurrency data from CryptoCompare API
- **Technical Analysis**: Calculates RSI, ATR, and OBV indicators
- **Reinforcement Learning Environment**: Custom Gym environment with Buy/Hold/Sell actions
- **Deep Neural Networks**: CNN-LSTM architecture for both actor and critic networks
- **PPO Implementation**: Complete PPO algorithm with clipping and Generalized Advantage Estimation
- **Risk Management**: Transaction fees, position sizing, and portfolio tracking

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

### 1. Data Collection
```python
from src.data_fetcher import CryptoCompareDataFetcher

fetcher = CryptoCompareDataFetcher()
btc_data = fetcher.get_multiple_days('BTC', 'USD', days=365)
fetcher.save_data(btc_data, 'btc_usd_data')
```

### 2. Feature Engineering
```python
from src.technical_indicators import TechnicalIndicators

ti = TechnicalIndicators()
data_with_indicators = ti.add_technical_indicators(btc_data)
features = ti.prepare_features_for_model(data_with_indicators)
```

### 3. Training
```python
from src.trading_env import CryptoTradingEnv
from src.neural_networks import PPOAgent

# Create environment
env = CryptoTradingEnv(data_with_indicators)

# Create PPO agent
agent = PPOAgent()

# Training loop would go here
# (Implementation of training loop is next step)
```

## 📁 Project Structure

```
crypto-trading-drl/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py          # Data acquisition from CryptoCompare
│   ├── technical_indicators.py  # RSI, ATR, OBV calculations
│   ├── trading_env.py           # Custom RL environment
│   └── neural_networks.py       # CNN-LSTM networks and PPO
├── data/                        # Downloaded market data
├── models/                      # Saved trained models
├── logs/                        # Training logs
├── config/                      # Configuration files
├── test_*.py                    # Test scripts
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

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

Run individual component tests:
```bash
# Test data fetching
python test_data_fetcher.py

# Test technical indicators
python test_technical_indicators.py

# Test trading environment
python test_trading_env.py

# Test neural networks
python test_neural_networks.py
```

## 📈 Results

Based on the research paper, this implementation should achieve:
- **Superior Performance**: Outperforms buy-and-hold strategy
- **Risk Management**: Handles market volatility effectively
- **Adaptability**: Learns optimal trading strategies from data

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
