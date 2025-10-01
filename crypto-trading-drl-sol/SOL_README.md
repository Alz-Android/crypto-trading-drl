# Solana Deep Reinforcement Learning Trading Bot

A sophisticated Deep Reinforcement Learning (DRL) trading system specifically designed for Solana (SOL) trading using Proximal Policy Optimization (PPO) with CNN-LSTM neural networks.

## Features

- **Advanced DRL Agent**: PPO (Proximal Policy Optimization) algorithm with CNN-LSTM architecture
- **Technical Indicators**: RSI, ATR, OBV integrated for enhanced market analysis
- **Multiple Training Modes**: 
  - Full training (1000 episodes)
  - Fast training (200 episodes) 
  - Minimal training (50 episodes)
- **Comprehensive Backtesting**: Multiple time period testing with both stochastic and deterministic strategies
- **Risk Management**: Transaction fees, position sizing, and portfolio management
- **Real-time Data**: Integration with CryptoCompare API for live Solana data
- **Extensive Logging**: Detailed training progress and performance tracking

## System Architecture

### Core Components

1. **Trading Environment** (`src/trading_env.py`)
   - Custom OpenAI Gym environment for Solana trading
   - Technical indicators integration
   - Portfolio management and risk controls

2. **Neural Networks** (`src/neural_networks.py`)
   - CNN-LSTM architecture for price pattern recognition
   - PPO agent implementation
   - Advanced memory management

3. **Data Management** (`src/data_fetcher.py`)
   - CryptoCompare API integration
   - Historical data preprocessing
   - Technical indicators calculation

4. **Technical Analysis** (`src/technical_indicators.py`)
   - RSI (Relative Strength Index)
   - ATR (Average True Range)
   - OBV (On-Balance Volume)

## Quick Start

### Installation

```bash
pip install -r SOL_requirements.txt
```

### Training Options

#### Full Training (1000 episodes)
```bash
python train_solana_agent.py
```

#### Fast Training (200 episodes)
```bash
python train_solana_fast.py
```

#### Minimal Training (50 episodes)
```bash
python train_solana_minimal.py
```

### Running Backtest

```bash
python backtest_solana_agent.py
```

## Configuration

### Training Parameters

- **Full Training**: 1000 episodes, comprehensive learning
- **Fast Training**: 200 episodes, quick validation
- **Minimal Training**: 50 episodes, rapid testing
- **Learning Rate**: 0.0003
- **Batch Size**: 64
- **Lookback Window**: 100 days
- **Initial Balance**: $10,000
- **Transaction Fee**: 0.1%

### Data Configuration

- **Symbol**: SOL (Solana)
- **Data Source**: CryptoCompare API
- **Training Period**: 6 years of historical data
- **Data Frequency**: Daily OHLCV data

## Results Structure

```
crypto-trading-drl-sol/
├── full_training_results/
│   ├── sol_best_model.pth
│   ├── sol_final_model.pth
│   ├── training_metrics.json
│   └── backtest_plots/
├── fast_training_results/
│   ├── training_metrics.json
│   └── plots/
├── minimal_training_results/
│   └── training_metrics.json
├── data/
│   └── sol_training_data.csv
└── src/
    ├── neural_networks.py
    ├── trading_env.py
    ├── data_fetcher.py
    └── technical_indicators.py
```

## Performance Metrics

The system tracks comprehensive metrics:

- **Portfolio Returns**: Percentage gains/losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Action Distribution**: Hold/Buy/Sell percentages
- **Benchmark Comparison**: vs Buy & Hold strategy

## Technical Specifications

### State Space
- Price data (OHLCV)
- Technical indicators (RSI, ATR, OBV)
- Portfolio status (cash, holdings, value)
- Market volatility measures

### Action Space
- 0: Hold (maintain current position)
- 1: Buy (purchase Solana with available cash)
- 2: Sell (sell Solana holdings)

### Reward Function
- Portfolio value change
- Risk-adjusted returns
- Transaction cost penalties

## Advanced Features

### Multiple Training Modes
- **Full Training**: Comprehensive 1000-episode training
- **Fast Training**: Quick 200-episode validation
- **Minimal Training**: Rapid 50-episode testing

### Stochastic vs Deterministic Trading
- **Stochastic**: Samples actions from probability distribution
- **Deterministic**: Always selects highest probability action
- Comparison analysis included in backtest results

### Risk Management
- Position sizing based on available capital
- Transaction fee incorporation
- Portfolio rebalancing constraints

## Monitoring and Analysis

### Training Monitoring
- Real-time training progress logging
- Episode reward tracking
- Action distribution analysis
- Model checkpoint saving

### Backtest Analysis
- Multiple time period testing
- Strategy comparison (stochastic vs deterministic)
- Performance visualization
- Detailed CSV exports

## Testing

Run the comprehensive test suite:

```bash
python test_system.py
python test_data_fetcher.py
python test_neural_networks.py
python test_trading_env.py
python test_technical_indicators.py
```

## Troubleshooting

### Common Issues

1. **Data Fetching Errors**
   - Check internet connection
   - Verify CryptoCompare API accessibility
   - Ensure sufficient historical data

2. **Training Issues**
   - Monitor GPU/CPU usage
   - Check memory availability
   - Verify model saving permissions

3. **Backtest Failures**
   - Ensure trained model exists
   - Check test data availability
   - Verify file permissions

## License

This project is for educational and research purposes. Use at your own risk for live trading.

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.