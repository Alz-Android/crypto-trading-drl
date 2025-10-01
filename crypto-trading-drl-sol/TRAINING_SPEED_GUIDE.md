# Training Time Optimization Guide

## 🚀 Training Options (Local Machine)

### 1. 🔬 MINIMAL Training - `train_solana_minimal.py`
**⏱️ Time: 5-10 minutes**
- **Episodes**: 50 (vs 1000)
- **Episode Length**: 100 steps (vs 1000)
- **Lookback Window**: 30 days (vs 100)
- **Data**: Last 6 months only (vs 6 years)
- **Purpose**: Quick system test, verify everything works

### 2. ⚡ FAST Training - `train_solana_fast.py`
**⏱️ Time: 15-30 minutes**
- **Episodes**: 200 (vs 1000)
- **Episode Length**: 300 steps (vs 1000)
- **Lookback Window**: 50 days (vs 100)
- **Data**: Last 2 years only (vs 6 years)
- **Purpose**: Good balance of speed and quality

### 3. 🎯 FULL Training - `train_solana_agent.py`
**⏱️ Time: 2-4 hours (CPU)**
- **Episodes**: 1000
- **Episode Length**: 1000 steps
- **Lookback Window**: 100 days
- **Data**: Full 6 years
- **Purpose**: Maximum performance, publication-quality

## 📊 Speed Optimization Techniques Applied

### 1. **Reduced Episodes**
- Minimal: 50 episodes (20x faster)
- Fast: 200 episodes (5x faster)
- Original: 1000 episodes

### 2. **Shorter Episodes**
- Minimal: 100 steps (10x faster)
- Fast: 300 steps (3.3x faster)
- Original: 1000 steps

### 3. **Smaller Lookback Window**
- Minimal: 30 days (3.3x less memory/computation)
- Fast: 50 days (2x less memory/computation)
- Original: 100 days

### 4. **Limited Data**
- Minimal: 6 months (12x less data)
- Fast: 2 years (3x less data)
- Original: 6 years

## 🎯 Recommended Approach

### For First-Time Testing:
```bash
python train_solana_minimal.py    # 5-10 minutes
```
Verify the system works end-to-end

### For Development/Iteration:
```bash
python train_solana_fast.py       # 15-30 minutes
```
Good results with reasonable training time

### For Production/Best Results:
```bash
python train_solana_agent.py      # 2-4 hours
```
Maximum performance using full dataset

## 💡 Additional Speed Tips

### 1. **Close Other Applications**
- Free up CPU and RAM
- Close browsers, video players, etc.

### 2. **Power Settings**
- Set Windows to "High Performance" mode
- Ensure laptop is plugged in

### 3. **Process Priority**
- Task Manager → Details → python.exe → Set Priority → High

### 4. **Monitor Progress**
The scripts show progress every few episodes:
```
Episodes 1-10:
  Average Reward: $45.32
  Average Return: 2.34%
  Average Length: 95.2
```

### 5. **Early Stopping**
You can stop training anytime with `Ctrl+C` and still get a saved model

## 🚀 GPU Acceleration (Future)

If you get a GPU later:
- Training time drops to 10-30 minutes (10x faster)
- Install CUDA-enabled PyTorch
- System automatically detects and uses GPU

## 📈 Performance Expectations

### Minimal (5-10 min):
- Basic learning, proof of concept
- May not beat buy-and-hold
- Good for testing

### Fast (15-30 min):
- Decent learning with reasonable patterns
- May beat buy-and-hold on some periods
- Good for development

### Full (2-4 hours):
- Best possible performance
- Highest chance of beating buy-and-hold
- Production-ready results

## 🏁 Ready to Start!

Choose your training speed based on your needs:
- **Just testing**: `python train_solana_minimal.py`
- **Good balance**: `python train_solana_fast.py`
- **Best results**: `python train_solana_agent.py`