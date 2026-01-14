# 📊 Technical Indicators Library

Custom implementation of RSI, MACD, and Bollinger Bands built from scratch using NumPy.

![Indicators Chart](AAPL_indicators.png)

## 🎯 What This Does

Calculates three essential technical indicators used by professional traders:

### RSI (Relative Strength Index)
- Identifies overbought/oversold conditions
- Range: 0-100
- **> 70**: Overbought (sell signal)
- **< 30**: Oversold (buy signal)

### MACD (Moving Average Convergence Divergence)
- Detects trend changes and momentum
- **MACD > Signal**: Bullish (buy signal)
- **MACD < Signal**: Bearish (sell signal)

### Bollinger Bands
- Measures price volatility
- Price at upper band: Potentially overbought
- Price at lower band: Potentially oversold
- Band squeeze: Breakout coming soon

## 🛠️ Tech Stack
- **Python** - Core language
- **NumPy** - Mathematical calculations
- **yfinance** - Real-time stock data
- **Matplotlib** - Data visualization

## 📦 Installation
```bash
pip install yfinance pandas numpy matplotlib
```

## 🚀 Usage
```python
python indicators.py
```

The script will:
1. Download 6 months of AAPL data
2. Calculate all three indicators
3. Generate visualization chart
4. Print current indicator values with trading signals

## 📈 Example Output
```
Current Price: $175.43

RSI: 45.32
  ➡️ NEUTRAL

MACD Line: 2.34
Signal Line: 1.87
  ✅ BULLISH - MACD above signal

Bollinger Bands:
  Upper: $178.50
  Middle: $172.20
  Lower: $165.90
  ➡️ Within bands
```

## 🎓 What I Learned

- How RSI measures momentum through gain/loss ratios
- MACD's use of exponential moving averages to detect trends
- Bollinger Bands' application of standard deviation for volatility
- NumPy array operations for efficient calculations
- Financial data visualization best practices


## 👨‍💻 Author

**Nakshith D N**
```
