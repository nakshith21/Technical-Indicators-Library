import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================
# RSI - RELATIVE STRENGTH INDEX
# ============================================

def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index)
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period
    
    Parameters:
    - prices: array of closing prices
    - period: lookback period (default 14 days)
    
    Returns: RSI values (0-100)
    """
    
    # Step 1: Calculate price changes (delta)
    deltas = np.diff(prices)  # Today's price - Yesterday's price
    
    # Step 2: Separate gains and losses
    gains = deltas.copy()
    losses = deltas.copy()
    
    gains[gains < 0] = 0  # Keep only positive changes
    losses[losses > 0] = 0  # Keep only negative changes
    losses = np.abs(losses)  # Make losses positive numbers
    
    # Step 3: Calculate average gains and losses
    # First average: simple mean of first 'period' values
    avg_gain = np.zeros(len(gains))
    avg_loss = np.zeros(len(losses))
    
    # Initial averages (simple mean)
    avg_gain[period-1] = np.mean(gains[:period])
    avg_loss[period-1] = np.mean(losses[:period])
    
    # Subsequent averages (smoothed with previous average)
    # Formula: ((previous_avg * 13) + current_value) / 14
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
    
    # Step 4: Calculate RS (Relative Strength)
    rs = np.divide(avg_gain, avg_loss, 
                   out=np.zeros_like(avg_gain), 
                   where=avg_loss!=0)  # Avoid division by zero
    
    # Step 5: Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Add NaN for first period values (not enough data)
    rsi = np.concatenate([[np.nan], rsi])  # Add NaN at start to match price length
    
    return rsi


# ============================================
# MACD - MOVING AVERAGE CONVERGENCE DIVERGENCE
# ============================================

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    MACD Line = 12-day EMA - 26-day EMA
    Signal Line = 9-day EMA of MACD Line
    Histogram = MACD Line - Signal Line
    
    Parameters:
    - prices: array of closing prices
    - fast: fast EMA period (default 12)
    - slow: slow EMA period (default 26)
    - signal: signal line EMA period (default 9)
    
    Returns: (macd_line, signal_line, histogram)
    """
    
    # Helper function: Calculate EMA (Exponential Moving Average)
    def calculate_ema(data, period):
        """
        EMA = (Price_today * K) + (EMA_yesterday * (1 - K))
        where K = 2 / (period + 1)
        """
        ema = np.zeros(len(data))
        
        # First EMA = Simple Moving Average
        ema[period-1] = np.mean(data[:period])
        
        # Multiplier
        k = 2 / (period + 1)
        
        # Calculate subsequent EMAs
        for i in range(period, len(data)):
            ema[i] = (data[i] * k) + (ema[i-1] * (1 - k))
        
        # Set first values to NaN (not enough data)
        ema[:period-1] = np.nan
        
        return ema
    
    # Step 1: Calculate fast and slow EMAs
    fast_ema = calculate_ema(prices, fast)
    slow_ema = calculate_ema(prices, slow)
    
    # Step 2: Calculate MACD Line (fast EMA - slow EMA)
    macd_line = fast_ema - slow_ema
    
    # Step 3: Calculate Signal Line (9-day EMA of MACD)
    # Need to handle NaN values
    macd_no_nan = macd_line[~np.isnan(macd_line)]
    signal_line_values = calculate_ema(macd_no_nan, signal)
    
    # Reconstruct signal line with NaN padding
    signal_line = np.full(len(prices), np.nan)
    signal_line[slow-1:] = signal_line_values
    
    # Step 4: Calculate Histogram (MACD - Signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# ============================================
# BOLLINGER BANDS
# ============================================

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Middle Band = 20-day Simple Moving Average (SMA)
    Upper Band = Middle Band + (2 * Standard Deviation)
    Lower Band = Middle Band - (2 * Standard Deviation)
    
    Parameters:
    - prices: array of closing prices
    - period: SMA period (default 20)
    - num_std: number of standard deviations (default 2)
    
    Returns: (middle_band, upper_band, lower_band)
    """
    
    # Step 1: Calculate Middle Band (Simple Moving Average)
    middle_band = np.zeros(len(prices))
    
    for i in range(period-1, len(prices)):
        middle_band[i] = np.mean(prices[i-period+1:i+1])
    
    middle_band[:period-1] = np.nan
    
    # Step 2: Calculate Standard Deviation for each window
    std_dev = np.zeros(len(prices))
    
    for i in range(period-1, len(prices)):
        std_dev[i] = np.std(prices[i-period+1:i+1])
    
    std_dev[:period-1] = np.nan
    
    # Step 3: Calculate Upper and Lower Bands
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    
    return middle_band, upper_band, lower_band


# ============================================
# TESTING & VISUALIZATION
# ============================================

def test_indicators(ticker='AAPL', period='6mo'):
    """
    Download stock data and test all indicators
    """
    print(f"📊 Testing indicators for {ticker}...")
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    prices = df['Close'].values
    dates = df.index
    
    print(f"✅ Downloaded {len(prices)} days of data")
    
    # Calculate indicators
    print("\n🔍 Calculating RSI...")
    rsi = calculate_rsi(prices)
    
    print("🔍 Calculating MACD...")
    macd_line, signal_line, histogram = calculate_macd(prices)
    
    print("🔍 Calculating Bollinger Bands...")
    middle_band, upper_band, lower_band = calculate_bollinger_bands(prices)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'{ticker} - Technical Indicators', fontsize=16, fontweight='bold')
    
    # Plot 1: Price with Bollinger Bands
    axes[0].plot(dates, prices, label='Price', color='black', linewidth=2)
    axes[0].plot(dates, middle_band, label='Middle Band (SMA 20)', color='blue', linestyle='--')
    axes[0].plot(dates, upper_band, label='Upper Band (+2σ)', color='red', linestyle='--')
    axes[0].plot(dates, lower_band, label='Lower Band (-2σ)', color='green', linestyle='--')
    axes[0].fill_between(dates, upper_band, lower_band, alpha=0.1, color='gray')
    axes[0].set_title('Price & Bollinger Bands')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: RSI
    axes[1].plot(dates, rsi, label='RSI (14)', color='purple', linewidth=2)
    axes[1].axhline(70, color='red', linestyle='--', label='Overbought (70)')
    axes[1].axhline(30, color='green', linestyle='--', label='Oversold (30)')
    axes[1].fill_between(dates, 70, 100, alpha=0.1, color='red')
    axes[1].fill_between(dates, 0, 30, alpha=0.1, color='green')
    axes[1].set_title('RSI (Relative Strength Index)')
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: MACD
    axes[2].plot(dates, macd_line, label='MACD Line', color='blue', linewidth=2)
    axes[2].plot(dates, signal_line, label='Signal Line', color='red', linewidth=2)
    axes[2].bar(dates, histogram, label='Histogram', color='gray', alpha=0.3)
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].set_title('MACD (Moving Average Convergence Divergence)')
    axes[2].set_ylabel('MACD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Volume
    axes[3].bar(dates, df['Volume'], color='skyblue', alpha=0.5)
    axes[3].set_title('Volume')
    axes[3].set_ylabel('Volume')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_indicators.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{ticker}_indicators.png'")
    plt.show()
    
    # Print current values
    print("\n" + "="*60)
    print(f"CURRENT INDICATORS FOR {ticker}")
    print("="*60)
    print(f"Current Price: ${prices[-1]:.2f}")
    print(f"\nRSI: {rsi[-1]:.2f}")
    if rsi[-1] > 70:
        print("  ⚠️ OVERBOUGHT - Consider selling")
    elif rsi[-1] < 30:
        print("  ✅ OVERSOLD - Consider buying")
    else:
        print("  ➡️ NEUTRAL")
    
    print(f"\nMACD Line: {macd_line[-1]:.2f}")
    print(f"Signal Line: {signal_line[-1]:.2f}")
    print(f"Histogram: {histogram[-1]:.2f}")
    if macd_line[-1] > signal_line[-1]:
        print("  ✅ BULLISH - MACD above signal")
    else:
        print("  ⚠️ BEARISH - MACD below signal")
    
    print(f"\nBollinger Bands:")
    print(f"  Upper: ${upper_band[-1]:.2f}")
    print(f"  Middle: ${middle_band[-1]:.2f}")
    print(f"  Lower: ${lower_band[-1]:.2f}")
    
    band_position = (prices[-1] - lower_band[-1]) / (upper_band[-1] - lower_band[-1])
    print(f"  Position: {band_position*100:.1f}% of band width")
    
    if prices[-1] > upper_band[-1]:
        print("  ⚠️ Above upper band - potentially overbought")
    elif prices[-1] < lower_band[-1]:
        print("  ✅ Below lower band - potentially oversold")
    else:
        print("  ➡️ Within bands")
    
    print("="*60)
    
    return {
        'prices': prices,
        'dates': dates,
        'rsi': rsi,
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram,
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("🚀 Technical Indicators Library")
    print("="*60)
    
    # Test with AAPL
    indicators = test_indicators('AAPL', period='6mo')
    
    print("\n" + "="*60)
    print("💡 UNDERSTANDING THE INDICATORS:")
    print("="*60)
    print("""
    RSI (Relative Strength Index):
    - Measures momentum (speed of price changes)
    - > 70: Overbought (price rose too fast, may drop)
    - < 30: Oversold (price fell too fast, may rise)
    - Use for: Finding potential reversals
    
    MACD (Moving Average Convergence Divergence):
    - Shows trend direction and momentum
    - MACD > Signal: Bullish (buy signal)
    - MACD < Signal: Bearish (sell signal)
    - Histogram shows strength of signal
    - Use for: Catching trend changes early
    
    Bollinger Bands:
    - Shows price volatility and extremes
    - Price touches upper band: Potentially overbought
    - Price touches lower band: Potentially oversold
    - Bands squeeze together: Low volatility (breakout coming)
    - Bands spread apart: High volatility (trend in progress)
    - Use for: Finding entry/exit points
    """)
    
    print("\n🎯 Try different stocks:")
    print("indicators = test_indicators('TSLA')")
    print("indicators = test_indicators('GOOGL')")


# ============================================
# BONUS: Create DataFrame with all indicators
# ============================================

def create_indicator_dataframe(ticker='AAPL', period='6mo'):
    """
    Create a pandas DataFrame with all indicators
    Perfect for backtesting strategies!
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    # Calculate all indicators
    df['RSI'] = calculate_rsi(df['Close'].values)
    
    macd_line, signal_line, histogram = calculate_macd(df['Close'].values)
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = histogram
    
    middle_band, upper_band, lower_band = calculate_bollinger_bands(df['Close'].values)
    df['BB_Middle'] = middle_band
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    
    # Add simple signals
    df['RSI_Signal'] = 'NEUTRAL'
    df.loc[df['RSI'] > 70, 'RSI_Signal'] = 'OVERBOUGHT'
    df.loc[df['RSI'] < 30, 'RSI_Signal'] = 'OVERSOLD'
    
    df['MACD_Signal_Type'] = 'NEUTRAL'
    df.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Signal_Type'] = 'BULLISH'
    df.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Signal_Type'] = 'BEARISH'
    
    return df


# Example usage:
# df = create_indicator_dataframe('AAPL')
# print(df.tail(10))  # See last 10 days with all indicators