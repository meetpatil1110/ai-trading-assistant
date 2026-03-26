# 📈 AI Stock Trading Assistant

A professional-grade trading dashboard built with Streamlit, combining **technical analysis**, **machine learning (LSTM)**, and **backtesting** to analyze stocks and generate trading signals.

## 🎯 Features

### 📊 Technical Analysis
- **Moving Averages**: MA20 and MA50 for trend detection
- **RSI (14-period)**: Overbought/oversold identification
- **MACD**: Momentum indicator with signal line and histogram
- **Volume Analysis**: Color-coded volume bars (green/red)

### 🤖 AI Prediction
- **LSTM Neural Network**: Pre-trained on historical stock data
- **Sequence Length**: 60-day lookback window
- **Close Price Prediction**: Forecasts next price movement
- **Trading Signal Generation**: BUY/SELL/HOLD recommendations

### 💹 Backtesting Engine
- **Initial Capital**: ₹100,000
- **MA Crossover Strategy**: Trend-following logic
- **Realistic Costs**: 
  - Transaction cost: 0.1%
  - Slippage: 0.05%
- **Risk Metrics**:
  - Sharpe Ratio (annualized)
  - Max Drawdown
  - Win Rate
  - Volatility

### 📈 Interactive Visualization
- **4-Subplot Dashboard**:
  1. Price chart with candlesticks + moving averages
  2. Volume bars
  3. RSI with overbought/oversold levels
  4. MACD with signal line and histogram
- **Dark theme** for professional appearance
- **Buy/Sell markers** on price chart
- **Range selector** (1m, 3m, 6m, 1y, All)
- **Equity curve** showing portfolio performance

### ⚡ Performance
- **Data Caching**: `@st.cache_data` decorator prevents re-downloads
- **8 Years of Data**: Historical data from 2018-01-01
- **Real-time Updates**: Fresh data on each query

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. **Clone/Download the project**
   ```bash
   cd stock-ai-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure LSTM model exists**
   - Model file: `lstm_model.h5` (must be in project root)
   - If missing, the app will display an error message

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at:
- Local: `http://localhost:8501`
- Network: See terminal output

## 📝 How to Use

1. **Enter Stock Symbol** (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`)
2. **Click "Predict"** button
3. **View Results**:
   - Technical analysis chart
   - Backtest performance metrics
   - AI trading signal recommendation
   - Equity curve visualization

## 📊 Interpretation Guide

### RSI (Relative Strength Index)
- **Above 70**: Overbought (potential pullback)
- **Below 30**: Oversold (potential bounce)
- **30-70**: Neutral zone

### MACD Histogram
- **Positive (green)**: Bullish momentum
- **Negative (red)**: Bearish momentum

### Moving Averages
- **MA20 > MA50**: Uptrend (BUY signal)
- **MA20 < MA50**: Downtrend (SELL signal)

### Sharpe Ratio
- Measures risk-adjusted return
- **> 1.0**: Good risk-adjusted performance
- **> 2.0**: Excellent

### Max Drawdown
- Largest peak-to-trough decline
- Lower is better
- Shows downside risk

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Charting | Plotly |
| Data Fetching | yfinance |
| Data Processing | pandas, numpy |
| ML Model | TensorFlow/Keras (LSTM) |
| Scaling | scikit-learn (MinMaxScaler) |

## 📈 Strategy Details

### Trading Logic
- **Entry**: Signal == 1 (Bullish trend detected)
- **Exit**: Signal == -1 (Bearish trend detected)
- **Position Management**: One position at a time

### Signal Generation
Based on MA20 vs MA50 moving average crossover:
- Signal = 1 when MA20 > MA50
- Signal = -1 when MA20 < MA50
- Signal = 0 otherwise

### Performance Tracking
- **Total Return**: Net profit/loss as percentage
- **Win Rate**: Percentage of profitable days
- **Volatility**: Annualized price fluctuation (252 trading days)

## ⚠️ Important Notes

### Data Leakage (Known Limitation)
- Current: Scaler fitted on full dataset (demo only)
- Production: Should fit scaler ONLY on training set
- Impact: Results are for demonstration, not production-ready

### LSTM Model Architecture
- Input: Close price only (single feature)
- Sequence: 60-day lookback
- Future Enhancement: Retrain with RSI, MACD, MA_20, MA_50

### Data Quality
- Requires valid stock symbols (NSE/BSE format: `SYMBOL.NS`)
- Minimum 80+ trading days for indicators to stabilize
- Missing data is automatically handled (forward fill)

## 🎓 For Interviews

### Key Talking Points
1. **Risk-Adjusted Returns**: "My Sharpe Ratio calculation annualizes returns over 252 trading days"
2. **Realistic Backtesting**: "Includes 0.1% transaction costs and 0.05% slippage"
3. **Technical Standards**: "Uses adjust=False for EMAs, matching TradingView/TA-Lib behavior"
4. **Caching**: "Implements Streamlit caching to prevent repeated API calls"
5. **Error Handling**: "Handles NaN values safely throughout the pipeline"

## 🔮 Future Enhancements

1. **Multi-feature LSTM**
   - Add RSI, MACD, MA_20, MA_50 as features
   - Reshape from (samples, 60, 1) → (samples, 60, 5)
   - Requires retraining LSTM model

2. **Advanced Strategies**
   - Bollinger Bands
   - Stochastic Oscillator
   - Support/Resistance levels

3. **Portfolio Optimization**
   - Multi-stock portfolio
   - Correlation analysis
   - Rebalancing logic

4. **Production Readiness**
   - Train/test split with proper scaler handling
   - Model validation metrics
   - Robustness testing

## 📞 Troubleshooting

### "LSTM model not found" Error
- Ensure `lstm_model.h5` exists in project root
- Check file permissions

### "Invalid stock symbol" Error
- Use correct format: `SYMBOL.NS` (with exchange suffix)
- Try: RELIANCE.NS, TCS.NS, INFY.NS, WIPRO.NS

### Empty chart or flat equity curve
- Verify data exists for the stock
- Check 8+ years of historical data available
- Ensure technical indicators calculated correctly

### "Portfolio values are NaN"
- Data quality issue (likely insufficient historical data)
- Try a major stock like RELIANCE.NS
- Check internet connection for data download

## 📄 License

This project is for educational and portfolio purposes.

## 👨‍💻 Author

Built as an AI trading assistant combining ML, technical analysis, and risk management.

---

**Last Updated**: March 27, 2026  
**Status**: Production-ready for demonstration
