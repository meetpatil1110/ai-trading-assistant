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
- **Confidence Scoring**: 
  - 60% volatility-relative confidence + 40% movement-based
  - Color-coded: Green (>70%), Yellow (40-70%), Red (<40%)
  - Balances signal strength with market context

### 💹 Backtesting Engine
- **Initial Capital**: Dynamic slider (₹10,000 - ₹10,00,000)
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
- **Day-wise clarity**: Hover tooltips show date (e.g., "Mon, Feb 26, 2024")
- **Range selector** (1m, 3m, 6m, 1y, All)
- **Equity curve** showing portfolio performance

### ⚡ Performance
- **Data Caching**: `@st.cache_data` decorator prevents re-downloads
- **8 Years of Data**: Historical data from 2018-01-01
- **Real-time Updates**: Fresh data on each query
- **Fully Standalone**: No external API dependencies

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

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Streamlit | 1.32.0 |
| Charting | Plotly | 5.20.0 |
| Data Fetching | yfinance | 0.2.36 |
| Data Processing | pandas | 2.2.2 |
| Numerical Computing | numpy | 1.26.4 |
| ML Model | Keras | 2.15.0 |
| Model IO | h5py | 3.10.0 |
| Scaling | scikit-learn | 1.5.1 |

**Environment**: Python 3.14.3 compatible, no external API dependencies

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

### Confidence Scoring
- **Volatility-Relative**: Compares predicted move to market volatility (60% weight)
- **Movement-Based**: Magnitude of predicted price change (40% weight)
- **Scale**: 0-100%, clamped safely to avoid edge cases
- **Interpretation**:
  - >70% = High confidence (significant predicted move relative to volatility)
  - 40-70% = Medium confidence (moderate move)
  - <40% = Low confidence (small move relative to volatility)

### Performance Tracking
- **Total Return**: Net profit/loss as percentage
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Annualized price fluctuation (252 trading days)
- **Sharpe Ratio**: Risk-adjusted return metric

## ⚠️ Important Notes

### Data Leakage (Known Limitation)
- Current: Scaler fitted on full dataset (demo only)
- Production: Should fit scaler ONLY on training set
- Impact: Results are for demonstration, not production-ready trading

### LSTM Model Architecture
- Input: Close price only (single feature)
- Sequence: 60-day lookback
- Future Enhancement: Retrain with RSI, MACD, MA_20, MA_50
- Pre-trained weights: `lstm_model.h5` (Keras format)

### Data Quality
- Requires valid stock symbols (NSE/BSE format: `SYMBOL.NS`)
- Minimum 60 trading days for LSTM predictions
- Minimum 80+ trading days for indicators to stabilize
- Missing data is automatically handled (forward fill)

## 🗂️ Project Structure

```
stock-ai-project/
├── app.py                 # Main Streamlit application
├── lstm_model.h5          # Pre-trained LSTM model (Keras)
├── backend/
│   └── scaler.pkl         # Pre-trained MinMaxScaler
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git exclusions
```

## 🌐 Deployment

### Local Execution
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub repository
2. Go to https://share.streamlit.io/
3. Paste GitHub repo URL
4. Select `app.py` as main file
5. Deploy ✅

The app is fully self-contained with no external API dependencies, making it ideal for Streamlit Cloud deployment.

## 🎓 For Interviews

### Key Talking Points
1. **Risk-Adjusted Returns**: "My Sharpe Ratio calculation annualizes returns over 252 trading days"
2. **Realistic Backtesting**: "Includes 0.1% transaction costs and 0.05% slippage for realistic analysis"
3. **Technical Standards**: "Uses adjust=False for EMAs, matching TradingView/TA-Lib behavior"
4. **Intelligent Confidence**: "Confidence scoring compares predicted move to market volatility"
5. **Caching Architecture**: "Implements Streamlit caching to prevent repeated API calls, improving performance"
6. **Error Handling**: "Safely handles NaN values, edge cases, and scalar conversions throughout pipeline"
7. **Fully Standalone**: "No external backend API dependencies; all inference runs locally in Streamlit"
8. **Production-Grade UI**: "4-subplot dashboard with interactive range selector and day-wise date clarity"

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
- Check file permissions: `ls -lh lstm_model.h5`

### "Scaler file not found" Error
- Ensure `backend/scaler.pkl` exists
- Check: `ls -lh backend/scaler.pkl`

### "Invalid stock symbol" Error
- Use correct format: `SYMBOL.NS` (with exchange suffix)
- Try: RELIANCE.NS, TCS.NS, INFY.NS, WIPRO.NS, UPL.NS

### Empty chart or flat equity curve
- Verify data exists for the stock
- Check 8+ years of historical data available
- Ensure technical indicators calculated correctly

### "Portfolio values are NaN"
- Data quality issue (likely insufficient historical data)
- Try a major stock like RELIANCE.NS
- Check internet connection for data download

### Confidence score shows 0.0%
- Ensure volatility calculation is working
- Volatility converts to float scalar automatically
- Check minimum 60 days of data available

## 📄 License

This project is for educational and portfolio purposes.

## 👨‍💻 Author

Built as an AI trading assistant combining ML (LSTM), technical analysis, and risk management. Production-ready for demonstration on Streamlit Cloud.

---

**Last Updated**: March 27, 2026  
**Status**: Production-ready for Streamlit Cloud  
**GitHub**: https://github.com/meetpatil1110/ai-trading-assistant
