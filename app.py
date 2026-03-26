import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

# ============================================
# CACHING & DATA LOADING (PERFORMANCE)
# ============================================

@st.cache_data
def load_stock_data(stock, start_date="2018-01-01", end_date=None):
    """Load stock data with caching for better performance"""
    from datetime import datetime
    # Convert datetime.date to string if needed
    if hasattr(start_date, 'strftime'):
        start_date = start_date.strftime('%Y-%m-%d')
    if end_date and hasattr(end_date, 'strftime'):
        end_date = end_date.strftime('%Y-%m-%d')
    # If end_date is None, yfinance will use today's date
    return yf.download(stock, start=start_date, end=end_date)

# ============================================
# TECHNICAL INDICATOR FUNCTIONS
# ============================================

def compute_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI) with industry-standard calculation"""
    delta = prices.diff()
    # Industry standard: use clip instead of where
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    # Handle division by zero
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD, MACD Signal line, and Histogram (Industry standard)
    Using adjust=False for exponential moving averages
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - macd_signal
    return macd_line, macd_signal, histogram

def detect_signals(df):
    """Detect BUY/SELL signals from MA crossover"""
    df['Signal'] = 0
    # 1 = BUY (MA20 > MA50), -1 = SELL (MA20 < MA50)
    df.loc[df['MA_20'] > df['MA_50'], 'Signal'] = 1
    df.loc[df['MA_20'] < df['MA_50'], 'Signal'] = -1
    
    # Detect crossovers (fill NaN with 0 to avoid propagation)
    df['Position'] = df['Signal'].diff().fillna(0)
    # Position = 1: BUY signal (MA20 crosses above MA50)
    # Position = -2: SELL signal (MA20 crosses below MA50)
    
    return df

def backtest_strategy(df, initial_capital=100000, transaction_cost=0.001, slippage=0.0005):
    """
    Backtest trading strategy with realistic transaction costs and slippage.
    
    Args:
        df: DataFrame with trading signals
        initial_capital: Starting capital (₹)
        transaction_cost: Broker fee as % (default 0.1% = 0.001)
        slippage: Price slippage as % (default 0.05% = 0.0005)
    """
    import pandas as pd
    
    df = df.copy().reset_index(drop=True)
    
    # Track portfolio
    df['Holdings'] = 0.0  # Number of shares held
    df['Cash'] = initial_capital
    df['Portfolio_Value'] = initial_capital
    
    # Initialize first row explicitly
    df.at[0, 'Holdings'] = 0.0
    df.at[0, 'Cash'] = initial_capital
    df.at[0, 'Portfolio_Value'] = initial_capital
    
    shares_held = 0.0
    cash = float(initial_capital)
    
    for i in range(1, len(df)):
        # Get scalar values and convert explicitly
        try:
            current_price = float(df['Close'].iloc[i])
        except (ValueError, TypeError):
            continue  # Skip if can't convert to float
        
        if pd.isna(current_price):
            continue
        
        # Use Signal directly for trend-following (not Position for crossovers)
        try:
            current_signal = float(df['Signal'].iloc[i])
        except (ValueError, TypeError):
            current_signal = 0.0
        
        # Get MA values for threshold confirmation
        try:
            ma_20 = float(df['MA_20'].iloc[i])
            ma_50 = float(df['MA_50'].iloc[i])
        except (ValueError, TypeError):
            ma_20 = current_price
            ma_50 = current_price
        
        # BUY signal: Signal == 1 (bullish trend) and not holding
        if current_signal == 1 and shares_held == 0:
            # Apply slippage (unfavorable price movement)
            buy_price = current_price * (1 + slippage)
            shares_to_buy = int(cash / (buy_price * (1 + transaction_cost)))
            
            if shares_to_buy > 0:
                cost = shares_to_buy * buy_price
                fee = cost * transaction_cost
                shares_held = float(shares_to_buy)
                cash -= (cost + fee)
        
        # SELL signal: Signal == -1 (bearish trend) and holding
        elif current_signal == -1 and shares_held > 0:
            # Apply slippage (unfavorable price movement)
            sell_price = current_price * (1 - slippage)
            revenue = shares_held * sell_price
            fee = revenue * transaction_cost
            cash += (revenue - fee)
            shares_held = 0.0
        
        # Update portfolio value (mark-to-market)
        portfolio_value = cash + (shares_held * current_price)
        
        # Use at[] for scalar assignment with integer position
        df.at[i, 'Holdings'] = shares_held
        df.at[i, 'Cash'] = cash
        df.at[i, 'Portfolio_Value'] = portfolio_value
    
    # Forward fill any NaN values and fill remaining with initial capital
    df['Portfolio_Value'] = df['Portfolio_Value'].fillna(method='ffill')
    df['Portfolio_Value'] = df['Portfolio_Value'].fillna(initial_capital)
    
    return df

# Page setup
st.set_page_config(page_title="AI Trading Assistant", layout="wide")

# Title
st.title("📈 AI Stock Trading Assistant")

# Input
stock = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")

# User-controlled initial capital (OUTSIDE button so it persists)
st.markdown("---")
initial_capital = st.slider(
    "💰 Initial Capital (₹)",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000,
    help="Adjust the starting capital for backtesting simulation"
)

# Date range selector
from datetime import datetime, timedelta
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "📅 Select Start Date",
        value=datetime(2018, 1, 1).date(),
        help="Beginning of the analysis period"
    )
with col2:
    end_date = st.date_input(
        "📅 Select End Date",
        value=datetime.today().date(),
        help="End of the analysis period"
    )

# Validate date range
if start_date >= end_date:
    st.error("❌ Start date must be before end date!")
    st.stop()

# Button
if st.button("Predict"):
    st.session_state.run_prediction = True

    with st.spinner("Fetching data & running AI model..."):

        # -------------------------------
        # 📥 Fetch data (cached)
        # -------------------------------
        data = load_stock_data(stock, start_date=start_date, end_date=end_date)

        if data.empty:
            st.error("❌ Invalid stock symbol or no data available for selected date range")
        elif len(data) < 60:
            st.warning(f"⚠️ Not enough data for analysis. Found {len(data)} rows, but need at least 60. Please select a larger date range.")
            st.stop()
        else:

            # -------------------------------
            # 📊 Add indicators (preserve original data for model)
            # -------------------------------
            data['MA_20'] = data['Close'].rolling(20).mean()
            data['MA_50'] = data['Close'].rolling(50).mean()
            
            # Drop initial NaN from rolling averages and convert index to column
            clean_data = data.dropna().reset_index(drop=False).copy()
            
            # Calculate all technical indicators on clean data
            clean_data['RSI'] = compute_rsi(clean_data['Close'], period=14)
            clean_data['MACD'], clean_data['MACD_Signal'], clean_data['Histogram'] = compute_macd(clean_data['Close'])
            
            # Drop any remaining NaN from technical indicators
            clean_data = clean_data.dropna().reset_index(drop=True)
            
            # Detect trading signals (fills NaN in Position column)
            clean_data = detect_signals(clean_data)
            
            # DEBUG: Show signal distribution
            buy_signal_count = (clean_data['Signal'] == 1).sum()
            sell_signal_count = (clean_data['Signal'] == -1).sum()
            st.write(f"🔍 **Debug Info:** Total BUY signals: **{buy_signal_count}** | Total SELL signals: **{sell_signal_count}**")
            
            # Run backtest (with NaN safeguards in place)
            backtest_data = backtest_strategy(clean_data.copy(), initial_capital=initial_capital)
            # Add Date column to backtest data for plotting
            backtest_data['Date'] = clean_data['Date'].values
            
            # Prepare plot data (Date column already exists from earlier reset_index)
            plot_df = clean_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'Histogram', 'Position']].copy()
            backtest_df = backtest_data[['Date', 'Portfolio_Value']].copy()
            
            # SAFETY CHECK: Validate backtest results
            if backtest_df['Portfolio_Value'].isna().all():
                st.error("❌ Backtest failed: Portfolio values are NaN. Please check data quality.")
                st.stop()

            # Set theme and chart type
            template = "plotly_dark"

            # -------------------------------
            # 📊 Interactive Stock Chart
            # -------------------------------
            st.subheader("📊 Interactive Stock Chart")

            # Create figure with 4 subplots: Price, Volume, RSI, MACD
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.15, 0.2, 0.2],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )

            # Color for candles
            colors = ['red' if o > c else 'green' 
                     for o, c in zip(plot_df['Open'], plot_df['Close'])]

            # ========== CANDLESTICK CHART (Row 1) ==========
            fig.add_trace(go.Candlestick(
                x=plot_df['Date'],
                open=plot_df['Open'],
                high=plot_df['High'],
                low=plot_df['Low'],
                close=plot_df['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red',
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>' +
                              'Open: ₹%{open:.2f}<br>' +
                              'High: ₹%{high:.2f}<br>' +
                              'Low: ₹%{low:.2f}<br>' +
                              'Close: ₹%{close:.2f}<extra></extra>'
            ), row=1, col=1)

            # Add Moving Averages to Price chart
            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['MA_20'],
                name='MA 20',
                line=dict(color='orange', width=1),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>MA 20: ₹%{y:.2f}<extra></extra>'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['MA_50'],
                name='MA 50',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>MA 50: ₹%{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # ========== BUY/SELL MARKERS (on price chart) ==========
            buy_signals = plot_df[plot_df['Position'] == 1]
            sell_signals = plot_df[plot_df['Position'] == -2]
            
            if len(buy_signals) > 0:
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Low'] * 0.95,  # Place slightly below
                    mode='markers',
                    name='BUY Signal',
                    marker=dict(size=12, color='green', symbol='triangle-up'),
                    hovertext=['BUY' for _ in buy_signals],
                    hoverinfo='x+text',
                    showlegend=True
                ), row=1, col=1)

            if len(sell_signals) > 0:
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['High'] * 1.05,  # Place slightly above
                    mode='markers',
                    name='SELL Signal',
                    marker=dict(size=12, color='red', symbol='triangle-down'),
                    hovertext=['SELL' for _ in sell_signals],
                    hoverinfo='x+text',
                    showlegend=True
                ), row=1, col=1)

            # ========== VOLUME (Row 2) ==========
            fig.add_trace(go.Bar(
                x=plot_df['Date'],
                y=plot_df['Volume'],
                name='Volume',
                marker=dict(color=colors, opacity=0.5),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>',
                showlegend=False
            ), row=2, col=1)

            # ========== RSI (Row 3) ==========
            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['RSI'],
                name='RSI (14)',
                line=dict(color='purple', width=2),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>RSI: %{y:.2f}<extra></extra>'
            ), row=3, col=1)

            # Add RSI levels (30 and 70)
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)",
                         annotation_position="right", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)",
                         annotation_position="right", row=3, col=1)

            # ========== MACD (Row 4) ==========
            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['MACD'],
                name='MACD',
                line=dict(color='blue', width=2),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>MACD: %{y:.2f}<extra></extra>'
            ), row=4, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['MACD_Signal'],
                name='MACD Signal Line',
                line=dict(color='red', width=1, dash='dash'),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>MACD Signal: %{y:.2f}<extra></extra>'
            ), row=4, col=1)

            fig.add_trace(go.Bar(
                x=plot_df['Date'],
                y=plot_df['Histogram'],
                name='Histogram',
                marker=dict(color=[('green' if h > 0 else 'red') for h in plot_df['Histogram']], opacity=0.3),
                hovertemplate='<b>%{x|%a, %b %d, %Y}</b><br>Histogram: %{y:.2f}<extra></extra>',
                showlegend=False
            ), row=4, col=1)

            # Update axes labels
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            # X-axis with range selector and date formatting
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month"),
                        dict(count=3, label="3m", step="month"),
                        dict(count=6, label="6m", step="month"),
                        dict(count=1, label="1y", step="year"),
                        dict(step="all", label="All")
                    ])
                ),
                tickformat='%b %d\n%Y',  # Format: "Mar 27\n2026"
                row=4, col=1
            )

            # Update overall layout
            fig.update_layout(
                title=f"📈 {stock} - Professional Trading Dashboard",
                template=template,
                height=1000,
                hovermode='x unified',
                font=dict(size=11),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                )
            )

            st.plotly_chart(fig, width='stretch')

            # ========== BACKTESTING RESULTS ==========
            st.markdown("---")
            st.subheader(f"📊 Strategy Backtest Results (₹{initial_capital:,} Initial Capital)")
            
            # Calculate backtest metrics
            final_portfolio = float(backtest_df['Portfolio_Value'].iloc[-1])
            total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            
            # Calculate Sharpe Ratio (quant-level metric)
            returns = backtest_df['Portfolio_Value'].pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
            
            # Calculate Max Drawdown (critical risk metric)
            rolling_max = backtest_df['Portfolio_Value'].cummax()
            drawdown = (backtest_df['Portfolio_Value'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate Win Rate
            trades = backtest_df['Portfolio_Value'].pct_change()
            wins = (trades > 0).sum()
            total = trades.count()
            win_rate = (wins / total) * 100 if total > 0 else 0
            
            # Calculate Volatility (annualized)
            volatility = returns.std() * (252 ** 0.5)
            
            backcol1, backcol2, backcol3, backcol4, backcol5, backcol6 = st.columns(6)
            with backcol1:
                st.metric("Final Portfolio Value", f"₹{final_portfolio:,.2f}", 
                         delta=f"{total_return:.2f}%")
            with backcol2:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}",
                         delta="Risk-adjusted return")
            with backcol3:
                st.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%",
                         delta="Largest peak-to-trough")
            with backcol4:
                st.metric("Win Rate", f"{win_rate:.2f}%",
                         delta=f"{int(wins)}/{int(total)} wins")
            with backcol5:
                st.metric("Volatility", f"{volatility:.2f}",
                         delta="Annualized")
            with backcol6:
                st.metric("Total Trades", f"{buy_count + sell_count}")
            
            # Plot equity curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=backtest_df['Date'],
                y=backtest_df['Portfolio_Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00CC96', width=2),
                fill='tozeroy',
                hovertemplate='₹%{y:,.2f}<extra></extra>'
            ))
            
            fig_equity.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                                annotation_text="Initial Capital")
            
            fig_equity.update_layout(
                title="Equity Curve - MA Crossover Strategy",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                template=template,
                height=350,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig_equity, width='stretch')

            # ========== AI MODEL PREDICTION ==========
            st.markdown("---")
            st.subheader("🤖 AI LSTM Price Prediction")
            
            # Load pre-trained scaler and model
            try:
                with open('backend/scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
            except FileNotFoundError:
                st.error("❌ Scaler file not found. Please ensure backend/scaler.pkl exists.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error loading scaler: {str(e)}")
                st.stop()
            
            # Load pre-trained LSTM model
            try:
                model = load_model("lstm_model.h5")
            except FileNotFoundError:
                st.error("❌ Model file not found. Please ensure lstm_model.h5 is in the project directory.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.stop()
            
            # Prepare input: last 60 days of Close prices
            close_prices = data[['Close']].values
            
            # Scale using pre-trained scaler
            scaled_data = scaler.transform(close_prices)
            
            # Create input sequence: (1, 60, 1) - last 60 days, 1 feature
            X = np.array([scaled_data[-60:]])
            
            # Make prediction
            prediction_scaled = model.predict(X, verbose=0)
            
            # Inverse transform to get actual price
            predicted_prices = scaler.inverse_transform(prediction_scaled)
            predicted_price = float(predicted_prices[0][0])

            # Current market price (latest close)
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate expected change %
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Get high and low for the period
            period_high = float(plot_df['High'].max())
            period_low = float(plot_df['Low'].min())
            
            # ========== CONFIDENCE CALCULATION ==========
            # Confidence based on predicted price movement relative to volatility
            
            # Calculate historical volatility
            price_returns = data['Close'].pct_change().dropna()
            volatility = float(price_returns.std() * 100)  # Convert to percentage (ensure scalar)
            
            # Method 1: Movement-based confidence (simple)
            # Confidence increases with magnitude of predicted change
            movement_confidence = min(abs(price_change_pct) * 25, 100)
            
            # Method 2: Volatility-relative confidence (advanced)
            # Compare predicted move to market volatility
            if volatility > 0.01:  # Avoid division by very small numbers
                vol_relative_confidence = min((abs(price_change_pct) / volatility) * 50, 100)
            else:
                vol_relative_confidence = movement_confidence
            
            # Combined confidence: 60% weight on volatility-relative, 40% on pure movement
            # This balances signal strength with market context
            confidence = (vol_relative_confidence * 0.6) + (movement_confidence * 0.4)
            confidence = max(0, min(confidence, 100))  # Clamp to [0, 100]
            
            # Handle any NaN edge cases
            if pd.isna(confidence) or np.isnan(confidence):
                confidence = 50.0  # Default neutral confidence

            # ========== TRADING SIGNAL LOGIC ==========
            # BUY if predicted rise > 2%
            # SELL if predicted fall < -2%
            # HOLD otherwise
            if price_change_pct > 2.0:
                trading_signal = "🟢 BUY"
                signal_emoji = "🟢"
                signal_reason = f"Price expected to rise by {price_change_pct:.2f}%"
            elif price_change_pct < -2.0:
                trading_signal = "🔴 SELL"
                signal_emoji = "🔴"
                signal_reason = f"Price expected to fall by {abs(price_change_pct):.2f}%"
            else:
                trading_signal = "🟡 HOLD"
                signal_emoji = "🟡"
                signal_reason = f"Expected change {price_change_pct:.2f}% (minimal movement)"

            # ========== PREDICTION DISPLAY ==========
            st.markdown("---")
            st.subheader("🤖 AI LSTM Prediction Results")

            # Get latest technical values
            latest_rsi = float(plot_df['RSI'].iloc[-1])
            latest_macd = float(plot_df['MACD'].iloc[-1])
            latest_macd_signal = float(plot_df['MACD_Signal'].iloc[-1])
            latest_histogram = float(plot_df['Histogram'].iloc[-1])

            # Metrics row - AI Prediction Results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Current Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("🎯 Predicted Price", f"₹{predicted_price:.2f}", 
                         delta=f"₹{price_change:.2f}" if price_change != 0 else "No change")
            with col3:
                st.metric("📈 Period High", f"₹{period_high:.2f}")
            with col4:
                st.metric("📉 Period Low", f"₹{period_low:.2f}")

            # Technical Indicators
            st.subheader("📉 Technical Indicators")
            tech_col1, tech_col2, tech_col3 = st.columns(3)
            
            with tech_col1:
                rsi_status = "🔴 Overbought" if latest_rsi > 70 else "🟢 Oversold" if latest_rsi < 30 else "🟡 Neutral"
                st.metric("RSI (14)", f"{latest_rsi:.2f}", delta=rsi_status)
            
            with tech_col2:
                macd_status = "🟢 Bullish" if latest_histogram > 0 else "🔴 Bearish"
                st.metric("MACD Histogram", f"{latest_histogram:.4f}", delta=macd_status)
            
            with tech_col3:
                ma_signal = "🟢 BUY" if plot_df['MA_20'].iloc[-1] > plot_df['MA_50'].iloc[-1] else "🔴 SELL"
                ma_diff = abs(plot_df['MA_20'].iloc[-1] - plot_df['MA_50'].iloc[-1])
                st.metric("MA 20/50", f"{ma_signal}", delta=f"Diff: ₹{ma_diff:.2f}")

            # Trading Signal
            st.markdown("---")
            st.subheader("📢 AI Trading Signal")

            # Display the trading signal prominently
            col_signal1, col_signal2 = st.columns([2, 1])
            
            with col_signal1:
                if "BUY" in trading_signal:
                    st.success(f"{trading_signal} - {signal_reason}")
                elif "SELL" in trading_signal:
                    st.error(f"{trading_signal} - {signal_reason}")
                else:  # HOLD
                    st.warning(f"{trading_signal} - {signal_reason}")
            
            with col_signal2:
                # Display confidence with color-coding
                if confidence > 70:
                    confidence_color = "🟢"  # Green - high confidence
                    confidence_level = "High"
                elif confidence >= 40:
                    confidence_color = "🟡"  # Yellow - medium confidence
                    confidence_level = "Medium"
                else:
                    confidence_color = "🔴"  # Red - low confidence
                    confidence_level = "Low"
                
                st.metric("Confidence", f"{confidence:.1f}%", 
                         delta=f"{confidence_color} {confidence_level}")

            # Indicator Analysis Summary
            st.markdown("---")
            st.subheader("📋 Comprehensive Analysis Summary")
            
            analysis_cols = st.columns(2)
            
            with analysis_cols[0]:
                st.info(f"""
                **Price Prediction:**
                • Current: ₹{current_price:.2f}
                • Predicted: ₹{predicted_price:.2f}
                • Expected Change: {price_change_pct:.2f}%
                • Signal: {trading_signal}
                • Confidence: {confidence:.1f}% ({confidence_level})
                
                **Strategy Performance:**
                • Initial Capital: ₹{initial_capital:,}
                • Final Portfolio: ₹{final_portfolio:,.2f}
                • Sharpe Ratio: {sharpe_ratio:.2f} (Risk-adjusted return)
                • Max Drawdown: {max_drawdown * 100:.2f}% (Downside risk)
                • Total Return: {total_return:.2f}%
                """)
            
            with analysis_cols[1]:
                rsi_interpretation = "Overbought - Potential pullback" if latest_rsi > 70 else "Oversold - Potential bounce" if latest_rsi < 30 else "Neutral - No extreme signal"
                macd_interpretation = "Bullish - Momentum increasing" if latest_histogram > 0 else "Bearish - Momentum decreasing"
                
                st.info(f"""
                **Technical Indicators:**
                • RSI: {latest_rsi:.2f} - {rsi_interpretation}
                • MACD Histogram: {latest_histogram:.4f} - {macd_interpretation}
                • Trend: {'📈 Uptrend (BUY)' if plot_df['MA_20'].iloc[-1] > plot_df['MA_50'].iloc[-1] else '📉 Downtrend (SELL)'}
                
                **Confidence Explanation:**
                • Volatility (30d): {volatility:.2f}%
                • >70% = High confidence (large move)
                • 40-70% = Medium (moderate move)
                • <40% = Low (small move relative to volatility)
                """)


# Footer
st.markdown("---")
