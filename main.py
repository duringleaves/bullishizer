import os
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
import plotly.graph_objs as go
import plotly.utils

# Configuration
class Config:
    DATABASE_PATH = 'data/stock_tracker.db'
    SIMPLEPUSH_KEY = os.environ.get('SIMPLEPUSH_KEY', '')
    UPDATE_INTERVAL = int(os.environ.get('UPDATE_INTERVAL', 300))
    MARKET_HOURS_ONLY = True
    # Authentication settings
    AUTH_USERNAME = os.environ.get('AUTH_USERNAME', 'admin')
    AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD', 'password123')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize HTTP Basic Auth
auth = HTTPBasicAuth()

# Simple user database (just one user for this use case)
users = {
    Config.AUTH_USERNAME: generate_password_hash(Config.AUTH_PASSWORD)
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None

@auth.error_handler
def auth_error(status):
    return jsonify({'error': 'Authentication required'}), status

class TechnicalIndicators:
    """Custom implementation of technical indicators without TA-Lib dependency"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> tuple:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index (Simplified)"""
        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

class StockAnalyzer:
    def __init__(self):
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Stocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Stock data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL,
                ema_20 REAL, ema_50 REAL, ema_100 REAL, ema_200 REAL,
                volume_sma REAL, obv REAL, adx REAL, stoch_k REAL, stoch_d REAL,
                bb_upper REAL, bb_middle REAL, bb_lower REAL,
                fib_236 REAL, fib_382 REAL, fib_500 REAL, fib_618 REAL,
                bullishness_score REAL,
                FOREIGN KEY (symbol) REFERENCES stocks (symbol)
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES stocks (symbol)
            )
        ''')
        
        # Backtests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                strategy_params TEXT NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators using custom implementations"""
        # Ensure we have enough data
        if len(df) < 200:
            return df
        
        # Use our custom indicators
        ti = TechnicalIndicators()
        
        # RSI
        df['rsi'] = ti.rsi(df['Close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = ti.macd(df['Close'])
        
        # EMAs
        df['ema_20'] = ti.ema(df['Close'], 20)
        df['ema_50'] = ti.ema(df['Close'], 50)
        df['ema_100'] = ti.ema(df['Close'], 100)
        df['ema_200'] = ti.ema(df['Close'], 200)
        
        # Volume indicators
        df['volume_sma'] = ti.sma(df['Volume'], 20)
        df['obv'] = ti.obv(df['Close'], df['Volume'])
        
        # ADX
        df['adx'] = ti.adx(df['High'], df['Low'], df['Close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = ti.stochastic(df['High'], df['Low'], df['Close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ti.bollinger_bands(df['Close'])
        
        # Fibonacci levels (using 52-week high/low)
        high_52 = df['High'].rolling(window=252).max()
        low_52 = df['Low'].rolling(window=252).min()
        range_52 = high_52 - low_52
        
        df['fib_236'] = high_52 - (range_52 * 0.236)
        df['fib_382'] = high_52 - (range_52 * 0.382)
        df['fib_500'] = high_52 - (range_52 * 0.500)
        df['fib_618'] = high_52 - (range_52 * 0.618)
        
        return df
        
    def calculate_percentile_rank(self, series: pd.Series, window: int = 52) -> pd.Series:
        """Calculate percentile rank for a series over a rolling window"""
        def percentile_rank(x):
            if len(x) < 10:  # Need at least 10 data points
                return 50
            current_value = x.iloc[-1]
            if pd.isna(current_value):
                return 50
            # Calculate what percentile the current value is
            rank = (x < current_value).sum() / len(x) * 100
            return rank
        
        return series.rolling(window=window, min_periods=10).apply(percentile_rank, raw=False)
    
    def calculate_bullishness_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bullishness score based on methodology with improved logic"""
        scores = pd.Series(index=df.index, dtype=float)
        
        # Use shorter windows for more responsive scoring
        rsi_percentile = self.calculate_percentile_rank(df['rsi'], 52)
        volume_percentile = self.calculate_percentile_rank(df['Volume'], 52)
        obv_percentile = self.calculate_percentile_rank(df['obv'], 52)
        adx_percentile = self.calculate_percentile_rank(df['adx'], 52)
        stoch_percentile = self.calculate_percentile_rank(df['stoch_k'], 52)
        
        for i in range(len(df)):
            if i < 20:  # Need minimum data for indicators
                scores.iloc[i] = 50
                continue
                
            row = df.iloc[i]
            score = 0
            
            # RSI Score (9% weight) - Use direct RSI value with thresholds
            if not pd.isna(row['rsi']):
                if row['rsi'] > 70:
                    rsi_score = 80  # Overbought but still bullish
                elif row['rsi'] > 50:
                    rsi_score = 70
                elif row['rsi'] > 30:
                    rsi_score = 40
                else:
                    rsi_score = 20  # Oversold
                score += 0.09 * rsi_score
            
            # MACD Slope Score (20% weight)
            if not pd.isna(row['macd']) and not pd.isna(row['macd_signal']):
                if row['macd'] > row['macd_signal']:
                    # Additional check for MACD momentum
                    if i > 0 and not pd.isna(df.iloc[i-1]['macd']):
                        if row['macd'] > df.iloc[i-1]['macd']:
                            macd_score = 90  # Strong upward momentum
                        else:
                            macd_score = 70  # Bullish but weakening
                    else:
                        macd_score = 80
                else:
                    macd_score = 20  # Bearish
                score += 0.20 * macd_score
            
            # EMA Alignment Score (14% weight)
            ema_score = 0
            ema_count = 0
            if all(not pd.isna(row[col]) for col in ['ema_20', 'ema_50', 'ema_100', 'ema_200']):
                if row['Close'] > row['ema_20']:
                    ema_score += 25
                    ema_count += 1
                if row['ema_20'] > row['ema_50']:
                    ema_score += 25
                    ema_count += 1
                if row['ema_50'] > row['ema_100']:
                    ema_score += 25
                    ema_count += 1
                if row['ema_100'] > row['ema_200']:
                    ema_score += 25
                    ema_count += 1
                
                # Normalize based on available EMAs
                if ema_count > 0:
                    score += 0.14 * ema_score
            
            # Volume Score (18% weight) - Use volume ratio
            if not pd.isna(row['Volume']) and not pd.isna(row['volume_sma']) and row['volume_sma'] > 0:
                volume_ratio = row['Volume'] / row['volume_sma']
                if volume_ratio > 2:
                    volume_score = 90  # Very high volume
                elif volume_ratio > 1.5:
                    volume_score = 75
                elif volume_ratio > 1:
                    volume_score = 60
                else:
                    volume_score = 40  # Below average volume
                score += 0.18 * volume_score
            
            # OBV Momentum Score (13% weight) - Use OBV trend
            if not pd.isna(row['obv']) and i > 5:
                # Check OBV trend over last 5 periods
                obv_recent = df['obv'].iloc[max(0, i-5):i+1]
                if len(obv_recent) > 1:
                    obv_trend = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / abs(obv_recent.iloc[0]) if abs(obv_recent.iloc[0]) > 0 else 0
                    if obv_trend > 0.05:
                        obv_score = 80
                    elif obv_trend > 0:
                        obv_score = 65
                    elif obv_trend > -0.05:
                        obv_score = 50
                    else:
                        obv_score = 30
                    score += 0.13 * obv_score
                
            # Volume Oscillator Score (6% weight)
            if not pd.isna(row['volume_sma']) and row['volume_sma'] > 0:
                vol_osc = min(100, max(0, (row['Volume'] / row['volume_sma']) * 50))
                score += 0.06 * vol_osc
            
            # ADX Score (4% weight) - Direct ADX value
            if not pd.isna(row['adx']):
                if row['adx'] > 40:
                    adx_score = 90  # Strong trend
                elif row['adx'] > 25:
                    adx_score = 70  # Moderate trend
                else:
                    adx_score = 40  # Weak trend
                score += 0.04 * adx_score
            
            # Stochastic Score (5% weight)
            if not pd.isna(row['stoch_k']):
                if 20 < row['stoch_k'] < 80:
                    stoch_score = 70  # Good range
                elif row['stoch_k'] >= 80:
                    stoch_score = 50  # Overbought
                else:
                    stoch_score = 30  # Oversold
                score += 0.05 * stoch_score
            
            # Bollinger Bands Breakout Score (7% weight)
            if all(not pd.isna(row[col]) for col in ['Close', 'bb_upper', 'bb_middle', 'bb_lower']):
                if row['Close'] > row['bb_upper']:
                    bb_score = 90  # Breakout above
                elif row['Close'] > row['bb_middle']:
                    bb_score = 70  # Above middle
                elif row['Close'] > row['bb_lower']:
                    bb_score = 40  # Above lower band
                else:
                    bb_score = 20  # Below lower band
                score += 0.07 * bb_score
            
            # Fibonacci Score (4% weight)
            if not pd.isna(row['fib_500']) and not pd.isna(row['Close']):
                if row['Close'] > row['fib_236']:
                    fib_score = 90
                elif row['Close'] > row['fib_382']:
                    fib_score = 70
                elif row['Close'] > row['fib_500']:
                    fib_score = 50
                else:
                    fib_score = 30
                score += 0.04 * fib_score
            
            scores.iloc[i] = min(100, max(0, score))
        
        return scores
    
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df.empty:
                return None
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def update_stock_data(self, symbol: str):
        """Update stock data and indicators"""
        df = self.fetch_stock_data(symbol)
        if df is None:
            return
        
        df = self.calculate_indicators(df)
        df['bullishness_score'] = self.calculate_bullishness_score(df)
        
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Clear old data for this symbol to avoid duplicates
        cursor.execute('DELETE FROM stock_data WHERE symbol = ?', (symbol,))
        
        # Insert new data
        for idx, row in df.iterrows():
            cursor.execute('''
                INSERT INTO stock_data 
                (symbol, timestamp, open, high, low, close, volume, rsi, macd, macd_signal, 
                 macd_hist, ema_20, ema_50, ema_100, ema_200, volume_sma, obv, adx, 
                 stoch_k, stoch_d, bb_upper, bb_middle, bb_lower, fib_236, fib_382, 
                 fib_500, fib_618, bullishness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, idx.strftime('%Y-%m-%d %H:%M:%S'),
                row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
                row.get('rsi'), row.get('macd'), row.get('macd_signal'), row.get('macd_hist'),
                row.get('ema_20'), row.get('ema_50'), row.get('ema_100'), row.get('ema_200'),
                row.get('volume_sma'), row.get('obv'), row.get('adx'),
                row.get('stoch_k'), row.get('stoch_d'),
                row.get('bb_upper'), row.get('bb_middle'), row.get('bb_lower'),
                row.get('fib_236'), row.get('fib_382'), row.get('fib_500'), row.get('fib_618'),
                row.get('bullishness_score')
            ))
        
        conn.commit()
        conn.close()
        
        # Check for alerts
        latest_score = df['bullishness_score'].iloc[-1]
        if not pd.isna(latest_score):
            self.check_alerts(symbol, latest_score)
    
    def check_alerts(self, symbol: str, current_score: float):
        """Check for buy/sell alerts"""
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get previous score
        cursor.execute('''
            SELECT bullishness_score FROM stock_data 
            WHERE symbol = ? ORDER BY timestamp DESC LIMIT 2
        ''', (symbol,))
        scores = cursor.fetchall()
        
        if len(scores) >= 2:
            prev_score = scores[1][0]
            
            alert_message = None
            alert_type = None
            
            if current_score >= 80 and prev_score < 80:
                alert_message = f"{symbol} STRONG BUY signal - Bullishness Score: {current_score:.1f}"
                alert_type = "BUY"
            elif current_score >= 60 and prev_score < 60:
                alert_message = f"{symbol} BUY signal - Bullishness Score: {current_score:.1f}"
                alert_type = "BUY"
            elif current_score <= 20 and prev_score > 20:
                alert_message = f"{symbol} SELL signal - Bullishness Score: {current_score:.1f}"
                alert_type = "SELL"
            elif current_score <= 40 and prev_score > 40:
                alert_message = f"{symbol} WEAK signal - Bullishness Score: {current_score:.1f}"
                alert_type = "WEAK"
            
            if alert_message:
                # Save alert to database
                cursor.execute('''
                    INSERT INTO alerts (symbol, alert_type, message)
                    VALUES (?, ?, ?)
                ''', (symbol, alert_type, alert_message))
                
                # Send push notification
                self.send_push_notification(alert_message)
                
                # Emit to connected clients
                socketio.emit('alert', {
                    'symbol': symbol,
                    'type': alert_type,
                    'message': alert_message,
                    'score': current_score
                })
        
        conn.commit()
        conn.close()
    
    def send_push_notification(self, message: str):
        """Send push notification via Simplepush"""
        if not Config.SIMPLEPUSH_KEY:
            return
            
        try:
            url = "https://api.simplepush.io/send"
            data = {
                'key': Config.SIMPLEPUSH_KEY,
                'title': 'Stock Alert',
                'msg': message
            }
            requests.post(url, data=data)
        except Exception as e:
            print(f"Error sending push notification: {e}")

# Initialize analyzer
analyzer = StockAnalyzer()

# Background monitoring thread
def monitoring_thread():
    """Background thread for continuous monitoring"""
    while True:
        try:
            conn = sqlite3.connect(Config.DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM stocks WHERE active = 1")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            for symbol in symbols:
                print(f"Updating {symbol}...")
                analyzer.update_stock_data(symbol)
                time.sleep(2)  # Rate limiting
            
            print(f"Completed update cycle. Sleeping for {Config.UPDATE_INTERVAL} seconds...")
            time.sleep(Config.UPDATE_INTERVAL)
        except Exception as e:
            print(f"Error in monitoring thread: {e}")
            time.sleep(60)

# Start monitoring thread
monitor_thread = threading.Thread(target=monitoring_thread, daemon=True)
monitor_thread.start()

# Flask routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    
    # Get all active stocks with latest data (avoid duplicates)
    query = '''
        SELECT s.symbol, s.name, 
               sd.close, sd.bullishness_score, sd.timestamp
        FROM stocks s
        LEFT JOIN (
            SELECT symbol, close, bullishness_score, timestamp,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
            FROM stock_data
        ) sd ON s.symbol = sd.symbol AND sd.rn = 1
        WHERE s.active = 1
        ORDER BY sd.bullishness_score DESC NULLS LAST
    '''
    
    stocks_data = pd.read_sql_query(query, conn)
    conn.close()
    
    return render_template('dashboard.html', stocks=stocks_data.to_dict('records'))

@app.route('/delete_stock/<symbol>', methods=['POST'])
@auth.login_required
def delete_stock(symbol):
    """Delete a stock from tracking"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Set stock as inactive instead of deleting (preserves historical data)
        cursor.execute('UPDATE stocks SET active = 0 WHERE symbol = ?', (symbol,))
        
        # Or completely remove if you prefer:
        # cursor.execute('DELETE FROM stocks WHERE symbol = ?', (symbol,))
        # cursor.execute('DELETE FROM stock_data WHERE symbol = ?', (symbol,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/stock/<symbol>')
@auth.login_required
def stock_detail(symbol):
    """Detailed view for a specific stock"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    
    # Get recent data
    query = '''
        SELECT * FROM stock_data 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT 100
    '''
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    
    if df.empty:
        return "No data found", 404
    
    # Create charts
    fig = go.Figure()
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['close'],
        mode='lines',
        name='Price',
        yaxis='y'
    ))
    
    # Bullishness score
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['bullishness_score'],
        mode='lines',
        name='Bullishness Score',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'{symbol} Analysis',
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(title='Bullishness Score', side='right', overlaying='y'),
        xaxis=dict(title='Date')
    )
    
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('stock_detail.html', 
                         symbol=symbol, 
                         chart=chart_json,
                         current_data=df.iloc[0].to_dict())

@app.route('/backtest')
@auth.login_required
def backtest_page():
    """Backtesting interface"""
    return render_template('backtest.html')

@app.route('/api/backtest', methods=['POST'])
@auth.login_required
def run_backtest():
    """Run a backtest"""
    data = request.json
    symbol = data.get('symbol')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    buy_threshold = float(data.get('buy_threshold', 60))
    sell_threshold = float(data.get('sell_threshold', 40))
    
    conn = sqlite3.connect(Config.DATABASE_PATH)
    query = '''
        SELECT timestamp, close, bullishness_score 
        FROM stock_data 
        WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    '''
    df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
    conn.close()
    
    if df.empty:
        return jsonify({'error': 'No data found for the specified period'}), 400
    
    # Simple backtest logic
    position = 0
    portfolio_value = 10000
    cash = portfolio_value
    trades = []
    
    for idx, row in df.iterrows():
        if position == 0 and row['bullishness_score'] >= buy_threshold:
            # Buy
            shares = cash / row['close']
            position = shares
            cash = 0
            trades.append({
                'date': row['timestamp'],
                'action': 'BUY',
                'price': row['close'],
                'shares': shares
            })
        elif position > 0 and row['bullishness_score'] <= sell_threshold:
            # Sell
            cash = position * row['close']
            trades.append({
                'date': row['timestamp'],
                'action': 'SELL',
                'price': row['close'],
                'shares': position
            })
            position = 0
    
    # Final portfolio value
    final_value = cash + (position * df.iloc[-1]['close'])
    total_return = (final_value - portfolio_value) / portfolio_value * 100
    
    results = {
        'initial_value': portfolio_value,
        'final_value': final_value,
        'total_return': total_return,
        'trades': trades,
        'num_trades': len(trades)
    }
    
    return jsonify(results)

if __name__ == '__main__':
    print("Starting Stock Tracker...")
    print("Dashboard will be available at: http://localhost:5555")
    
    # Check if running in production
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    if is_production:
        # Production mode - use gevent
        socketio.run(app, 
                    host='0.0.0.0',
                    port=5555, 
                    debug=False,
                    allow_unsafe_werkzeug=True)
    else:
        # Development mode
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5555, 
                    debug=True)