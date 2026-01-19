"""
Indicator Registry: Catalog of 80+ technical indicators with metadata.

Each indicator is registered with:
- Name and category
- Required input columns (e.g., ['Close'], ['High', 'Low'], ['Open', 'High', 'Low', 'Close', 'Volume'])
- Default parameters
- Computation function

Groups:
- returns: Price transformations and returns
- trend: Moving averages and trend indicators
- volatility: Volatility and channel indicators
- momentum: Momentum oscillators
- volume: Volume-based indicators
- candle: Candlestick patterns (numeric proxies)
- regime: Statistical regime indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """Central registry of all technical indicators."""
    
    def __init__(self):
        self.indicators = {}
        self._register_all_indicators()
    
    def _register_all_indicators(self):
        """Register all indicators across all categories."""
        # A) Returns / Transformations
        self._register_returns_indicators()
        
        # B) Trend / Moving Averages
        self._register_trend_indicators()
        
        # C) Volatility / Channels
        self._register_volatility_indicators()
        
        # D) Momentum / Oscillators
        self._register_momentum_indicators()
        
        # E) Volume / Money Flow
        self._register_volume_indicators()
        
        # F) Candle Structure
        self._register_candle_indicators()
        
        # G) Regime / Statistical
        self._register_regime_indicators()
        
        logger.info(f"Registered {len(self.indicators)} indicators across {len(self.get_categories())} categories")
    
    def register(self, name: str, func: Callable, category: str, 
                 inputs: List[str], params: Dict[str, Any] = None):
        """Register a new indicator."""
        self.indicators[name] = {
            'func': func,
            'category': category,
            'inputs': inputs,
            'params': params or {}
        }
    
    def get_categories(self) -> List[str]:
        """Get unique categories."""
        return list(set(ind['category'] for ind in self.indicators.values()))
    
    def get_indicators_by_category(self, category: str) -> List[str]:
        """Get indicator names in a category."""
        return [name for name, ind in self.indicators.items() if ind['category'] == category]
    
    def compute(self, name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute an indicator."""
        if name not in self.indicators:
            raise ValueError(f"Indicator '{name}' not registered")
        
        ind = self.indicators[name]
        params = {**ind['params'], **kwargs}  # Override defaults with kwargs
        return ind['func'](data, **params)
    
    # ==================== A) RETURNS / TRANSFORMATIONS ====================
    
    def _register_returns_indicators(self):
        """Register returns and price transformation indicators."""
        
        def log_return(data: pd.DataFrame, period: int = 1) -> pd.Series:
            """Log return."""
            return np.log(data['Close'] / data['Close'].shift(period))
        
        def pct_return(data: pd.DataFrame, period: int = 1) -> pd.Series:
            """Percentage return."""
            return data['Close'].pct_change(period)
        
        def realized_vol(data: pd.DataFrame, window: int = 5) -> pd.Series:
            """Realized volatility (std of returns)."""
            returns = data['Close'].pct_change()
            return returns.rolling(window).std() * np.sqrt(252)
        
        def zscore_close(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Z-score of Close price."""
            rolling_mean = data['Close'].rolling(window).mean()
            rolling_std = data['Close'].rolling(window).std()
            return (data['Close'] - rolling_mean) / rolling_std.replace(0, np.nan)
        
        def dpo(data: pd.DataFrame, period: int = 20) -> pd.Series:
            """Detrended Price Oscillator."""
            sma = data['Close'].rolling(period).mean()
            return data['Close'].shift(period // 2 + 1) - sma
        
        def price_acceleration(data: pd.DataFrame) -> pd.Series:
            """Second derivative of returns."""
            returns = data['Close'].pct_change()
            return returns.diff()
        
        def gap_open_close(data: pd.DataFrame) -> pd.Series:
            """Gap between Open and previous Close."""
            return data['Open'] - data['Close'].shift(1)
        
        def intraday_range(data: pd.DataFrame) -> pd.Series:
            """High - Low."""
            return data['High'] - data['Low']
        
        # Register returns indicators
        for period in [1, 5, 10]:
            self.register(f'log_return_{period}', lambda d, p=period: log_return(d, p), 
                         'returns', ['Close'])
            self.register(f'pct_return_{period}', lambda d, p=period: pct_return(d, p),
                         'returns', ['Close'])
        
        for window in [5, 10, 20]:
            self.register(f'realized_vol_{window}', lambda d, w=window: realized_vol(d, w),
                         'returns', ['Close'])
            self.register(f'zscore_close_{window}', lambda d, w=window: zscore_close(d, w),
                         'returns', ['Close'])
        
        self.register('dpo_20', dpo, 'returns', ['Close'])
        self.register('price_acceleration', price_acceleration, 'returns', ['Close'])
        self.register('gap_open_close', gap_open_close, 'returns', ['Open', 'Close'])
        self.register('intraday_range', intraday_range, 'returns', ['High', 'Low'])
    
    # ==================== B) TREND / MOVING AVERAGES ====================
    
    def _register_trend_indicators(self):
        """Register trend and moving average indicators."""
        
        def sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Simple Moving Average."""
            return data['Close'].rolling(window).mean()
        
        def ema(data: pd.DataFrame, span: int = 20) -> pd.Series:
            """Exponential Moving Average."""
            return data['Close'].ewm(span=span, adjust=False).mean()
        
        def wma(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Weighted Moving Average."""
            weights = np.arange(1, window + 1)
            return data['Close'].rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        def vwma(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Volume Weighted Moving Average."""
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            return (typical_price * data['Volume']).rolling(window).sum() / data['Volume'].rolling(window).sum()
        
        def hma(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Hull Moving Average."""
            half_length = window // 2
            sqrt_length = int(np.sqrt(window))
            wma1 = data['Close'].rolling(half_length).mean()
            wma2 = data['Close'].rolling(window).mean()
            return (2 * wma1 - wma2).rolling(sqrt_length).mean()
        
        def dema(data: pd.DataFrame, span: int = 20) -> pd.Series:
            """Double Exponential Moving Average."""
            ema1 = data['Close'].ewm(span=span, adjust=False).mean()
            ema2 = ema1.ewm(span=span, adjust=False).mean()
            return 2 * ema1 - ema2
        
        def tema(data: pd.DataFrame, span: int = 20) -> pd.Series:
            """Triple Exponential Moving Average."""
            ema1 = data['Close'].ewm(span=span, adjust=False).mean()
            ema2 = ema1.ewm(span=span, adjust=False).mean()
            ema3 = ema2.ewm(span=span, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        
        def kama(data: pd.DataFrame, window: int = 10) -> pd.Series:
            """Kaufman's Adaptive Moving Average."""
            change = abs(data['Close'] - data['Close'].shift(window))
            volatility = data['Close'].diff().abs().rolling(window).sum()
            er = change / volatility.replace(0, np.nan)  # Efficiency Ratio
            fast_sc = 2 / (2 + 1)  # Fast smoothing constant
            slow_sc = 2 / (30 + 1)  # Slow smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama = pd.Series(index=data.index, dtype=float)
            kama.iloc[window] = data['Close'].iloc[window]
            for i in range(window + 1, len(data)):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data['Close'].iloc[i] - kama.iloc[i-1])
            return kama
        
        def ma_cross_signal(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
            """MA crossover signal (1 if fast > slow, else 0)."""
            sma_fast = data['Close'].rolling(fast).mean()
            sma_slow = data['Close'].rolling(slow).mean()
            return (sma_fast > sma_slow).astype(int)
        
        def ma_spread(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
            """MA spread ratio."""
            sma_fast = data['Close'].rolling(fast).mean()
            sma_slow = data['Close'].rolling(slow).mean()
            return (sma_fast - sma_slow) / data['Close']
        
        def slope_ma(data: pd.DataFrame, window: int = 20, lookback: int = 5) -> pd.Series:
            """Slope of MA via linear fit."""
            ma = data['Close'].rolling(window).mean()
            def linear_slope(y):
                if len(y) < lookback:
                    return np.nan
                x = np.arange(len(y))
                return np.polyfit(x, y, 1)[0]
            return ma.rolling(lookback).apply(linear_slope, raw=True)
        
        def price_to_ma_ratio(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Price to MA ratio."""
            ma = data['Close'].rolling(window).mean()
            return data['Close'] / ma - 1
        
        # Register SMA for multiple windows
        for window in [5, 10, 20, 50, 200]:
            self.register(f'sma_{window}', lambda d, w=window: sma(d, w), 'trend', ['Close'])
            self.register(f'sma_{window}_ratio', lambda d, w=window: price_to_ma_ratio(d, w), 'trend', ['Close'])
        
        # Register EMA for multiple spans
        for span in [5, 10, 20, 50, 200]:
            self.register(f'ema_{span}', lambda d, s=span: ema(d, s), 'trend', ['Close'])
            self.register(f'ema_{span}_ratio', lambda d, s=span: d['Close'] / ema(d, s) - 1, 'trend', ['Close'])
        
        # Register other MAs
        self.register('wma_20', wma, 'trend', ['Close'])
        self.register('vwma_20', vwma, 'trend', ['High', 'Low', 'Close', 'Volume'])
        self.register('hma_20', hma, 'trend', ['Close'])
        self.register('dema_20', dema, 'trend', ['Close'])
        self.register('tema_20', tema, 'trend', ['Close'])
        self.register('kama_10', lambda d: kama(d, 10), 'trend', ['Close'])
        self.register('kama_20', lambda d: kama(d, 20), 'trend', ['Close'])
        
        # Register derived indicators
        self.register('ma_cross_20_50', lambda d: ma_cross_signal(d, 20, 50), 'trend', ['Close'])
        self.register('ma_spread_20_50', lambda d: ma_spread(d, 20, 50), 'trend', ['Close'])
        self.register('slope_sma_20', lambda d: slope_ma(d, 20, 5), 'trend', ['Close'])
        self.register('price_to_sma_20_ratio', lambda d: price_to_ma_ratio(d, 20), 'trend', ['Close'])
        self.register('price_to_ema_20_ratio', lambda d: d['Close'] / ema(d, 20) - 1, 'trend', ['Close'])
    
    # ==================== C) VOLATILITY / CHANNELS ====================
    
    def _register_volatility_indicators(self):
        """Register volatility and channel indicators."""
        
        def true_range(data: pd.DataFrame) -> pd.Series:
            """True Range."""
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift(1))
            low_close = abs(data['Low'] - data['Close'].shift(1))
            return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Average True Range."""
            tr = true_range(data)
            return tr.rolling(window).mean()
        
        def atr_normalized(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Normalized Average True Range (ATR / Close)."""
            return atr(data, window) / data['Close']
        
        def bbands_width(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.Series:
            """Bollinger Bands width."""
            ma = data['Close'].rolling(window).mean()
            std = data['Close'].rolling(window).std()
            upper = ma + num_std * std
            lower = ma - num_std * std
            return (upper - lower) / ma
        
        def bbands_percent_b(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.Series:
            """Bollinger Bands %B (position within bands)."""
            ma = data['Close'].rolling(window).mean()
            std = data['Close'].rolling(window).std()
            upper = ma + num_std * std
            lower = ma - num_std * std
            return (data['Close'] - lower) / (upper - lower).replace(0, np.nan)
        
        def keltner_width(data: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.Series:
            """Keltner Channel width."""
            ma = data['Close'].rolling(window).mean()
            atr_val = atr(data, window)
            upper = ma + atr_mult * atr_val
            lower = ma - atr_mult * atr_val
            return (upper - lower) / ma
        
        def keltner_position(data: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.Series:
            """Position within Keltner Channels."""
            ma = data['Close'].rolling(window).mean()
            atr_val = atr(data, window)
            upper = ma + atr_mult * atr_val
            lower = ma - atr_mult * atr_val
            return (data['Close'] - lower) / (upper - lower).replace(0, np.nan)
        
        def donchian_width(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Donchian Channel width."""
            upper = data['High'].rolling(window).max()
            lower = data['Low'].rolling(window).min()
            return (upper - lower) / data['Close']
        
        def donchian_position(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Position within Donchian Channels."""
            upper = data['High'].rolling(window).max()
            lower = data['Low'].rolling(window).min()
            return (data['Close'] - lower) / (upper - lower).replace(0, np.nan)
        
        def historical_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Historical volatility (annualized std of returns)."""
            returns = data['Close'].pct_change()
            return returns.rolling(window).std() * np.sqrt(252)
        
        def parkinson_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Parkinson volatility (High/Low based)."""
            hl = np.log(data['High'] / data['Low'])
            return hl.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2))), raw=True) * np.sqrt(252)
        
        def garman_klass_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Garman-Klass volatility (OHLC based)."""
            hl = np.log(data['High'] / data['Low']) ** 2
            co = np.log(data['Close'] / data['Open']) ** 2
            return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window).mean()) * np.sqrt(252)
        
        def chaikin_volatility(data: pd.DataFrame, window: int = 10, rate_period: int = 10) -> pd.Series:
            """Chaikin Volatility."""
            hl_ema = (data['High'] - data['Low']).ewm(span=window, adjust=False).mean()
            return hl_ema.pct_change(rate_period)
        
        # Register volatility indicators
        self.register('true_range', true_range, 'volatility', ['High', 'Low', 'Close'])
        self.register('atr_14', lambda d: atr(d, 14), 'volatility', ['High', 'Low', 'Close'])
        self.register('atr_normalized_14', lambda d: atr_normalized(d, 14), 'volatility', ['High', 'Low', 'Close'])
        self.register('bbands_width_20', lambda d: bbands_width(d, 20), 'volatility', ['Close'])
        self.register('bbands_percent_b_20', lambda d: bbands_percent_b(d, 20), 'volatility', ['Close'])
        self.register('keltner_width_20', lambda d: keltner_width(d, 20), 'volatility', ['High', 'Low', 'Close'])
        self.register('keltner_position_20', lambda d: keltner_position(d, 20), 'volatility', ['High', 'Low', 'Close'])
        self.register('donchian_width_20', lambda d: donchian_width(d, 20), 'volatility', ['High', 'Low', 'Close'])
        self.register('donchian_position_20', lambda d: donchian_position(d, 20), 'volatility', ['High', 'Low', 'Close'])
        
        for window in [20, 50]:
            self.register(f'historical_volatility_{window}', lambda d, w=window: historical_volatility(d, w),
                         'volatility', ['Close'])
        
        self.register('parkinson_volatility_20', lambda d: parkinson_volatility(d, 20), 'volatility', ['High', 'Low'])
        self.register('garman_klass_volatility_20', lambda d: garman_klass_volatility(d, 20),
                     'volatility', ['Open', 'High', 'Low', 'Close'])
        self.register('chaikin_volatility_10', chaikin_volatility, 'volatility', ['High', 'Low'])
    
    # ==================== D) MOMENTUM / OSCILLATORS ====================
    
    def _register_momentum_indicators(self):
        """Register momentum oscillators (RSI, MACD, Stochastic, etc.)."""
        
        def rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Relative Strength Index."""
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        
        def stochastic_k(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Stochastic %K."""
            low_min = data['Low'].rolling(window).min()
            high_max = data['High'].rolling(window).max()
            return 100 * (data['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        
        def stochastic_d(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.Series:
            """Stochastic %D (smoothed %K)."""
            k = stochastic_k(data, k_window)
            return k.rolling(d_window).mean()
        
        def stochastic_rsi(data: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.Series:
            """Stochastic RSI."""
            rsi_val = rsi(data, window)
            rsi_min = rsi_val.rolling(window).min()
            rsi_max = rsi_val.rolling(window).max()
            stoch_rsi = 100 * (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
            return stoch_rsi.rolling(smooth).mean()
        
        def macd_line(data: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
            """MACD line."""
            ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
            return ema_fast - ema_slow
        
        def macd_signal(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
            """MACD signal line."""
            macd = macd_line(data, fast, slow)
            return macd.ewm(span=signal, adjust=False).mean()
        
        def macd_histogram(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
            """MACD histogram."""
            macd = macd_line(data, fast, slow)
            signal_line = macd_signal(data, fast, slow, signal)
            return macd - signal_line
        
        def cci(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Commodity Channel Index."""
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma = typical_price.rolling(window).mean()
            mad = typical_price.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            return (typical_price - sma) / (0.015 * mad).replace(0, np.nan)
        
        def williams_r(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Williams %R."""
            high_max = data['High'].rolling(window).max()
            low_min = data['Low'].rolling(window).min()
            return -100 * (high_max - data['Close']) / (high_max - low_min).replace(0, np.nan)
        
        def roc(data: pd.DataFrame, period: int = 10) -> pd.Series:
            """Rate of Change."""
            return 100 * (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period).replace(0, np.nan)
        
        def momentum(data: pd.DataFrame, period: int = 10) -> pd.Series:
            """Momentum."""
            return data['Close'] - data['Close'].shift(period)
        
        def trix(data: pd.DataFrame, period: int = 15) -> pd.Series:
            """TRIX (Triple Exponential Average)."""
            ema1 = data['Close'].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return 100 * ema3.pct_change()
        
        def ultimate_oscillator(data: pd.DataFrame, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
            """Ultimate Oscillator."""
            bp = data['Close'] - pd.concat([data['Low'], data['Close'].shift(1)], axis=1).min(axis=1)
            tr = pd.concat([
                data['High'] - data['Low'],
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            ], axis=1).max(axis=1)
            
            avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
            avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
            avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
            
            return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        
        def awesome_oscillator(data: pd.DataFrame) -> pd.Series:
            """Awesome Oscillator."""
            median = (data['High'] + data['Low']) / 2
            ao = median.rolling(5).mean() - median.rolling(34).mean()
            return ao
        
        def ppo(data: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
            """Percentage Price Oscillator."""
            ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
            return 100 * (ema_fast - ema_slow) / ema_slow
        
        def dmi_plus(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Directional Movement Index +."""
            high_diff = data['High'].diff()
            low_diff = -data['Low'].diff()
            
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            
            tr = pd.concat([
                data['High'] - data['Low'],
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            ], axis=1).max(axis=1)
            
            atr_val = tr.rolling(window).mean()
            return 100 * pos_dm.rolling(window).mean() / atr_val.replace(0, np.nan)
        
        def dmi_minus(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Directional Movement Index -."""
            high_diff = data['High'].diff()
            low_diff = -data['Low'].diff()
            
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = pd.concat([
                data['High'] - data['Low'],
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            ], axis=1).max(axis=1)
            
            atr_val = tr.rolling(window).mean()
            return 100 * neg_dm.rolling(window).mean() / atr_val.replace(0, np.nan)
        
        def adx(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Average Directional Index."""
            di_plus = dmi_plus(data, window)
            di_minus = dmi_minus(data, window)
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
            return dx.rolling(window).mean()
        
        def aroon_up(data: pd.DataFrame, window: int = 25) -> pd.Series:
            """Aroon Up."""
            return 100 * data['High'].rolling(window + 1).apply(lambda x: x.argmax(), raw=True) / window
        
        def aroon_down(data: pd.DataFrame, window: int = 25) -> pd.Series:
            """Aroon Down."""
            return 100 * data['Low'].rolling(window + 1).apply(lambda x: x.argmin(), raw=True) / window
        
        def aroon_oscillator(data: pd.DataFrame, window: int = 25) -> pd.Series:
            """Aroon Oscillator."""
            return aroon_up(data, window) - aroon_down(data, window)
        
        def vortex_plus(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Vortex Indicator VI+."""
            vm_plus = abs(data['High'] - data['Low'].shift(1)).rolling(window).sum()
            tr = pd.concat([
                data['High'] - data['Low'],
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            ], axis=1).max(axis=1).rolling(window).sum()
            return vm_plus / tr.replace(0, np.nan)
        
        def vortex_minus(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Vortex Indicator VI-."""
            vm_minus = abs(data['Low'] - data['High'].shift(1)).rolling(window).sum()
            tr = pd.concat([
                data['High'] - data['Low'],
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            ], axis=1).max(axis=1).rolling(window).sum()
            return vm_minus / tr.replace(0, np.nan)
        
        def kst(data: pd.DataFrame) -> pd.Series:
            """Know Sure Thing (KST)."""
            roc1 = roc(data, 10).rolling(10).mean()
            roc2 = roc(data, 15).rolling(10).mean()
            roc3 = roc(data, 20).rolling(10).mean()
            roc4 = roc(data, 30).rolling(15).mean()
            return roc1 + 2 * roc2 + 3 * roc3 + 4 * roc4
        
        def tsi(data: pd.DataFrame, long_period: int = 25, short_period: int = 13) -> pd.Series:
            """True Strength Index."""
            price_change = data['Close'].diff()
            double_smoothed_pc = price_change.ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
            double_smoothed_abs_pc = abs(price_change).ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
            return 100 * double_smoothed_pc / double_smoothed_abs_pc.replace(0, np.nan)
        
        def fisher_transform(data: pd.DataFrame, window: int = 10) -> pd.Series:
            """Fisher Transform."""
            high_max = data['High'].rolling(window).max()
            low_min = data['Low'].rolling(window).min()
            value = 2 * ((data['Close'] - low_min) / (high_max - low_min).replace(0, np.nan) - 0.5)
            value = value.clip(-0.999, 0.999)
            return 0.5 * np.log((1 + value) / (1 - value))
        
        def elder_ray_bull(data: pd.DataFrame, window: int = 13) -> pd.Series:
            """Elder Ray Bull Power."""
            ema = data['Close'].ewm(span=window, adjust=False).mean()
            return data['High'] - ema
        
        def elder_ray_bear(data: pd.DataFrame, window: int = 13) -> pd.Series:
            """Elder Ray Bear Power."""
            ema = data['Close'].ewm(span=window, adjust=False).mean()
            return data['Low'] - ema
        
        # Register momentum indicators
        for window in [14, 28]:
            self.register(f'rsi_{window}', lambda d, w=window: rsi(d, w), 'momentum', ['Close'])
        
        self.register('stoch_k_14', lambda d: stochastic_k(d, 14), 'momentum', ['High', 'Low', 'Close'])
        self.register('stoch_d_14_3', lambda d: stochastic_d(d, 14, 3), 'momentum', ['High', 'Low', 'Close'])
        self.register('stoch_rsi_14', lambda d: stochastic_rsi(d, 14), 'momentum', ['Close'])
        
        self.register('macd_12_26_9', lambda d: macd_line(d, 12, 26), 'momentum', ['Close'])
        self.register('macd_12_26_9_signal', lambda d: macd_signal(d, 12, 26, 9), 'momentum', ['Close'])
        self.register('macd_12_26_9_hist', lambda d: macd_histogram(d, 12, 26, 9), 'momentum', ['Close'])
        
        self.register('cci_20', lambda d: cci(d, 20), 'momentum', ['High', 'Low', 'Close'])
        self.register('williams_r_14', lambda d: williams_r(d, 14), 'momentum', ['High', 'Low', 'Close'])
        
        for period in [10, 20]:
            self.register(f'roc_{period}', lambda d, p=period: roc(d, p), 'momentum', ['Close'])
            self.register(f'momentum_{period}', lambda d, p=period: momentum(d, p), 'momentum', ['Close'])
        
        self.register('trix_15', lambda d: trix(d, 15), 'momentum', ['Close'])
        self.register('ultimate_osc', ultimate_oscillator, 'momentum', ['High', 'Low', 'Close'])
        self.register('awesome_osc', awesome_oscillator, 'momentum', ['High', 'Low'])
        self.register('ppo', ppo, 'momentum', ['Close'])
        
        self.register('dmi_plus_14', lambda d: dmi_plus(d, 14), 'momentum', ['High', 'Low', 'Close'])
        self.register('dmi_minus_14', lambda d: dmi_minus(d, 14), 'momentum', ['High', 'Low', 'Close'])
        self.register('adx_14', lambda d: adx(d, 14), 'momentum', ['High', 'Low', 'Close'])
        
        self.register('aroon_up_25', lambda d: aroon_up(d, 25), 'momentum', ['High'])
        self.register('aroon_down_25', lambda d: aroon_down(d, 25), 'momentum', ['Low'])
        self.register('aroon_osc_25', lambda d: aroon_oscillator(d, 25), 'momentum', ['High', 'Low'])
        
        self.register('vortex_plus_14', lambda d: vortex_plus(d, 14), 'momentum', ['High', 'Low', 'Close'])
        self.register('vortex_minus_14', lambda d: vortex_minus(d, 14), 'momentum', ['High', 'Low', 'Close'])
        
        self.register('kst', kst, 'momentum', ['Close'])
        self.register('tsi_25_13', lambda d: tsi(d, 25, 13), 'momentum', ['Close'])
        self.register('fisher_transform_10', lambda d: fisher_transform(d, 10), 'momentum', ['High', 'Low', 'Close'])
        
        self.register('elder_ray_bull_13', lambda d: elder_ray_bull(d, 13), 'momentum', ['High', 'Close'])
        self.register('elder_ray_bear_13', lambda d: elder_ray_bear(d, 13), 'momentum', ['Low', 'Close'])
    
    # ==================== E) VOLUME / MONEY FLOW ====================
    
    def _register_volume_indicators(self):
        """Register volume-based indicators."""
        
        def obv(data: pd.DataFrame) -> pd.Series:
            """On-Balance Volume."""
            return (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        
        def ad_line(data: pd.DataFrame) -> pd.Series:
            """Accumulation/Distribution Line."""
            clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']).replace(0, np.nan)
            return (clv * data['Volume']).fillna(0).cumsum()
        
        def cmf(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Chaikin Money Flow."""
            mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']).replace(0, np.nan) * data['Volume']
            return mfv.rolling(window).sum() / data['Volume'].rolling(window).sum()
        
        def mfi(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Money Flow Index."""
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            money_flow = typical_price * data['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
            
            mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
            return 100 - (100 / (1 + mfi_ratio))
        
        def force_index(data: pd.DataFrame, window: int = 13) -> pd.Series:
            """Force Index."""
            return (data['Close'].diff() * data['Volume']).ewm(span=window, adjust=False).mean()
        
        def eom(data: pd.DataFrame, window: int = 14) -> pd.Series:
            """Ease of Movement."""
            distance = ((data['High'] + data['Low']) / 2).diff()
            box_ratio = data['Volume'] / (data['High'] - data['Low']).replace(0, np.nan) / 100000000
            return (distance / box_ratio).rolling(window).mean()
        
        def volume_roc(data: pd.DataFrame, period: int = 10) -> pd.Series:
            """Volume Rate of Change."""
            return 100 * (data['Volume'] - data['Volume'].shift(period)) / data['Volume'].shift(period).replace(0, np.nan)
        
        def pvt(data: pd.DataFrame) -> pd.Series:
            """Price-Volume Trend."""
            return (data['Close'].pct_change() * data['Volume']).fillna(0).cumsum()
        
        def kvo(data: pd.DataFrame, fast: int = 34, slow: int = 55) -> pd.Series:
            """Klinger Volume Oscillator."""
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            dm = data['High'] - data['Low']
            trend = np.where(typical_price > typical_price.shift(1), 1, -1)
            
            vf = pd.Series(trend, index=data.index) * data['Volume']
            kvo_val = vf.ewm(span=fast, adjust=False).mean() - vf.ewm(span=slow, adjust=False).mean()
            return kvo_val
        
        def vwap_proxy(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """VWAP proxy (rolling typical price * volume)."""
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            return (typical_price * data['Volume']).rolling(window).sum() / data['Volume'].rolling(window).sum()
        
        def volume_zscore(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Volume Z-score."""
            vol_mean = data['Volume'].rolling(window).mean()
            vol_std = data['Volume'].rolling(window).std()
            return (data['Volume'] - vol_mean) / vol_std.replace(0, np.nan)
        
        def volume_spike(data: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
            """Volume spike flag (volume > mean + 2*std)."""
            vol_mean = data['Volume'].rolling(window).mean()
            vol_std = data['Volume'].rolling(window).std()
            return (data['Volume'] > vol_mean + threshold * vol_std).astype(int)
        
        # Register volume indicators
        self.register('obv', obv, 'volume', ['Close', 'Volume'])
        self.register('ad_line', ad_line, 'volume', ['High', 'Low', 'Close', 'Volume'])
        self.register('cmf_20', lambda d: cmf(d, 20), 'volume', ['High', 'Low', 'Close', 'Volume'])
        self.register('mfi_14', lambda d: mfi(d, 14), 'volume', ['High', 'Low', 'Close', 'Volume'])
        self.register('force_index_13', lambda d: force_index(d, 13), 'volume', ['Close', 'Volume'])
        self.register('eom_14', lambda d: eom(d, 14), 'volume', ['High', 'Low', 'Volume'])
        self.register('volume_roc_10', lambda d: volume_roc(d, 10), 'volume', ['Volume'])
        self.register('pvt', pvt, 'volume', ['Close', 'Volume'])
        self.register('kvo_34_55', lambda d: kvo(d, 34, 55), 'volume', ['High', 'Low', 'Close', 'Volume'])
        self.register('vwap_proxy_20', lambda d: vwap_proxy(d, 20), 'volume', ['High', 'Low', 'Close', 'Volume'])
        self.register('volume_zscore_20', lambda d: volume_zscore(d, 20), 'volume', ['Volume'])
        self.register('volume_spike_20', lambda d: volume_spike(d, 20), 'volume', ['Volume'])
    
    # ==================== F) CANDLE / PRICE ACTION ====================
    
    def _register_candle_indicators(self):
        """Register candlestick pattern indicators."""
        
        def candle_body(data: pd.DataFrame) -> pd.Series:
            """Candle body (Close - Open)."""
            return data['Close'] - data['Open']
        
        def upper_wick(data: pd.DataFrame) -> pd.Series:
            """Upper wick length."""
            return data['High'] - pd.concat([data['Close'], data['Open']], axis=1).max(axis=1)
        
        def lower_wick(data: pd.DataFrame) -> pd.Series:
            """Lower wick length."""
            return pd.concat([data['Close'], data['Open']], axis=1).min(axis=1) - data['Low']
        
        def body_to_range_ratio(data: pd.DataFrame) -> pd.Series:
            """Body to range ratio."""
            body = abs(data['Close'] - data['Open'])
            range_val = data['High'] - data['Low']
            return body / range_val.replace(0, np.nan)
        
        def close_position_in_range(data: pd.DataFrame) -> pd.Series:
            """Close position within range (0=Low, 1=High)."""
            return (data['Close'] - data['Low']) / (data['High'] - data['Low']).replace(0, np.nan)
        
        def breakout_20(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Breakout flag (Close > rolling max)."""
            return (data['Close'] > data['High'].shift(1).rolling(window).max()).astype(int)
        
        def breakdown_20(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Breakdown flag (Close < rolling min)."""
            return (data['Close'] < data['Low'].shift(1).rolling(window).min()).astype(int)
        
        def inside_bar(data: pd.DataFrame) -> pd.Series:
            """Inside bar flag (High < prev High AND Low > prev Low)."""
            return ((data['High'] < data['High'].shift(1)) & (data['Low'] > data['Low'].shift(1))).astype(int)
        
        def engulfing_proxy(data: pd.DataFrame) -> pd.Series:
            """Engulfing proxy (current body > 1.5x previous body)."""
            body = abs(data['Close'] - data['Open'])
            prev_body = body.shift(1)
            return (body > 1.5 * prev_body).astype(int)
        
        # Register candle indicators
        self.register('candle_body', candle_body, 'candle', ['Open', 'Close'])
        self.register('upper_wick', upper_wick, 'candle', ['High', 'Open', 'Close'])
        self.register('lower_wick', lower_wick, 'candle', ['Low', 'Open', 'Close'])
        self.register('body_to_range_ratio', body_to_range_ratio, 'candle', ['Open', 'High', 'Low', 'Close'])
        self.register('close_pos_in_range', close_position_in_range, 'candle', ['High', 'Low', 'Close'])
        self.register('breakout_20', lambda d: breakout_20(d, 20), 'candle', ['High', 'Close'])
        self.register('breakdown_20', lambda d: breakdown_20(d, 20), 'candle', ['Low', 'Close'])
        self.register('inside_bar', inside_bar, 'candle', ['High', 'Low'])
        self.register('engulfing_proxy', engulfing_proxy, 'candle', ['Open', 'Close'])
    
    # ==================== G) REGIME / STATISTICAL ====================
    
    def _register_regime_indicators(self):
        """Register regime and statistical indicators."""
        
        def hurst_approx(data: pd.DataFrame, window: int = 100) -> pd.Series:
            """Approximate Hurst exponent."""
            def calc_hurst(prices):
                if len(prices) < 20:
                    return np.nan
                try:
                    lags = range(2, min(20, len(prices) // 2))
                    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0]
                except:
                    return np.nan
            return data['Close'].rolling(window).apply(calc_hurst, raw=True)
        
        def autocorr_return(data: pd.DataFrame, lag: int = 1, window: int = 50) -> pd.Series:
            """Autocorrelation of returns."""
            returns = data['Close'].pct_change()
            return returns.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)
        
        def skew_return(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Skewness of returns."""
            returns = data['Close'].pct_change()
            return returns.rolling(window).skew()
        
        def kurt_return(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Kurtosis of returns."""
            returns = data['Close'].pct_change()
            return returns.rolling(window).kurt()
        
        def rolling_drawdown(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Rolling drawdown."""
            rolling_max = data['Close'].rolling(window).max()
            return (data['Close'] - rolling_max) / rolling_max.replace(0, np.nan)
        
        def dist_to_rolling_max(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Distance to rolling max."""
            rolling_max = data['Close'].rolling(window).max()
            return (data['Close'] - rolling_max) / data['Close'].replace(0, np.nan)
        
        def dist_to_rolling_min(data: pd.DataFrame, window: int = 20) -> pd.Series:
            """Distance to rolling min."""
            rolling_min = data['Close'].rolling(window).min()
            return (data['Close'] - rolling_min) / data['Close'].replace(0, np.nan)
        
        def entropy_proxy(data: pd.DataFrame, window: int = 20, bins: int = 10) -> pd.Series:
            """Entropy proxy (simplified)."""
            def calc_entropy(x):
                if len(x) < bins:
                    return np.nan
                hist, _ = np.histogram(x, bins=bins)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                return -np.sum(hist * np.log(hist))
            
            returns = data['Close'].pct_change()
            return returns.rolling(window).apply(calc_entropy, raw=True)
        
        def change_point_score(data: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
            """Change point score (abs zscore of returns > threshold)."""
            returns = data['Close'].pct_change()
            zscore = (returns - returns.rolling(window).mean()) / returns.rolling(window).std().replace(0, np.nan)
            return (abs(zscore) > threshold).astype(int)
        
        # Register regime indicators
        self.register('hurst_100', lambda d: hurst_approx(d, 100), 'regime', ['Close'])
        self.register('autocorr_return_1_50', lambda d: autocorr_return(d, 1, 50), 'regime', ['Close'])
        self.register('skew_return_20', lambda d: skew_return(d, 20), 'regime', ['Close'])
        self.register('kurt_return_20', lambda d: kurt_return(d, 20), 'regime', ['Close'])
        self.register('rolling_drawdown_20', lambda d: rolling_drawdown(d, 20), 'regime', ['Close'])
        self.register('dist_to_rolling_max_20', lambda d: dist_to_rolling_max(d, 20), 'regime', ['Close'])
        self.register('dist_to_rolling_min_20', lambda d: dist_to_rolling_min(d, 20), 'regime', ['Close'])
        self.register('entropy_proxy_20', lambda d: entropy_proxy(d, 20), 'regime', ['Close'])
        self.register('change_point_score_20', lambda d: change_point_score(d, 20), 'regime', ['Close'])


# Global registry instance
INDICATOR_REGISTRY = IndicatorRegistry()
