"""
DataLoader: Download stock and cryptocurrency historical data.

Data Sources:
  - yfinance: Stocks and crypto (limited historical data for intraday)
  - binance: Crypto only (years of historical intraday data, no API key needed)

API Constraints (yfinance):
  - 1m: max 7 days | 2m-90m: max 60 days | 1h: max 730 days | 1d+: unlimited

Binance Advantages:
  - Years of 1m, 5m, 15m, 1h data available (vs 60 days on Yahoo)
  - Free public API, no authentication required
  - More reliable for crypto data

Extended Download Mode:
  - Set extended_download: true to bypass API limits by making multiple requests
  - Useful for getting >60 days of 5m data (e.g., 1.5 years)
  - Requests are chunked and concatenated automatically

Usage:
  config = {
      "keys": ["BTC", "AAPL"],
      "category": "crypto" or "stock",
      "candle_length": "5m" (or "5min", "1h", "1d", etc.),
      "period": 1.5,  # years - can exceed API limit with extended_download
      "extended_download": true,  # Enable chunked downloading
      "data_source": "binance"  # Use "binance" for crypto, "yfinance" for stocks
  }
  loader = DataLoader(config)
  data = loader.assemble_data()
"""

import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, List, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Stock and crypto data loader with automatic interval-period validation."""
    
    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    INTERVAL_ALIASES = {
        '5min': '5m', '1min': '1m', '2min': '2m', '15min': '15m', '30min': '30m',
        '60min': '60m', '90min': '90m', '1hour': '1h', '1day': '1d', '5day': '5d',
        '1week': '1wk', '1month': '1mo', '3month': '3mo',
    }
    
    INTERVAL_LIMITS = {  # Maximum days for each interval (yfinance API limits)
        # Note: Using slightly less than actual limits to avoid edge cases
        '1m': 6, '2m': 58, '5m': 58, '15m': 58, '30m': 58, '60m': 58, '90m': 58,
        '1h': 725, '1d': 36500, '5d': 36500, '1wk': 36500, '1mo': 36500, '3mo': 36500,
    }
    
    # Binance interval mapping and limits (supports years of data)
    BINANCE_INTERVALS = {
        '1m': '1m', '2m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
        '60m': '1h', '1h': '1h', '1d': '1d', '1wk': '1w', '1mo': '1M',
    }
    
    BINANCE_INTERVAL_LIMITS = {  # Binance allows much longer history
        '1m': 1000, '5m': 1500, '15m': 1500, '30m': 1500,
        '60m': 1500, '1h': 1500, '1d': 36500, '1wk': 36500, '1mo': 36500,
    }
    
    # Common crypto symbol mappings for Binance
    BINANCE_SYMBOLS = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'XRP': 'XRPUSDT', 'SOL': 'SOLUSDT',
        'ADA': 'ADAUSDT', 'DOGE': 'DOGEUSDT', 'DOT': 'DOTUSDT', 'MATIC': 'MATICUSDT',
        'LTC': 'LTCUSDT', 'SHIB': 'SHIBUSDT', 'AVAX': 'AVAXUSDT', 'LINK': 'LINKUSDT',
        'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT', 'UNI': 'UNIUSDT', 'ETC': 'ETCUSDT',
        'BCH': 'BCHUSDT', 'FIL': 'FILUSDT', 'APT': 'APTUSDT', 'ARB': 'ARBUSDT',
        'OP': 'OPUSDT', 'NEAR': 'NEARUSDT', 'VET': 'VETUSDT', 'ALGO': 'ALGOUSDT',
    }
    
    CRYPTO_SUFFIX = '-USD'
    
    def __init__(self, config: dict = None):
        if config is None:
            raise ValueError("Config dictionary is required")
        
        self.keys = config.get("keys")
        self.category = config.get("category")
        self.candle_length = config.get("candle_length")
        self.period = config.get("period")  # Can be None for auto-max
        
        # Data source: "yfinance" (default), "binance" (crypto only, longer history)
        self.data_source = config.get("data_source", "yfinance").lower()
        
        # Extended download mode - bypass API limits by chunking requests
        self.extended_download = config.get("extended_download", False)
        self.chunk_overlap_days = config.get("chunk_overlap_days", 1)  # Overlap for deduplication
        self.request_delay_seconds = config.get("request_delay_seconds", 0.5)  # Delay between requests
        
        if not self.keys or not isinstance(self.keys, list):
            raise ValueError("Keys must be a non-empty list")
        if self.category not in ['crypto', 'stock']:
            raise ValueError("Category must be 'crypto' or 'stock'")
        if not self.candle_length:
            raise ValueError("Candle length is required")
        
        # Validate data source
        if self.data_source == "binance" and self.category != "crypto":
            logger.warning("Binance only supports crypto. Falling back to yfinance for stocks.")
            self.data_source = "yfinance"
        
        if self.data_source not in ["yfinance", "binance"]:
            raise ValueError(f"Invalid data_source '{self.data_source}'. Use 'yfinance' or 'binance'")
        
        self.candle_length = self._normalize_interval(self.candle_length)
        
        # Validate interval for Binance
        if self.data_source == "binance" and self.candle_length not in self.BINANCE_INTERVALS:
            raise ValueError(f"Interval '{self.candle_length}' not supported by Binance. "
                           f"Supported: {list(self.BINANCE_INTERVALS.keys())}")
        
        # Handle period: null -> auto-max for the interval
        if self.data_source == "binance":
            max_days = self.BINANCE_INTERVAL_LIMITS.get(self.candle_length, 36500)
        else:
            max_days = self.INTERVAL_LIMITS.get(self.candle_length, 36500)
            
        if self.period is None:
            # Auto-max: use maximum allowed period for this interval
            # If extended_download is enabled, use a reasonable default (1 year)
            if self.extended_download:
                self.period = 1.0  # Default to 1 year for extended download
                logger.info(f"Period=null with extended_download → default: 1 year")
            else:
                self.period = max_days / 365.0
                logger.info(f"Period=null → auto-max: {max_days} days ({self.period:.2f} years) for {self.candle_length}")
        
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        
        if not self.start_date or not self.end_date:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=int(self.period * 365))
            self.start_date = start_dt.strftime("%Y-%m-%d")
            self.end_date = end_dt.strftime("%Y-%m-%d")
        
        # Only validate/adjust period if NOT using extended download AND using yfinance
        if not self.extended_download and self.data_source == "yfinance":
            self._validate_interval_period()
        else:
            logger.info(f"Extended download mode: will fetch {int(self.period * 365)} days in chunks "
                       f"(source: {self.data_source})")
        
        self.data = {}
        
        logger.info(f"DataLoader: {self.category}, {len(self.keys)} assets, "
                   f"{self.candle_length}, {self.start_date} to {self.end_date}"
                   f" [source: {self.data_source.upper()}]"
                   f"{' (EXTENDED)' if self.extended_download else ''}")
    
    def _normalize_interval(self, interval: str) -> str:
        interval = interval.lower().strip()
        if interval in self.INTERVAL_ALIASES:
            interval = self.INTERVAL_ALIASES[interval]
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid: {self.VALID_INTERVALS}")
        return interval
    
    def _validate_interval_period(self) -> None:
        """Validate and adjust period to fit interval limits. Keeps interval, adjusts period."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        requested_days = (end - start).days
        max_days = self.INTERVAL_LIMITS.get(self.candle_length, 36500)
        
        if requested_days > max_days:
            # Adjust period to fit interval limit (keep interval, reduce period)
            logger.warning(
                f"Requested period ({requested_days}d) exceeds '{self.candle_length}' limit ({max_days}d). "
                f"Reducing period to {max_days} days to preserve {self.candle_length} interval."
            )
            
            # Adjust start_date to fit within max_days
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=max_days)
            self.start_date = start_dt.strftime("%Y-%m-%d")
            self.period = max_days / 365.0
            
            logger.info(f"✓ Period auto-adjusted to: {max_days} days ({self.period:.2f} years), interval preserved: {self.candle_length}")
    
    def _format_ticker(self, key: str) -> str:
        if self.category == 'crypto':
            if not key.endswith('-USD') and not key.endswith('-USDT'):
                return f"{key.upper()}{self.CRYPTO_SUFFIX}"
            return key.upper()
        else:
            return key.split(':')[0].upper() if ':' in key else key.upper()
    
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            formatted_ticker = self._format_ticker(ticker)
            logger.info(f"Downloading {formatted_ticker} ({self.candle_length})...")
            
            data = yf.download(
                formatted_ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.candle_length,
                progress=False,
                timeout=60  # Increased timeout for large downloads
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty:
                logger.warning(f"No data for {formatted_ticker}")
                return None
            
            data['Ticker'] = ticker
            data.reset_index(inplace=True)
            
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
            elif 'Datetime' not in data.columns and data.index.name == 'Datetime':
                data.reset_index(inplace=True)
            
            logger.info(f"Downloaded {len(data)} records for {formatted_ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return None
    
    def _format_binance_symbol(self, key: str) -> str:
        """Convert ticker to Binance symbol format (e.g., BTC -> BTCUSDT)."""
        key = key.upper().replace('-USD', '').replace('-USDT', '')
        
        # Check if we have a predefined mapping
        if key in self.BINANCE_SYMBOLS:
            return self.BINANCE_SYMBOLS[key]
        
        # Default: append USDT
        return f"{key}USDT"
    
    def _download_binance_klines(self, symbol: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
        """
        Download klines (candlestick data) from Binance public API.
        
        Args:
            symbol: Binance trading pair (e.g., 'BTCUSDT')
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds
            
        Returns:
            DataFrame with OHLCV data
        """
        interval = self.BINANCE_INTERVALS.get(self.candle_length, '5m')
        url = "https://api.binance.com/api/v3/klines"
        
        all_data = []
        current_start = start_ms
        request_count = 0
        max_requests = 500  # Safety limit to prevent infinite loops
        
        # Estimate total requests needed for progress reporting
        interval_ms = self._get_interval_ms(interval)
        total_ms = end_ms - start_ms
        estimated_candles = total_ms // interval_ms if interval_ms > 0 else 0
        estimated_requests = max(1, (estimated_candles // 1000) + 1)
        
        print(f"  📥 Fetching ~{estimated_candles:,} candles in ~{estimated_requests} requests...", flush=True)
        logger.info(f"  Fetching ~{estimated_candles:,} candles in ~{estimated_requests} requests...")
        
        while current_start < end_ms and request_count < max_requests:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000  # Max per request
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                request_count += 1
                
                # Progress logging every 5 requests (more frequent for visibility)
                if request_count % 5 == 0 or request_count == 1:
                    progress_pct = min(100, (len(all_data) / max(1, estimated_candles)) * 100)
                    msg = f"  📊 Progress: {len(all_data):,} candles ({progress_pct:.0f}%) - request {request_count}/{estimated_requests}"
                    print(msg, flush=True)
                    logger.info(msg)
                
                # Move to next batch (last candle timestamp + 1ms)
                last_timestamp = data[-1][0]
                if last_timestamp <= current_start:
                    # Prevent infinite loop if timestamps don't advance
                    logger.warning(f"  Timestamp not advancing, breaking loop")
                    break
                current_start = last_timestamp + 1
                
                # Minimal rate limiting (Binance allows 1200 requests/min)
                if request_count % 5 == 0:
                    time.sleep(0.1)  # Brief pause every 5 requests
                
            except requests.exceptions.Timeout:
                logger.warning(f"  Request timeout, retrying...")
                time.sleep(1)
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Binance API error: {e}")
                if request_count > 0:
                    logger.info(f"  Partial data retrieved: {len(all_data)} candles")
                break
        
        if request_count >= max_requests:
            logger.warning(f"  Reached max requests limit ({max_requests}), data may be incomplete")
        
        if not all_data:
            return None
        
        msg = f"  ✅ Download complete: {len(all_data):,} candles in {request_count} requests"
        print(msg, flush=True)
        logger.info(msg)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
        
        # Keep only OHLCV columns and convert to float
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def _get_interval_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        return interval_map.get(interval, 5 * 60 * 1000)  # Default to 5m
    
    def download_binance(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download cryptocurrency data from Binance.
        
        Binance provides years of historical data for all intervals,
        unlike Yahoo Finance which limits intraday data to 60 days.
        
        Args:
            ticker: Crypto ticker (e.g., 'BTC', 'ETH')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            symbol = self._format_binance_symbol(ticker)
            msg = f"🔄 Downloading {symbol} from Binance ({self.candle_length}, {self.start_date} to {self.end_date})..."
            print(msg, flush=True)
            logger.info(msg)
            
            # Convert dates to milliseconds
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            
            df = self._download_binance_klines(symbol, start_ms, end_ms)
            
            if df is None or df.empty:
                logger.warning(f"No Binance data for {symbol}")
                return None
            
            df['Ticker'] = ticker
            
            msg = f"✅ Downloaded {len(df):,} records for {symbol} from Binance"
            print(msg, flush=True)
            logger.info(msg)
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {ticker} from Binance: {e}")
            return None
    
    def _calculate_chunk_ranges(self) -> List[Tuple[str, str]]:
        """
        Calculate date ranges for chunked downloading.
        
        Breaks the total requested period into API-compliant chunks
        with overlap for deduplication.
        
        Returns:
            List of (start_date, end_date) tuples for each chunk
        """
        max_days = self.INTERVAL_LIMITS.get(self.candle_length, 36500)
        chunk_size = max_days - self.chunk_overlap_days  # Days per chunk minus overlap
        
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        
        if total_days <= max_days:
            # Single request is sufficient
            return [(self.start_date, self.end_date)]
        
        chunks = []
        current_end = end_dt
        
        while current_end > start_dt:
            chunk_start = current_end - timedelta(days=max_days)
            if chunk_start < start_dt:
                chunk_start = start_dt
            
            chunks.append((
                chunk_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d")
            ))
            
            # Move to next chunk (with overlap)
            current_end = chunk_start + timedelta(days=self.chunk_overlap_days)
            
            if chunk_start <= start_dt:
                break
        
        # Reverse to go chronologically (oldest first)
        chunks.reverse()
        logger.info(f"Extended download: {len(chunks)} chunks needed for {total_days} days")
        return chunks
    
    def download_extended(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download data in chunks to bypass API limits.
        
        Makes multiple requests with delays, then concatenates and deduplicates
        the results to create a seamless DataFrame spanning the full period.
        
        Args:
            ticker: Asset ticker symbol
            
        Returns:
            Combined DataFrame with deduplicated data spanning full period
        """
        try:
            formatted_ticker = self._format_ticker(ticker)
            chunks = self._calculate_chunk_ranges()
            
            all_chunks = []
            
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                logger.info(f"Chunk {i+1}/{len(chunks)}: {formatted_ticker} "
                           f"({chunk_start} to {chunk_end})")
                
                data = yf.download(
                    formatted_ticker,
                    start=chunk_start,
                    end=chunk_end,
                    interval=self.candle_length,
                    progress=False,
                    timeout=60  # Increased timeout for large downloads
                )
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if not data.empty:
                    data.reset_index(inplace=True)
                    all_chunks.append(data)
                    logger.info(f"  → {len(data)} records")
                else:
                    logger.warning(f"  → Empty chunk")
                
                # Delay between requests (except for last chunk)
                if i < len(chunks) - 1:
                    time.sleep(self.request_delay_seconds)
            
            if not all_chunks:
                logger.warning(f"No data retrieved for {formatted_ticker}")
                return None
            
            # Concatenate all chunks
            combined = pd.concat(all_chunks, ignore_index=True)
            
            # Normalize datetime column name
            if 'Date' in combined.columns:
                combined.rename(columns={'Date': 'Datetime'}, inplace=True)
            
            # Remove duplicates (from overlapping chunks)
            combined = combined.drop_duplicates(subset=['Datetime'], keep='last')
            combined = combined.sort_values('Datetime').reset_index(drop=True)
            
            combined['Ticker'] = ticker
            
            logger.info(f"Extended download complete: {len(combined)} total records "
                       f"for {formatted_ticker} ({chunks[0][0]} to {chunks[-1][1]})")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in extended download for {ticker}: {e}")
            return None
    
    def assemble_data(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"Assembling data for {len(self.keys)} assets from {self.data_source.upper()}...")
        for key in self.keys:
            # Choose download method based on data source
            if self.data_source == "binance":
                # Binance handles large date ranges internally
                df = self.download_binance(key)
            elif self.extended_download:
                # yfinance extended download (chunked)
                df = self.download_extended(key)
            else:
                # Standard yfinance download
                df = self.download_data(key)
            
            if df is not None:
                self.data[key] = df
            time.sleep(0.5)
        logger.info(f"Loaded {len(self.data)}/{len(self.keys)} assets")
        return self.data
    
    def return_data(self) -> Dict[str, pd.DataFrame]:
        if not self.data:
            logger.warning("No data. Run assemble_data() first.")
        return self.data
    
    def get_combined_data(self) -> pd.DataFrame:
        if not self.data:
            logger.warning("No data. Run assemble_data() first.")
            return pd.DataFrame()
        return pd.concat(self.data.values(), ignore_index=True)
    
    def get_summary(self) -> pd.DataFrame:
        if not self.data:
            return pd.DataFrame()
        summary = []
        for ticker, df in self.data.items():
            summary.append({
                'Ticker': ticker,
                'Records': len(df),
                'Start': df['Datetime'].min(),
                'End': df['Datetime'].max(),
                'Columns': ', '.join(df.columns)
            })
        return pd.DataFrame(summary)

    def safe_data_to_files(self, folder_path: str) -> None:
        self_based_project_directory = os.path.dirname(os.path.abspath(__file__))
        full_folder_path = os.path.join(self_based_project_directory, folder_path)
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)
        
        for ticker, df in self.data.items():
            file_path = os.path.join(full_folder_path, f"{ticker.replace(':', '_')}.csv")
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {ticker} data to {file_path}")
    
    @staticmethod
    def get_example_data(keys: list = None, period: int = 2) -> pd.DataFrame:
        """
        Quick helper to load example data for testing/examples.
        
        Args:
            keys: List of tickers (default: ["AAPL", "GOOGL:NASDAQ"])
            period: Years of data (default: 2)
        
        Returns:
            Combined DataFrame ready for feature engineering
        """
        if keys is None:
            keys = ["AAPL", "GOOGL:NASDAQ"]
        
        config = {
            "keys": keys,
            "category": "stock",
            "candle_length": "1d",
            "period": period,
        }
        
        loader = DataLoader(config)
        loader.assemble_data()
        return loader.get_combined_data()