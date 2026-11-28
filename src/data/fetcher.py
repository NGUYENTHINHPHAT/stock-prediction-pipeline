"""
Daily OHLCV data from Yahoo Finance
"""
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import yfinance as yf
from src.utils.logger import log
from src.utils.config import settings


class DataFetcher:
    """Fetch daily OHLCV bars for a list of symbols"""

    def fetch_yahoo_finance(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=settings.lookback_days)
        if end_date is None:
            end_date = datetime.now()
        
        log.info(f"Fetching data from Yahoo Finance for {symbols}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d")
                )
                
                if df.empty:
                    log.warning(f"No data found for {symbol}")
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                df['symbol'] = symbol
                
                # Rename columns to lowercase
                df.columns = df.columns.str.lower()
                
                # Select relevant columns
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                all_data.append(df)
                log.info(f"Fetched {len(df)} rows for {symbol}")
                
            except Exception as e:
                log.error(f"Error fetching {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched for any symbol")
        
        # Combine all symbols
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        log.info(f"Total rows fetched: {len(combined_df)}")
        return combined_df
    
    def fetch(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch daily bars (defaults: configured symbols, lookback_days ago -> today)
        
        Args:
            symbols: List of stock symbols (uses config if not provided)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbols is None:
            symbols = settings.symbol_list
        
        return self.fetch_yahoo_finance(symbols, start_date, end_date)
