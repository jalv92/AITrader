"""
Data Preparation Module

This module handles the preprocessing and preparation of market data for the AITrader system,
creating specialized feature sets optimized for each agent type in the hierarchical architecture.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
import talib
from gym import spaces
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureSet(Enum):
    """Enum representing different feature sets for specialized agents."""
    COMMON = 0       # Basic features used by all agents
    NO_TRADE = 1     # Features for the No-Trade Filter Agent
    META = 2         # Features for the Meta-Agent Selector
    BULLISH = 3      # Features optimized for Bullish Agent
    BEARISH = 4      # Features optimized for Bearish Agent
    RANGE = 5        # Features optimized for Range Agent


class DataPreparationPipeline:
    """
    Pipeline for preprocessing and preparing market data for the AITrader system.
    This class handles data cleaning, feature engineering, and normalization.
    """
    
    def __init__(self, use_talib=True, window_sizes=(5, 10, 20, 50, 100), log_transform_volume=True):
        """
        Initialize the DataPreparationPipeline.
        
        Parameters:
        -----------
        use_talib : bool
            Whether to use TA-Lib for technical indicators calculation
        window_sizes : tuple
            Tuple of window sizes for calculating moving averages and other indicators
        log_transform_volume : bool
            Whether to apply log transform to volume data
        """
        self.use_talib = use_talib
        self.window_sizes = window_sizes
        self.log_transform_volume = log_transform_volume
        
        # Feature lists for different agent types
        self.feature_sets = {
            FeatureSet.COMMON: [],
            FeatureSet.NO_TRADE: [],
            FeatureSet.META: [],
            FeatureSet.BULLISH: [],
            FeatureSet.BEARISH: [],
            FeatureSet.RANGE: []
        }
        
        # Normalization parameters
        self.normalization_stats = {}
    
    def prepare_data(self, data: pd.DataFrame, feature_set: FeatureSet = FeatureSet.COMMON, 
                    add_common_features: bool = True) -> pd.DataFrame:
        """
        Prepare market data with appropriate features for the specified agent type.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw market data containing OHLCV columns
        feature_set : FeatureSet
            Type of feature set to generate (specialized for different agent types)
        add_common_features : bool
            Whether to include common features with specialized features
            
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with engineered features
        """
        # Validate input data
        self._validate_input_data(data)
        
        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Apply initial cleaning and preprocessing
        df = self._clean_data(df)
        
        # Calculate common features used by all agents
        if add_common_features or feature_set == FeatureSet.COMMON:
            df = self._add_common_features(df)
        
        # Add specialized features based on the requested feature set
        if feature_set == FeatureSet.NO_TRADE:
            df = self._add_no_trade_filter_features(df)
        elif feature_set == FeatureSet.META:
            df = self._add_meta_agent_features(df)
        elif feature_set == FeatureSet.BULLISH:
            df = self._add_bullish_agent_features(df)
        elif feature_set == FeatureSet.BEARISH:
            df = self._add_bearish_agent_features(df)
        elif feature_set == FeatureSet.RANGE:
            df = self._add_range_agent_features(df)
        
        # Drop NaN values created by indicator calculations
        df = df.dropna()
        
        # Update feature list for the requested feature set
        self.feature_sets[feature_set] = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date']]
        
        return df
    
    def create_observation_spaces(self) -> Dict[FeatureSet, spaces.Space]:
        """
        Create Gym observation spaces for each feature set.
        
        Returns:
        --------
        Dict[FeatureSet, spaces.Space]
            Dictionary mapping feature sets to their corresponding observation spaces
        """
        observation_spaces = {}
        
        for feature_set, features in self.feature_sets.items():
            if not features:
                continue
                
            # Create a Box space with appropriate dimensions
            n_features = len(features)
            
            # Using standard bounds for normalized features
            observation_spaces[feature_set] = spaces.Box(
                low=-10.0, high=10.0, shape=(n_features,), dtype=np.float32
            )
        
        return observation_spaces
    
    def normalize_data(self, data: pd.DataFrame, feature_set: FeatureSet, 
                      fit: bool = False) -> pd.DataFrame:
        """
        Normalize data for model input based on the specified feature set.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Processed market data with engineered features
        feature_set : FeatureSet
            The feature set type to normalize
        fit : bool
            Whether to fit normalization parameters (for training data)
            or use existing parameters (for test data)
            
        Returns:
        --------
        pd.DataFrame
            Normalized DataFrame
        """
        if feature_set not in self.feature_sets or not self.feature_sets[feature_set]:
            raise ValueError(f"No features defined for feature set {feature_set.name}")
        
        features = self.feature_sets[feature_set]
        df = data.copy()
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Initialize normalization stats for this feature set if fitting
        if fit and feature_set not in self.normalization_stats:
            self.normalization_stats[feature_set] = {}
        
        # Normalize each feature
        for feature in features:
            if fit:
                # Calculate and store normalization parameters
                mean = df[feature].mean()
                std = df[feature].std()
                if std == 0:
                    std = 1.0  # Prevent division by zero
                
                self.normalization_stats[feature_set][feature] = {'mean': mean, 'std': std}
            else:
                # Use stored parameters
                if feature_set not in self.normalization_stats or feature not in self.normalization_stats[feature_set]:
                    raise ValueError(f"Normalization parameters not available for feature {feature} in {feature_set.name}")
                
                mean = self.normalization_stats[feature_set][feature]['mean']
                std = self.normalization_stats[feature_set][feature]['std']
            
            # Apply z-score normalization
            df[feature] = (df[feature] - mean) / std
        
        return df[features]
    
    def create_observation_vector(self, data_row: pd.Series, feature_set: FeatureSet) -> np.ndarray:
        """
        Create an observation vector from a single data row for agent input.
        
        Parameters:
        -----------
        data_row : pd.Series
            A row from the processed market data
        feature_set : FeatureSet
            The feature set type to use
            
        Returns:
        --------
        np.ndarray
            Observation vector for agent input
        """
        if feature_set not in self.feature_sets or not self.feature_sets[feature_set]:
            raise ValueError(f"No features defined for feature set {feature_set.name}")
        
        features = self.feature_sets[feature_set]
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in data_row.index]
        if missing_features:
            raise ValueError(f"Missing features in data row: {missing_features}")
        
        # Extract feature values
        observation = np.array([data_row[feature] for feature in features], dtype=np.float32)
        
        return observation
    
    def prepare_training_data(self, data: pd.DataFrame, train_test_split: float = 0.8, 
                             shuffle: bool = False) -> Tuple[Dict[FeatureSet, pd.DataFrame], 
                                                          Dict[FeatureSet, pd.DataFrame]]:
        """
        Prepare training and testing data for all agent types.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw market data containing OHLCV columns
        train_test_split : float
            Proportion of data to use for training (0 to 1)
        shuffle : bool
            Whether to shuffle data before splitting
            
        Returns:
        --------
        Tuple[Dict[FeatureSet, pd.DataFrame], Dict[FeatureSet, pd.DataFrame]]
            Dictionaries of training and testing DataFrames for each feature set
        """
        # Validate input data
        self._validate_input_data(data)
        
        # Split data into training and testing sets
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        
        split_idx = int(len(data) * train_test_split)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        train_sets = {}
        test_sets = {}
        
        # Prepare each feature set
        for feature_set in FeatureSet:
            # Prepare and normalize training data
            train_df = self.prepare_data(train_data, feature_set)
            train_sets[feature_set] = self.normalize_data(train_df, feature_set, fit=True)
            
            # Prepare and normalize testing data (using training normalization parameters)
            test_df = self.prepare_data(test_data, feature_set)
            test_sets[feature_set] = self.normalize_data(test_df, feature_set, fit=False)
        
        return train_sets, test_sets
    
    def save_normalization_params(self, path: str) -> None:
        """
        Save normalization parameters to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the parameters
        """
        # Convert enum keys to strings for serialization
        params_to_save = {}
        for feature_set, params in self.normalization_stats.items():
            params_to_save[feature_set.name] = params
        
        pd.to_pickle(params_to_save, path)
        logger.info(f"Saved normalization parameters to {path}")
    
    def load_normalization_params(self, path: str) -> None:
        """
        Load normalization parameters from a file.
        
        Parameters:
        -----------
        path : str
            Path to load the parameters from
        """
        params_loaded = pd.read_pickle(path)
        
        # Convert string keys back to enums
        self.normalization_stats = {}
        for feature_set_str, params in params_loaded.items():
            feature_set = FeatureSet[feature_set_str]
            self.normalization_stats[feature_set] = params
        
        logger.info(f"Loaded normalization parameters from {path}")
    
    def _add_no_trade_filter_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specific to the No-Trade Filter Agent.
        These focus on market risk and tradability conditions.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with common features
            
        Returns:
        --------
        pd.DataFrame
            Market data with No-Trade Filter specific features added
        """
        df = data.copy()
        
        # Market volatility features
        df['high_volatility'] = (df['atr'] / df['close']) > df['atr'].rolling(window=50).mean() * 1.5
        
        # Gaps and limit moves
        df['gap_up'] = (df['open'] > df['close'].shift(1) * 1.02).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1) * 0.98).astype(int)
        
        # Trend strength and stability
        if self.use_talib:
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # Simplified ADX calculation (not exact but directionally similar)
            pos_dm = df['high'].diff()
            neg_dm = df['low'].diff() * -1
            pos_dm = pos_dm.clip(lower=0)
            neg_dm = neg_dm.clip(lower=0)
            tr = pd.DataFrame({
                'tr1': df['high'] - df['low'],
                'tr2': abs(df['high'] - df['close'].shift(1)),
                'tr3': abs(df['low'] - df['close'].shift(1))
            }).max(axis=1)
            tr_smoothed = tr.rolling(window=14).mean()
            pos_di = 100 * (pos_dm.rolling(window=14).mean() / tr_smoothed)
            neg_di = 100 * (neg_dm.rolling(window=14).mean() / tr_smoothed)
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            df['adx'] = dx.rolling(window=14).mean()
        
        # Volume consistency
        df['volume_consistency'] = df['volume'].rolling(window=10).std() / df['volume'].rolling(window=10).mean()
        
        # Unusual price action
        df['price_velocity'] = df['close'].diff(5) / 5
        df['unusual_price_action'] = abs(df['price_velocity']) > df['price_velocity'].rolling(window=50).std() * 3
        
        # Risk indicators
        df['risk_indicator'] = ((df['high_volatility'].astype(int) + 
                               df['gap_up'] + 
                               df['gap_down'] + 
                               df['unusual_price_action'].astype(int)) >= 2).astype(int)
        
        # Market regime stability
        df['regime_change'] = ((df['close'] > df['ema_20']) != (df['close'].shift(1) > df['ema_20'].shift(1))).astype(int)
        df['regime_stability'] = 1 - df['regime_change'].rolling(window=10).mean()
        
        return df
    
    def _add_meta_agent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specific to the Meta-Agent Selector.
        These focus on market regime classification.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with common features
            
        Returns:
        --------
        pd.DataFrame
            Market data with Meta-Agent specific features added
        """
        df = data.copy()
        
        # Trend features
        df['price_above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
        df['price_above_sma_100'] = (df['close'] > df['sma_100']).astype(int) if 'sma_100' in df.columns else 0
        
        # Trend strength
        df['trend_strength'] = (df['adx'] > 25).astype(int) if 'adx' in df.columns else np.nan
        
        # Momentum features
        df['momentum_10d'] = df['close'] - df['close'].shift(10)
        df['momentum_sign'] = np.sign(df['momentum_10d'])
        
        # Range-bound indicators
        df['in_bollinger_band'] = ((df['close'] >= df['bb_lower']) & 
                                  (df['close'] <= df['bb_upper'])).astype(int)
        df['bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['range_indicator'] = (df['bandwidth'] < df['bandwidth'].rolling(window=50).mean()).astype(int)
        
        # Moving average slope
        df['sma_20_slope'] = (df['sma_20'] - df['sma_20'].shift(5)) / 5
        df['ema_20_slope'] = (df['ema_20'] - df['ema_20'].shift(5)) / 5
        
        # MACD and RSI state
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['rsi_state'] = pd.cut(df['rsi'], 
                               bins=[0, 30, 45, 55, 70, 100], 
                               labels=[0, 1, 2, 3, 4]).astype(int)
        
        # Composite regime indicators
        bullish_signals = (df['price_above_sma_20'].astype(int) + 
                          df['price_above_sma_50'].astype(int) + 
                          df['macd_above_signal'].astype(int) + 
                          (df['rsi_state'] >= 3).astype(int))
        
        bearish_signals = ((~df['price_above_sma_20']).astype(int) + 
                          (~df['price_above_sma_50']).astype(int) + 
                          (~df['macd_above_signal']).astype(int) + 
                          (df['rsi_state'] <= 1).astype(int))
        
        range_signals = (df['range_indicator'].astype(int) + 
                        (df['rsi_state'] == 2).astype(int) + 
                        (abs(df['sma_20_slope']) < 0.001).astype(int))
        
        df['bullish_score'] = bullish_signals / 4
        df['bearish_score'] = bearish_signals / 4
        df['range_score'] = range_signals / 3
        
        # Dominant regime feature
        df['dominant_regime'] = df[['bullish_score', 'bearish_score', 'range_score']].idxmax(axis=1)
        df['dominant_regime'] = df['dominant_regime'].map({
            'bullish_score': 0, 
            'bearish_score': 1, 
            'range_score': 2
        })
        
        return df
    
    def _add_bullish_agent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specific to the Bullish Agent.
        These focus on identifying opportunities in bullish markets.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with common features
            
        Returns:
        --------
        pd.DataFrame
            Market data with Bullish Agent specific features added
        """
        df = data.copy()
        
        # Pullback detection
        df['pullback'] = ((df['close'] < df['close'].shift(1)) & 
                         (df['close'] > df['sma_20']) & 
                         (df['close'].shift(1) > df['close'].shift(2))).astype(int)
        
        # Breakout detection
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['breakout'] = (df['close'] > df['resistance_level'].shift(1)).astype(int)
        
        # Volume confirmation
        df['volume_breakout'] = ((df['close'] > df['close'].shift(1)) & 
                               (df['volume'] > df['volume'].shift(1) * 1.5)).astype(int)
        
        # Support levels
        df['support_level'] = df['low'].rolling(window=20).min()
        df['close_to_support'] = (abs(df['close'] - df['support_level']) / df['close'] < 0.02).astype(int)
        
        # Momentum strength
        if self.use_talib:
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            df['strong_momentum'] = (df['cci'] > 100).astype(int)
        else:
            # Simplified CCI calculation
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=14).mean()
            mean_deviation = abs(typical_price - sma_tp).rolling(window=14).mean()
            df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
            df['strong_momentum'] = (df['cci'] > 100).astype(int)
        
        # Buy signal strength
        df['buy_signal_strength'] = (df['pullback'].astype(int) + 
                                   df['breakout'].astype(int) + 
                                   df['volume_breakout'].astype(int) + 
                                   df['close_to_support'].astype(int) + 
                                   df['strong_momentum'].astype(int)) / 5
        
        return df
    
    def _add_bearish_agent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specific to the Bearish Agent.
        These focus on identifying opportunities in bearish markets.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with common features
            
        Returns:
        --------
        pd.DataFrame
            Market data with Bearish Agent specific features added
        """
        df = data.copy()
        
        # Dead cat bounce detection
        df['bounce'] = ((df['close'] > df['close'].shift(1)) & 
                      (df['close'] < df['sma_20']) & 
                      (df['close'].shift(1) < df['close'].shift(2))).astype(int)
        
        # Breakdown detection
        df['support_level'] = df['low'].rolling(window=20).min()
        df['breakdown'] = (df['close'] < df['support_level'].shift(1)).astype(int)
        
        # Volume confirmation for downtrend
        df['volume_breakdown'] = ((df['close'] < df['close'].shift(1)) & 
                                (df['volume'] > df['volume'].shift(1) * 1.5)).astype(int)
        
        # Resistance levels
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['close_to_resistance'] = (abs(df['close'] - df['resistance_level']) / df['close'] < 0.02).astype(int)
        
        # Downward momentum strength
        df['weak_momentum'] = (df['cci'] < -100).astype(int) if 'cci' in df.columns else np.nan
        
        # Sell signal strength
        df['sell_signal_strength'] = (df['bounce'].astype(int) + 
                                    df['breakdown'].astype(int) + 
                                    df['volume_breakdown'].astype(int) + 
                                    df['close_to_resistance'].astype(int) + 
                                    df['weak_momentum'].astype(int)) / 5
        
        # Capitulation signals
        df['down_days_streak'] = df['close'] < df['close'].shift(1)
        df['down_days_streak'] = df['down_days_streak'].astype(int).groupby(
            (df['down_days_streak'] != df['down_days_streak'].shift(1)).cumsum()).cumsum()
        df['capitulation'] = ((df['down_days_streak'] >= 3) & 
                            (df['volume'] > df['volume_sma_5'] * 2)).astype(int)
        
        return df
    
    def _add_range_agent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specific to the Range Agent.
        These focus on identifying opportunities in range-bound markets.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with common features
            
        Returns:
        --------
        pd.DataFrame
            Market data with Range Agent specific features added
        """
        df = data.copy()
        
        # Range identification
        df['price_range'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
        df['range_width'] = df['price_range'] / df['close']
        df['range_bound'] = (df['range_width'] < df['range_width'].rolling(window=50).mean()).astype(int)
        
        # Distance from range boundaries
        df['upper_range'] = df['high'].rolling(window=20).max()
        df['lower_range'] = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - df['lower_range']) / (df['upper_range'] - df['lower_range'])
        
        # Overbought/oversold within range
        df['overbought'] = (df['price_position'] > 0.8).astype(int)
        df['oversold'] = (df['price_position'] < 0.2).astype(int)
        
        # Mean reversion signals
        df['distance_from_mean'] = abs(df['close'] - df['sma_20']) / df['sma_20']
        df['mean_reversion'] = (df['distance_from_mean'] > df['distance_from_mean'].rolling(window=50).mean()).astype(int)
        
        # Oscillator extremes
        df['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
        
        # Price rejection at boundaries
        df['upper_wick'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['lower_wick'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['rejection_upper'] = ((df['price_position'] > 0.7) & (df['upper_wick'] > 0.6)).astype(int)
        df['rejection_lower'] = ((df['price_position'] < 0.3) & (df['lower_wick'] > 0.6)).astype(int)
        
        # Range trading signals
        df['sell_signal_range'] = df['overbought'] & (df['rsi'] > 70)
        df['buy_signal_range'] = df['oversold'] & (df['rsi'] < 30)
        
        return df
    
    # Private methods for data validation and cleaning
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        Validate that input data contains required columns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw market data
            
        Raises:
        -------
        ValueError
            If required columns are missing
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {missing_columns}")
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw market data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw market data
            
        Returns:
        --------
        pd.DataFrame
            Cleaned market data
        """
        df = data.copy()
        
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex) and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex) and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Handle missing values
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows with zero or negative values for prices and volume
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) & (df['volume'] > 0)]
        
        # Apply log transform to volume if specified
        if self.log_transform_volume:
            df['volume'] = np.log1p(df['volume'])
        
        return df
    
    def _add_common_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common features used by all agents to the market data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Cleaned market data
            
        Returns:
        --------
        pd.DataFrame
            Market data with common features added
        """
        df = data.copy()
        
        # Price-based features
        
        # Returns
        df['return_1d'] = df['close'].pct_change(1)
        
        # Calculate returns for different periods
        for window in self.window_sizes:
            df[f'return_{window}d'] = df['close'].pct_change(window)
        
        # Simple Moving Averages
        for window in self.window_sizes:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Moving Average Convergence Divergence (MACD)
        if self.use_talib:
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            # Simplified MACD calculation without TA-Lib
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Relative Strength Index (RSI)
        if self.use_talib:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        else:
            # Simplified RSI calculation
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if self.use_talib:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        else:
            # Manual Bollinger Bands calculation
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Average True Range (ATR) for volatility
        if self.use_talib:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # Manual ATR calculation
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
        
        # Price volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # Volume features
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_5']