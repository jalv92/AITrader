import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class MarketDataLoader:
    """
    Loads and preprocesses market data from CSV files for training RL agents.
    Specifically designed to handle 1-minute candlestick data for MNQ/NQ futures.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the market data CSV files
        """
        # Default to the data directory in the project structure
        if data_dir is None:
            # Get the current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate to the data directory
            self.data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data"))
        else:
            self.data_dir = data_dir
            
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets in the data directory.
        
        Returns:
            List of dataset filenames
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return csv_files
    
    def load_dataset(self, dataset_name: str, force_reload: bool = False) -> pd.DataFrame:
        """
        Load a specific dataset by name.
        
        Args:
            dataset_name: Name of the dataset file
            force_reload: Whether to force reload from disk even if cached
            
        Returns:
            Pandas DataFrame containing the market data
        """
        # Check if the dataset is already cached and we don't need to force reload
        if dataset_name in self.data_cache and not force_reload:
            return self.data_cache[dataset_name]
        
        # Construct the full path to the dataset
        dataset_path = os.path.join(self.data_dir, dataset_name)
        
        # Check if the file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Load the dataset
        try:
            data = pd.read_csv(dataset_path)
            
            # Convert timestamp to datetime
            if 'Timestamp' in data.columns:
                data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            
            # Cache the loaded data
            self.data_cache[dataset_name] = data
            return data
        except Exception as e:
            raise IOError(f"Error loading dataset {dataset_name}: {str(e)}")
    
    def preprocess_for_training(self, data: pd.DataFrame, features: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess the data for training RL agents.
        
        Args:
            data: Raw market data as a DataFrame
            features: List of feature columns to include (default: OHLC)
            
        Returns:
            Tuple of (preprocessed numpy array, feature names)
        """
        if features is None:
            # Default to OHLC data
            features = ['Open', 'High', 'Low', 'Close']
        
        # Ensure all required columns exist
        missing_cols = [col for col in features if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Create a copy to avoid modifying the original
        df = data[features].copy()
        
        # Handle missing values
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill for any remaining NaNs
        
        # Normalize the data for better RL training
        # Min-max scaling for OHLC data
        for col in df.columns:
            if col in ['Open', 'High', 'Low', 'Close']:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:  # Prevent division by zero
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df.values, df.columns.tolist()
    
    def split_train_test(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.
        
        Args:
            data: DataFrame to split
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (training data, testing data)
        """
        # Time-series split (not random)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        return train_data, test_data
    
    def extract_market_regimes(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Extract market regimes (bullish, bearish, range-bound) based on price trends.
        
        Args:
            data: Market data DataFrame with OHLC
            window: Window size for trend detection
            
        Returns:
            DataFrame with an additional 'regime' column
        """
        df = data.copy()
        
        # Ensure we have Close prices
        if 'Close' not in df.columns:
            raise ValueError("Close prices are required for regime detection")
        
        # Calculate simple trend indicators
        df['sma'] = df['Close'].rolling(window=window).mean()
        df['std'] = df['Close'].rolling(window=window).std()
        
        # Calculate price slope
        df['slope'] = (df['Close'] - df['Close'].shift(window)) / window
        
        # Determine market regime
        df['regime'] = 'unknown'
        df.loc[df['slope'] > 0.00005, 'regime'] = 'bullish'
        df.loc[df['slope'] < -0.00005, 'regime'] = 'bearish'
        df.loc[(df['slope'] >= -0.00005) & (df['slope'] <= 0.00005), 'regime'] = 'range'
        
        # Forward fill any unknown regimes
        df['regime'] = df['regime'].replace('unknown', np.nan).fillna(method='ffill').fillna('range')
        
        # Drop intermediate columns
        df.drop(['sma', 'std', 'slope'], axis=1, inplace=True)
        
        return df
    
    def load_and_prepare_for_agents(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """
        Load, preprocess, and prepare data for the hierarchical RL agents.
        
        Args:
            dataset_name: Name of the dataset file
            
        Returns:
            Dictionary of prepared data for each agent type
        """
        # Load the raw data
        data = self.load_dataset(dataset_name)
        
        # Extract market regimes
        data_with_regimes = self.extract_market_regimes(data)
        
        # Split data by regime
        bullish_data = data_with_regimes[data_with_regimes['regime'] == 'bullish']
        bearish_data = data_with_regimes[data_with_regimes['regime'] == 'bearish']
        range_data = data_with_regimes[data_with_regimes['regime'] == 'range']
        
        # Preprocess each dataset
        bullish_features, _ = self.preprocess_for_training(bullish_data)
        bearish_features, _ = self.preprocess_for_training(bearish_data)
        range_features, _ = self.preprocess_for_training(range_data)
        
        # Prepare all data for meta-agent
        all_features, _ = self.preprocess_for_training(data_with_regimes)
        
        return {
            'all': all_features,
            'bullish': bullish_features,
            'bearish': bearish_features,
            'range': range_features,
            'raw_data': data,
            'data_with_regimes': data_with_regimes
        }

# Example usage
if __name__ == "__main__":
    loader = MarketDataLoader()
    
    try:
        # List available datasets
        datasets = loader.list_available_datasets()
        print(f"Available datasets: {datasets}")
        
        if datasets:
            # Load the first dataset
            dataset = datasets[0]
            print(f"Loading dataset: {dataset}")
            
            data = loader.load_dataset(dataset)
            print(f"Data shape: {data.shape}")
            print(f"Data columns: {data.columns.tolist()}")
            print(f"First few rows:\n{data.head()}")
            
            # Prepare data for agents
            prepared_data = loader.load_and_prepare_for_agents(dataset)
            
            # Print summary of prepared data
            for regime, features in prepared_data.items():
                if regime not in ['raw_data', 'data_with_regimes']:
                    print(f"{regime} data shape: {features.shape}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
