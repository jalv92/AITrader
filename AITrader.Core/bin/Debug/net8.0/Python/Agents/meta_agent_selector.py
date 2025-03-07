"""
Meta-Agent Selector Module

This module implements the Meta-Agent Selector component responsible for analyzing
market regimes and dynamically selecting the most appropriate Specialized Trading
Agent (Bullish, Bearish, or Range).
"""

import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from enum import Enum


class MarketRegime(Enum):
    """Enum representing different market regimes."""
    BULLISH = 0
    BEARISH = 1
    RANGE = 2
    UNKNOWN = 3


class MetaAgentSelector:
    """
    Meta-Agent Selector that analyzes market conditions and selects the most
    appropriate specialized trading agent based on the identified market regime.
    
    This agent serves as the second decision layer in the hierarchical agent
    architecture, after the No-Trade Filter Agent has determined that trading
    is viable.
    """
    
    def __init__(self, observation_space, action_space, model_path=None, learning_rate=0.0003):
        """
        Initialize the Meta-Agent Selector.
        
        Parameters:
        -----------
        observation_space : gym.spaces
            The observation space defining market features used by the agent
        action_space : gym.spaces
            The action space (typically discrete with 3 options: Bullish, Bearish, Range)
        model_path : str, optional
            Path to a pre-trained model file
        learning_rate : float, optional
            Learning rate for the agent's policy network
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Initialize the PPO model for the Meta-Agent Selector
        self.model = PPO("MlpPolicy", 
                         observation_space=self.observation_space,
                         action_space=self.action_space,
                         learning_rate=self.learning_rate,
                         verbose=1)
        
        # Load pre-trained model if provided
        if model_path:
            self.load(model_path)
        
        # Dictionary mapping regime indices to their respective agents
        # To be populated by the calling code
        self.specialized_agents = {}
        
        # Default confidence thresholds for regime classification
        self.regime_confidence_threshold = 0.6
    
    def register_specialized_agent(self, regime, agent):
        """
        Register a specialized agent for a specific market regime.
        
        Parameters:
        -----------
        regime : MarketRegime
            The market regime for which the agent is specialized
        agent : object
            The specialized agent instance
        """
        self.specialized_agents[regime] = agent
    
    def train(self, env, total_timesteps=10000, callback=None):
        """
        Train the Meta-Agent Selector.
        
        Parameters:
        -----------
        env : gym.Env
            The training environment
        total_timesteps : int
            Total number of training timesteps
        callback : BaseCallback, optional
            Callback for monitoring training
            
        Returns:
        --------
        self : MetaAgentSelector
            The trained agent
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict the market regime based on the current market observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
        state : np.ndarray, optional
            State for recurrent policies
        deterministic : bool
            Whether to use deterministic actions
            
        Returns:
        --------
        MarketRegime
            The identified market regime
        dict
            Confidence levels for each regime
        """
        action, _ = self.model.predict(observation, state, deterministic)
        confidences = self.get_regime_confidences(observation)
        
        # Convert action index to MarketRegime enum
        regime = MarketRegime(action)
        
        return regime, confidences
    
    def get_regime_confidences(self, observation):
        """
        Calculate the confidence levels for each market regime.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        dict
            Dictionary mapping MarketRegime to confidence levels
        """
        action_probs = self.model.policy.get_distribution(observation).distribution.probs.cpu().numpy()[0]
        
        return {
            MarketRegime.BULLISH: float(action_probs[MarketRegime.BULLISH.value]),
            MarketRegime.BEARISH: float(action_probs[MarketRegime.BEARISH.value]),
            MarketRegime.RANGE: float(action_probs[MarketRegime.RANGE.value])
        }
    
    def select_agent(self, observation, confidence_threshold=None):
        """
        Select the appropriate specialized agent based on the identified market regime.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
        confidence_threshold : float, optional
            Override the default confidence threshold
            
        Returns:
        --------
        object
            The selected specialized agent
        MarketRegime
            The identified market regime
        float
            Confidence level of the regime classification
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.regime_confidence_threshold
        
        regime, confidences = self.predict(observation)
        confidence = confidences[regime]
        
        # If confidence is below threshold, return Unknown regime
        if confidence < threshold:
            return None, MarketRegime.UNKNOWN, confidence
        
        # Return the specialized agent for the identified regime
        if regime in self.specialized_agents:
            return self.specialized_agents[regime], regime, confidence
        else:
            # Handle case where no agent is registered for the regime
            return None, regime, confidence
    
    def analyze_market_regime(self, historical_data, window_size=20):
        """
        Analyze historical data to identify market regime using traditional indicators.
        This method serves as a backup or complement to the ML-based regime classification.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data containing at least 'close' prices
        window_size : int
            Size of the rolling window for calculations
            
        Returns:
        --------
        MarketRegime
            The identified market regime
        dict
            Additional metrics about the market analysis
        """
        # Calculate some basic indicators for regime classification
        df = historical_data.copy()
        
        # Simple Moving Average
        df['sma'] = df['close'].rolling(window=window_size).mean()
        
        # Bollinger Bands
        df['std'] = df['close'].rolling(window=window_size).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)
        
        # Average True Range (volatility)
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=window_size).mean()
        
        # Last window_size rows for analysis
        recent_data = df.iloc[-window_size:].dropna()
        
        if len(recent_data) < window_size * 0.8:  # Check if we have enough data
            return MarketRegime.UNKNOWN, {}
        
        # Calculate trend
        price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        price_change_pct = price_change / recent_data['close'].iloc[0]
        
        # Calculate volatility
        volatility = recent_data['atr'].mean() / recent_data['close'].mean()
        
        # Calculate range behavior
        in_range = ((recent_data['close'] >= recent_data['lower_band']) & 
                    (recent_data['close'] <= recent_data['upper_band'])).mean()
        
        # Determine regime
        strong_uptrend = price_change_pct > 0.02  # 2% increase
        strong_downtrend = price_change_pct < -0.02  # 2% decrease
        high_volatility = volatility > 0.015  # 1.5% average volatility
        range_bound = in_range > 0.8  # 80% of time within bands
        
        metrics = {
            'price_change_pct': price_change_pct,
            'volatility': volatility,
            'in_range': in_range
        }
        
        # Regime classification logic
        if strong_uptrend and not range_bound:
            return MarketRegime.BULLISH, metrics
        elif strong_downtrend and not range_bound:
            return MarketRegime.BEARISH, metrics
        elif range_bound and not (strong_uptrend or strong_downtrend):
            return MarketRegime.RANGE, metrics
        else:
            return MarketRegime.UNKNOWN, metrics
    
    def save(self, path):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        path : str
            Path where the model will be saved
        """
        self.model.save(path)
    
    def load(self, path):
        """
        Load a pre-trained model from a file.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
            
        Returns:
        --------
        self : MetaAgentSelector
            The agent with loaded model
        """
        self.model = PPO.load(path, 
                              observation_space=self.observation_space,
                              action_space=self.action_space)
        return self


class MetaAgentSelectorCallback(BaseCallback):
    """
    Callback for monitoring and logging the Meta-Agent Selector's training process.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        verbose : int
            Verbosity level
        """
        super(MetaAgentSelectorCallback, self).__init__(verbose)
        self.training_metrics = {
            'bullish_selections': [],
            'bearish_selections': [],
            'range_selections': [],
            'avg_confidence': []
        }
    
    def _on_step(self):
        """
        Method called at each step of training.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        # Extract information from the most recent update
        if hasattr(self.model, 'actions') and len(self.model.actions) > 0:
            bullish_selections = (self.model.actions == MarketRegime.BULLISH.value).sum()
            bearish_selections = (self.model.actions == MarketRegime.BEARISH.value).sum()
            range_selections = (self.model.actions == MarketRegime.RANGE.value).sum()
            
            self.training_metrics['bullish_selections'].append(bullish_selections)
            self.training_metrics['bearish_selections'].append(bearish_selections)
            self.training_metrics['range_selections'].append(range_selections)
            
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls}: Bullish: {bullish_selections}, Bearish: {bearish_selections}, "
                      f"Range: {range_selections}")
        
        return True
