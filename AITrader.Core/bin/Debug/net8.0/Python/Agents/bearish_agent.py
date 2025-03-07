"""
Bearish Agent Module

This module implements the Bearish Agent, a specialized trading agent optimized
for bearish market conditions within the hierarchical agent architecture.
"""

import numpy as np
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from AITrader.Core.Python.Agents.base_agent import BaseAgent


class BearishAgent(BaseAgent):
    """
    A specialized trading agent optimized for bearish market conditions.
    
    This agent is designed to excel in market environments characterized by downward trends,
    implementing strategies that perform best during bear markets.
    """
    
    def __init__(self, observation_space, action_space, model_path=None, learning_rate=0.0003):
        """
        Initialize the Bearish Agent.
        
        Parameters:
        -----------
        observation_space : gym.spaces
            The observation space defining market features used by the agent
        action_space : gym.spaces
            The action space for trading decisions (typically buy, sell, hold)
        model_path : str, optional
            Path to a pre-trained model file
        learning_rate : float, optional
            Learning rate for the agent's policy network
        """
        super(BearishAgent, self).__init__(observation_space, action_space, model_path, learning_rate)
        
        # Bearish agent-specific parameters
        self.strategy_type = "bearish"
        self.position_sizing_factor = 0.8  # More conservative position sizing
        self.trend_confirmation_threshold = 0.7  # Higher threshold for confirming downtrend
        self.stop_loss_percentage = 0.02  # Tighter stop-loss for bearish markets
        self.max_holding_period = 15  # Shorter holding period for bearish markets
    
    def preprocess_observation(self, observation):
        """
        Preprocess the observation to emphasize features relevant to bearish conditions.
        
        Parameters:
        -----------
        observation : np.ndarray
            Raw market observation
            
        Returns:
        --------
        np.ndarray
            Processed observation with enhanced features for bearish market analysis
        """
        # This would typically include calculations like:
        # - Downtrend momentum indicators
        # - Resistance levels
        # - Short-term overbought conditions
        # - Volume analysis for selling pressure
        
        # For demonstration, we'll return the observation unchanged
        # In a real implementation, this would apply transformations
        return observation
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict an action based on the current market observation,
        with specific logic tailored for bearish market conditions.
        
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
        int
            The predicted action
        dict
            Additional information including bearish-specific indicators
        """
        # Preprocess observation for bearish-specific features
        processed_obs = self.preprocess_observation(observation)
        
        # Get base prediction
        action, info = super().predict(processed_obs, state, deterministic)
        
        # Add bearish-specific analysis
        info['downtrend_strength'] = self.calculate_downtrend_strength(observation)
        info['resistance_levels'] = self.identify_resistance_levels(observation)
        info['capitulation_signals'] = self.detect_capitulation(observation)
        
        return action, info
    
    def calculate_downtrend_strength(self, observation):
        """
        Calculate the strength of the bearish trend from the observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        float
            Downtrend strength indicator (0-1 where 1 is strongest bearish trend)
        """
        # This is a placeholder implementation
        # In a real system, this would analyze price momentum, volume, etc.
        # For demonstration, return a random value
        return np.random.random() * 0.5 + 0.5  # Random value between 0.5 and 1.0
    
    def identify_resistance_levels(self, observation):
        """
        Identify key resistance levels from the observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        list
            List of identified resistance price levels
        """
        # Placeholder implementation
        # In a real system, this would analyze price history to find resistance levels
        return [round(np.random.random() * 100, 2) for _ in range(3)]
    
    def detect_capitulation(self, observation):
        """
        Detect potential capitulation signals from the observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        dict
            Dictionary of capitulation signals and their strengths
        """
        # Placeholder implementation
        # In a real system, this would analyze volume spikes, price gaps, etc.
        signals = ['volume_spike', 'price_gap', 'momentum_exhaustion']
        return {signal: np.random.random() for signal in signals}
    
    def calculate_position_size(self, confidence, account_balance):
        """
        Calculate the appropriate position size based on signal confidence
        and account balance, with logic optimized for bearish markets.
        
        Parameters:
        -----------
        confidence : float
            Confidence level of the trading signal (0-1)
        account_balance : float
            Current account balance
            
        Returns:
        --------
        float
            Recommended position size as a percentage of account balance
        """
        # Base position sizing - more conservative in bearish markets
        base_size = 0.015  # 1.5% of account as base risk
        
        # Adjust based on confidence and downtrend strength
        adjusted_size = base_size * confidence * self.position_sizing_factor
        
        # Ensure position size is within reasonable limits
        return min(max(adjusted_size, 0.01), 0.08)  # Between 1% and 8%
    
    def calculate_stop_loss(self, entry_price, position_type):
        """
        Calculate appropriate stop-loss levels for bearish market trades.
        
        Parameters:
        -----------
        entry_price : float
            Entry price of the trade
        position_type : str
            'long' or 'short'
            
        Returns:
        --------
        float
            Stop-loss price level
        """
        if position_type == 'long':
            # For long positions in bearish markets, tighter stop-loss
            return entry_price * (1 - self.stop_loss_percentage)
        else:  # short
            # For short positions in bearish markets
            return entry_price * (1 + self.stop_loss_percentage * 0.8)  # Tighter for shorts
    
    def set_strategy_parameters(self, position_sizing=None, trend_threshold=None, 
                                stop_loss=None, max_holding=None):
        """
        Update the agent's strategy parameters.
        
        Parameters:
        -----------
        position_sizing : float, optional
            Position sizing factor
        trend_threshold : float, optional
            Threshold for confirming downtrend
        stop_loss : float, optional
            Stop-loss percentage
        max_holding : int, optional
            Maximum holding period
            
        Returns:
        --------
        self : BearishAgent
            The agent with updated parameters
        """
        if position_sizing is not None:
            self.position_sizing_factor = position_sizing
        if trend_threshold is not None:
            self.trend_confirmation_threshold = trend_threshold
        if stop_loss is not None:
            self.stop_loss_percentage = stop_loss
        if max_holding is not None:
            self.max_holding_period = max_holding
        
        return self


class BearishAgentCallback(BaseCallback):
    """
    Callback for monitoring and logging the Bearish Agent's training process.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        verbose : int
            Verbosity level
        """
        super(BearishAgentCallback, self).__init__(verbose)
        self.training_metrics = {
            'buy_actions': [],
            'sell_actions': [],
            'hold_actions': [],
            'avg_profit_per_trade': [],
            'win_rate': [],
            'max_drawdown': []
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
            # Assuming action space is {0: Buy, 1: Sell, 2: Hold}
            buy_actions = (self.model.actions == 0).sum()
            sell_actions = (self.model.actions == 1).sum()
            hold_actions = (self.model.actions == 2).sum()
            
            self.training_metrics['buy_actions'].append(buy_actions)
            self.training_metrics['sell_actions'].append(sell_actions)
            self.training_metrics['hold_actions'].append(hold_actions)
            
            # In bearish markets, we expect more sell than buy actions
            sell_to_buy_ratio = sell_actions / max(buy_actions, 1)
            
            # Log metrics periodically
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls} - Bearish Agent: "
                      f"Sell/Buy Ratio: {sell_to_buy_ratio:.2f}, "
                      f"Buy: {buy_actions}, Sell: {sell_actions}, Hold: {hold_actions}")
        
        return True
