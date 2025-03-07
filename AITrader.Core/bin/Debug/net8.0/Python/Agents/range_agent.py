"""
Range Agent Module

This module implements the Range Agent, a specialized trading agent optimized
for range-bound market conditions within the hierarchical agent architecture.
"""

import numpy as np
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from AITrader.Core.Python.Agents.base_agent import BaseAgent


class RangeAgent(BaseAgent):
    """
    A specialized trading agent optimized for range-bound market conditions.
    
    This agent is designed to excel in market environments characterized by sideways
    price action within defined ranges, implementing mean-reversion strategies.
    """
    
    def __init__(self, observation_space, action_space, model_path=None, learning_rate=0.0003):
        """
        Initialize the Range Agent.
        
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
        super(RangeAgent, self).__init__(observation_space, action_space, model_path, learning_rate)
        
        # Range agent-specific parameters
        self.strategy_type = "range"
        self.position_sizing_factor = 0.7  # More conservative for range markets
        self.range_detection_window = 20  # Window for detecting price ranges
        self.mean_reversion_threshold = 0.8  # Threshold for mean reversion signals
        self.overbought_threshold = 0.7  # Threshold for overbought conditions
        self.oversold_threshold = 0.3  # Threshold for oversold conditions
    
    def preprocess_observation(self, observation):
        """
        Preprocess the observation to emphasize features relevant to range-bound conditions.
        
        Parameters:
        -----------
        observation : np.ndarray
            Raw market observation
            
        Returns:
        --------
        np.ndarray
            Processed observation with enhanced features for range market analysis
        """
        # This would typically include calculations like:
        # - Range boundaries detection
        # - Mean reversion indicators
        # - Oscillator values (RSI, Stochastic)
        # - Support/resistance within the range
        
        # For demonstration, we'll return the observation unchanged
        # In a real implementation, this would apply transformations
        return observation
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict an action based on the current market observation,
        with specific logic tailored for range-bound market conditions.
        
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
            Additional information including range-specific indicators
        """
        # Preprocess observation for range-specific features
        processed_obs = self.preprocess_observation(observation)
        
        # Get base prediction
        action, info = super().predict(processed_obs, state, deterministic)
        
        # Add range-specific analysis
        info['range_boundaries'] = self.detect_range_boundaries(observation)
        info['oscillator_values'] = self.calculate_oscillators(observation)
        info['mean_reversion_strength'] = self.calculate_mean_reversion(observation)
        
        return action, info
    
    def detect_range_boundaries(self, observation):
        """
        Detect the upper and lower boundaries of the current price range.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        dict
            Upper and lower range boundaries
        """
        # This is a placeholder implementation
        # In a real system, this would analyze recent price history
        price_base = np.random.random() * 50 + 50  # Random base price between 50-100
        range_width = price_base * 0.05  # 5% range width
        
        return {
            'upper': round(price_base + range_width, 2),
            'lower': round(price_base - range_width, 2),
            'mid': round(price_base, 2),
            'width_pct': 5.0  # 5% range width
        }
    
    def calculate_oscillators(self, observation):
        """
        Calculate oscillator values for range trading.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        dict
            Dictionary of oscillator values
        """
        # Placeholder implementation
        # In a real system, this would calculate RSI, Stochastic, etc.
        return {
            'rsi': round(np.random.random() * 100, 2),  # 0-100
            'stochastic_k': round(np.random.random() * 100, 2),
            'stochastic_d': round(np.random.random() * 100, 2),
            'cci': round(np.random.random() * 200 - 100, 2)  # -100 to +100
        }
    
    def calculate_mean_reversion(self, observation):
        """
        Calculate mean reversion strength within the current range.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        float
            Mean reversion strength indicator (0-1)
        """
        # Placeholder implementation
        # In a real system, this would use distance from mean and momentum
        return np.random.random() * 0.6 + 0.4  # Random value between 0.4 and 1.0
    
    def is_overbought(self, observation):
        """
        Determine if the market is overbought within the range.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        bool
            True if market is overbought
        """
        oscillators = self.calculate_oscillators(observation)
        return oscillators['rsi'] > 70 or oscillators['stochastic_k'] > 80
    
    def is_oversold(self, observation):
        """
        Determine if the market is oversold within the range.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        bool
            True if market is oversold
        """
        oscillators = self.calculate_oscillators(observation)
        return oscillators['rsi'] < 30 or oscillators['stochastic_k'] < 20
    
    def calculate_position_size(self, confidence, account_balance):
        """
        Calculate the appropriate position size based on signal confidence
        and account balance, with logic optimized for range-bound markets.
        
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
        # Base position sizing - generally smaller in range markets
        base_size = 0.01  # 1% of account as base risk
        
        # Range markets often have clearer entry/exit, so scale based on distance from boundaries
        price_position = self.calculate_price_position_in_range(None)  # would use actual price
        
        # If price is near boundaries, increase position size (stronger mean-reversion signal)
        position_multiplier = 1.0
        if price_position < 0.2 or price_position > 0.8:
            position_multiplier = 1.5
        
        adjusted_size = base_size * confidence * self.position_sizing_factor * position_multiplier
        
        # Ensure position size is within reasonable limits
        return min(max(adjusted_size, 0.005), 0.03)  # Between 0.5% and 3%
    
    def calculate_price_position_in_range(self, current_price=None):
        """
        Calculate where the current price is within the detected range.
        
        Parameters:
        -----------
        current_price : float, optional
            Current price, if None a random value is used (for demonstration)
            
        Returns:
        --------
        float
            Position within range (0 = at lower bound, 1 = at upper bound)
        """
        # Placeholder implementation
        if current_price is None:
            return np.random.random()  # Random position in range
        
        boundaries = self.detect_range_boundaries(None)
        range_size = boundaries['upper'] - boundaries['lower']
        
        if range_size <= 0:
            return 0.5  # Default to middle if no valid range
        
        # Calculate position within range (0 to 1)
        position = (current_price - boundaries['lower']) / range_size
        return max(0, min(1, position))  # Clamp between 0 and 1
    
    def set_strategy_parameters(self, position_sizing=None, range_window=None, 
                               mean_reversion=None, overbought=None, oversold=None):
        """
        Update the agent's strategy parameters.
        
        Parameters:
        -----------
        position_sizing : float, optional
            Position sizing factor
        range_window : int, optional
            Window for detecting price ranges
        mean_reversion : float, optional
            Threshold for mean reversion signals
        overbought : float, optional
            Threshold for overbought conditions
        oversold : float, optional
            Threshold for oversold conditions
            
        Returns:
        --------
        self : RangeAgent
            The agent with updated parameters
        """
        if position_sizing is not None:
            self.position_sizing_factor = position_sizing
        if range_window is not None:
            self.range_detection_window = range_window
        if mean_reversion is not None:
            self.mean_reversion_threshold = mean_reversion
        if overbought is not None:
            self.overbought_threshold = overbought
        if oversold is not None:
            self.oversold_threshold = oversold
        
        return self


class RangeAgentCallback(BaseCallback):
    """
    Callback for monitoring and logging the Range Agent's training process.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        verbose : int
            Verbosity level
        """
        super(RangeAgentCallback, self).__init__(verbose)
        self.training_metrics = {
            'buy_actions': [],
            'sell_actions': [],
            'hold_actions': [],
            'avg_profit_per_trade': [],
            'win_rate': [],
            'mean_reversion_opportunities': []
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
            
            # In range markets, we expect more balanced buy and sell actions
            buy_sell_ratio = buy_actions / max(sell_actions, 1)
            
            # Log metrics periodically
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls} - Range Agent: "
                      f"Buy/Sell Ratio: {buy_sell_ratio:.2f}, "
                      f"Buy: {buy_actions}, Sell: {sell_actions}, Hold: {hold_actions}")
                
                # In range markets, we want to see alternating buy/sell rather than consecutive
                if hasattr(self.model, 'episode_actions'):
                    action_changes = sum(1 for i in range(1, len(self.model.episode_actions))
                                        if self.model.episode_actions[i] != self.model.episode_actions[i-1])
                    action_change_rate = action_changes / max(len(self.model.episode_actions) - 1, 1)
                    print(f"Action Change Rate: {action_change_rate:.2f} "
                          f"(Higher is better for range markets)")
        
        return True
