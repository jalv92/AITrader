"""
Bullish Agent Module

This module implements the Bullish Agent, a specialized trading agent optimized
for bullish market conditions within the hierarchical agent architecture.
"""

import numpy as np
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from AITrader.Core.Python.Agents.base_agent import BaseAgent


class BullishAgent(BaseAgent):
    """
    A specialized trading agent optimized for bullish market conditions.
    
    This agent is designed to excel in market environments characterized by upward trends,
    implementing strategies that perform best during bull markets.
    """
    
    def __init__(self, observation_space, action_space, model_path=None, learning_rate=0.0003):
        """
        Initialize the Bullish Agent.
        
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
        super(BullishAgent, self).__init__(observation_space, action_space, model_path, learning_rate)
        
        # Bullish agent-specific parameters
        self.strategy_type = "bullish"
        self.position_sizing_factor = 1.0  # Standard position sizing
        self.trend_confirmation_threshold = 0.6  # Threshold for confirming uptrend
        self.profit_target_multiplier = 3.0  # Risk-reward ratio target
        self.max_holding_period = 20  # Maximum holding period in time steps
    
    def preprocess_observation(self, observation):
        """
        Preprocess the observation to emphasize features relevant to bullish conditions.
        
        Parameters:
        -----------
        observation : np.ndarray
            Raw market observation
            
        Returns:
        --------
        np.ndarray
            Processed observation with enhanced features for bullish market analysis
        """
        # This would typically include calculations like:
        # - Short-term momentum indicators
        # - Breakout signals
        # - Pull-back identification
        # - Support level detection
        
        # For demonstration, we'll return the observation unchanged
        # In a real implementation, this would apply transformations
        return observation
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict an action based on the current market observation,
        with specific logic tailored for bullish market conditions.
        
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
            Additional information including bullish-specific indicators
        """
        # Preprocess observation for bullish-specific features
        processed_obs = self.preprocess_observation(observation)
        
        # Get base prediction
        action, info = super().predict(processed_obs, state, deterministic)
        
        # Add bullish-specific analysis
        info['trend_strength'] = self.calculate_trend_strength(observation)
        info['support_levels'] = self.identify_support_levels(observation)
        info['breakout_signals'] = self.detect_breakouts(observation)
        
        return action, info
    
    def calculate_trend_strength(self, observation):
        """
        Calculate the strength of the bullish trend from the observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        float
            Trend strength indicator (0-1 where 1 is strongest bullish trend)
        """
        # This is a placeholder implementation
        # In a real system, this would analyze price momentum, volume, etc.
        # For demonstration, return a random value
        return np.random.random() * 0.5 + 0.5  # Random value between 0.5 and 1.0
    
    def identify_support_levels(self, observation):
        """
        Identify key support levels from the observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        list
            List of identified support price levels
        """
        # Placeholder implementation
        # In a real system, this would analyze price history to find support levels
        return [round(np.random.random() * 100, 2) for _ in range(3)]
    
    def detect_breakouts(self, observation):
        """
        Detect potential breakout signals from the observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        dict
            Dictionary of breakout signals and their strengths
        """
        # Placeholder implementation
        # In a real system, this would analyze price and volume patterns
        signals = ['price_channel', 'volume_surge', 'resistance_break']
        return {signal: np.random.random() for signal in signals}
    
    def calculate_position_size(self, confidence, account_balance):
        """
        Calculate the appropriate position size based on signal confidence
        and account balance, with logic optimized for bullish markets.
        
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
        # Base position sizing
        base_size = 0.02  # 2% of account as base risk
        
        # Adjust based on confidence and trend strength
        adjusted_size = base_size * confidence * self.position_sizing_factor
        
        # Ensure position size is within reasonable limits
        return min(max(adjusted_size, 0.01), 0.1)  # Between 1% and 10%
    
    def set_strategy_parameters(self, position_sizing=None, trend_threshold=None, 
                                profit_target=None, max_holding=None):
        """
        Update the agent's strategy parameters.
        
        Parameters:
        -----------
        position_sizing : float, optional
            Position sizing factor
        trend_threshold : float, optional
            Threshold for confirming uptrend
        profit_target : float, optional
            Risk-reward ratio target
        max_holding : int, optional
            Maximum holding period
            
        Returns:
        --------
        self : BullishAgent
            The agent with updated parameters
        """
        if position_sizing is not None:
            self.position_sizing_factor = position_sizing
        if trend_threshold is not None:
            self.trend_confirmation_threshold = trend_threshold
        if profit_target is not None:
            self.profit_target_multiplier = profit_target
        if max_holding is not None:
            self.max_holding_period = max_holding
        
        return self


class BullishAgentCallback(BaseCallback):
    """
    Callback for monitoring and logging the Bullish Agent's training process.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        verbose : int
            Verbosity level
        """
        super(BullishAgentCallback, self).__init__(verbose)
        self.training_metrics = {
            'buy_actions': [],
            'sell_actions': [],
            'hold_actions': [],
            'avg_profit_per_trade': [],
            'win_rate': []
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
            
            # Log metrics periodically
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                # Calculate action distribution in percentage
                total_actions = buy_actions + sell_actions + hold_actions
                if total_actions > 0:
                    buy_pct = (buy_actions / total_actions) * 100
                    sell_pct = (sell_actions / total_actions) * 100
                    hold_pct = (hold_actions / total_actions) * 100
                    
                    print(f"Step {self.n_calls} - Action Distribution: "
                          f"Buy: {buy_pct:.1f}%, Sell: {sell_pct:.1f}%, Hold: {hold_pct:.1f}%")
        
        return True
