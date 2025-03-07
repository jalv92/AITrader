"""
No-Trade Filter Agent Module

This module implements the top-level No-Trade Filter Agent responsible for determining 
whether current market conditions are suitable for trading.

The agent analyzes various market indicators to make a binary decision: 
Trade or No-Trade.
"""

import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class NoTradeFilterAgent:
    """
    A top-level agent that determines if market conditions are suitable for trading.
    
    This agent serves as the first decision layer in the hierarchical agent architecture,
    acting as a gatekeeper that prevents trading during unfavorable market conditions.
    """
    
    def __init__(self, observation_space, action_space, model_path=None, learning_rate=0.0003):
        """
        Initialize the No-Trade Filter Agent.
        
        Parameters:
        -----------
        observation_space : gym.spaces
            The observation space defining market features used by the agent
        action_space : gym.spaces
            The action space (typically binary: Trade/No-Trade)
        model_path : str, optional
            Path to a pre-trained model file
        learning_rate : float, optional
            Learning rate for the agent's policy network
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Initialize the PPO model for the No-Trade Filter Agent
        self.model = PPO("MlpPolicy", 
                         observation_space=self.observation_space,
                         action_space=self.action_space,
                         learning_rate=self.learning_rate,
                         verbose=1)
        
        # Load pre-trained model if provided
        if model_path:
            self.load(model_path)
            
        self.trade_threshold = 0.5  # Confidence threshold for trading decision
    
    def train(self, env, total_timesteps=10000, callback=None):
        """
        Train the No-Trade Filter Agent.
        
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
        self : NoTradeFilterAgent
            The trained agent
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict whether to trade or not based on the current market observation.
        
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
            Binary decision: 1 (Trade) or 0 (No-Trade)
        float
            Confidence level of the decision
        """
        action, _ = self.model.predict(observation, state, deterministic)
        return action, self.get_trade_confidence(observation)
    
    def get_trade_confidence(self, observation):
        """
        Calculate the confidence level of the trading decision.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
            
        Returns:
        --------
        float
            Confidence level between 0 and 1
        """
        # This is a simplified implementation
        # In a real-world scenario, this would involve a more sophisticated calculation
        action_probs = self.model.policy.get_distribution(observation).distribution.probs.cpu().numpy()
        return float(action_probs[0][1])  # Probability of the "Trade" action
    
    def should_trade(self, observation, confidence_threshold=None):
        """
        Make a binary decision on whether to trade based on current market conditions.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
        confidence_threshold : float, optional
            Override the default confidence threshold
            
        Returns:
        --------
        bool
            True if trading is recommended, False otherwise
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.trade_threshold
        _, confidence = self.predict(observation)
        return confidence >= threshold
    
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
        self : NoTradeFilterAgent
            The agent with loaded model
        """
        self.model = PPO.load(path, 
                              observation_space=self.observation_space,
                              action_space=self.action_space)
        return self


class NoTradeFilterCallback(BaseCallback):
    """
    Callback for monitoring and logging the No-Trade Filter Agent's training process.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        verbose : int
            Verbosity level
        """
        super(NoTradeFilterCallback, self).__init__(verbose)
        self.training_metrics = {
            'no_trade_decisions': [],
            'trade_decisions': [],
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
        if hasattr(self.model, 'action_probability') and len(self.model.action_probability) > 0:
            no_trade_decisions = (self.model.actions == 0).sum()
            trade_decisions = (self.model.actions == 1).sum()
            avg_confidence = np.mean(np.max(self.model.action_probability, axis=1))
            
            self.training_metrics['no_trade_decisions'].append(no_trade_decisions)
            self.training_metrics['trade_decisions'].append(trade_decisions)
            self.training_metrics['avg_confidence'].append(avg_confidence)
            
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls}: No-Trade: {no_trade_decisions}, Trade: {trade_decisions}, "
                      f"Avg Confidence: {avg_confidence:.4f}")
        
        return True
