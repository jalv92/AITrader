"""
Base Agent Module

This module implements the base agent class that serves as a foundation 
for all specialized trading agents in the hierarchical architecture.
"""

import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class BaseAgent:
    """
    Base class for all specialized trading agents in the hierarchical architecture.
    
    This class provides common functionality and interface that will be shared
    across all specialized agents (Bullish, Bearish, Range).
    """
    
    def __init__(self, observation_space, action_space, model_path=None, learning_rate=0.0003):
        """
        Initialize the Base Agent.
        
        Parameters:
        -----------
        observation_space : gym.spaces
            The observation space defining market features used by the agent
        action_space : gym.spaces
            The action space for trading decisions
        model_path : str, optional
            Path to a pre-trained model file
        learning_rate : float, optional
            Learning rate for the agent's policy network
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Initialize the PPO model
        self.model = PPO("MlpPolicy", 
                         observation_space=self.observation_space,
                         action_space=self.action_space,
                         learning_rate=self.learning_rate,
                         verbose=1)
        
        # Load pre-trained model if provided
        if model_path:
            self.load(model_path)
        
        # Trading metrics
        self.trade_history = []
        self.performance_metrics = {
            'cumulative_return': 0.0,
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def train(self, env, total_timesteps=10000, callback=None):
        """
        Train the agent.
        
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
        self : BaseAgent
            The trained agent
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict an action based on the current market observation.
        
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
            Additional information including action probabilities
        """
        action, _ = self.model.predict(observation, state, deterministic)
        action_probs = self.model.policy.get_distribution(observation).distribution.probs.cpu().numpy()[0]
        
        return action, {'action_probs': action_probs}
    
    def get_action_confidence(self, observation, action=None):
        """
        Calculate the confidence level for a specific action or the selected action.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market features
        action : int, optional
            Specific action to get confidence for. If None, get confidence for the predicted action.
            
        Returns:
        --------
        float
            Confidence level between 0 and 1
        """
        if action is None:
            action, _ = self.predict(observation)
        
        action_probs = self.model.policy.get_distribution(observation).distribution.probs.cpu().numpy()[0]
        return float(action_probs[action])
    
    def record_trade(self, trade_data):
        """
        Record a completed trade for performance tracking.
        
        Parameters:
        -----------
        trade_data : dict
            Dictionary containing trade details (entry_price, exit_price, profit, etc.)
        """
        self.trade_history.append(trade_data)
        self.update_performance_metrics()
    
    def update_performance_metrics(self):
        """
        Update the agent's performance metrics based on trading history.
        """
        if not self.trade_history:
            return
        
        # Calculate basic metrics
        profits = [trade['profit'] for trade in self.trade_history]
        wins = sum(1 for profit in profits if profit > 0)
        
        self.performance_metrics['cumulative_return'] = sum(profits)
        self.performance_metrics['win_rate'] = wins / len(profits) if profits else 0
        self.performance_metrics['avg_profit_per_trade'] = np.mean(profits) if profits else 0
        
        # Calculate drawdown and Sharpe ratio (simplified)
        if len(profits) > 1:
            cumulative = np.cumsum(profits)
            max_so_far = np.maximum.accumulate(cumulative)
            drawdown = (max_so_far - cumulative)
            self.performance_metrics['max_drawdown'] = np.max(drawdown)
            
            returns = np.array(profits)
            self.performance_metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    def get_performance_metrics(self):
        """
        Get the agent's performance metrics.
        
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        return self.performance_metrics
    
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
        self : BaseAgent
            The agent with loaded model
        """
        self.model = PPO.load(path, 
                              observation_space=self.observation_space,
                              action_space=self.action_space)
        return self


class BaseAgentCallback(BaseCallback):
    """
    Base callback for monitoring and logging agent training process.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        verbose : int
            Verbosity level
        """
        super(BaseAgentCallback, self).__init__(verbose)
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actions_taken': {},
            'learning_rate': []
        }
    
    def _on_step(self):
        """
        Method called at each step of training.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        # Record current learning rate
        if hasattr(self.model, 'learning_rate'):
            self.training_metrics['learning_rate'].append(self.model.learning_rate)
        
        # Log information periodically
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            if len(self.training_metrics['episode_rewards']) > 0:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-100:])
                print(f"Step {self.n_calls}: Average reward over last 100 episodes: {avg_reward:.2f}")
        
        return True
    
    def on_rollout_end(self):
        """Record metrics at the end of a rollout."""
        if self.model.ep_info_buffer:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:  # reward and length
                    self.training_metrics['episode_rewards'].append(info['r'])
                    self.training_metrics['episode_lengths'].append(info['l'])
        
        # Record action distribution
        if hasattr(self.model, 'actions'):
            unique, counts = np.unique(self.model.actions, return_counts=True)
            for i, count in zip(unique, counts):
                if i not in self.training_metrics['actions_taken']:
                    self.training_metrics['actions_taken'][i] = 0
                self.training_metrics['actions_taken'][i] += count
