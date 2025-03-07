"""
Callback Handlers Module

This module implements custom callbacks for monitoring and handling training processes
within the hierarchical trading agent architecture.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsLoggerCallback(BaseCallback):
    """
    Custom callback for logging training metrics during agent training.
    """
    
    def __init__(self, agent_name: str, log_dir: str, verbose: int = 1, log_freq: int = 1000):
        """
        Initialize the metrics logger callback.
        
        Parameters:
        -----------
        agent_name : str
            Name of the agent being trained (e.g., 'no_trade_filter', 'meta_selector')
        log_dir : str
            Directory to save logs
        verbose : int
            Verbosity level
        log_freq : int
            Frequency (in steps) for logging
        """
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'learning_rate': [],
            'explained_variance': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'time_elapsed': []
        }
        self.start_time = time.time()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Method called at each step of training.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        # Log metrics periodically
        if self.n_calls % self.log_freq == 0:
            self.metrics['timesteps'].append(self.num_timesteps)
            self.metrics['time_elapsed'].append(time.time() - self.start_time)
            
            if len(self.model.ep_info_buffer) > 0:
                avg_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                avg_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                self.metrics['episode_rewards'].append(avg_reward)
                self.metrics['episode_lengths'].append(avg_length)
            
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                for key in ['explained_variance', 'policy_loss', 'value_loss', 'entropy_loss']:
                    if key in self.model.logger.name_to_value:
                        self.metrics[key].append(self.model.logger.name_to_value[key])
            
            # Log learning rate
            if hasattr(self.model, 'learning_rate'):
                self.metrics['learning_rate'].append(self.model.learning_rate)
            
            # Save metrics to file
            self._save_metrics()
            
            if self.verbose > 0:
                avg_reward = self.metrics['episode_rewards'][-1] if self.metrics['episode_rewards'] else float('nan')
                logger.info(f"{self.agent_name}: Step {self.num_timesteps} - Avg Reward: {avg_reward:.2f}")
        
        return True
    
    def _save_metrics(self):
        """Save the collected metrics to a CSV file."""
        metrics_df = pd.DataFrame({k: v for k, v in self.metrics.items() if v})
        metrics_path = os.path.join(self.log_dir, f"{self.agent_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Also save as JSON for easy loading
        metrics_json_path = os.path.join(self.log_dir, f"{self.agent_name}_metrics.json")
        with open(metrics_json_path, 'w') as f:
            json.dump({k: list(map(float, v)) for k, v in self.metrics.items() if v}, f)
    
    def on_training_end(self):
        """Method called at the end of training."""
        self._save_metrics()
        
        # Generate training performance plots
        self._plot_learning_curves()
    
    def _plot_learning_curves(self):
        """Generate and save learning curve plots."""
        if not self.metrics['episode_rewards']:
            return
        
        # Rewards plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['timesteps'], self.metrics['episode_rewards'])
        plt.title(f'{self.agent_name} - Training Rewards')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, f"{self.agent_name}_rewards.png"))
        plt.close()
        
        # Learning rate plot
        if self.metrics['learning_rate']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['timesteps'], self.metrics['learning_rate'])
            plt.title(f'{self.agent_name} - Learning Rate')
            plt.xlabel('Timesteps')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"{self.agent_name}_lr.png"))
            plt.close()
        
        # Episode length plot
        if self.metrics['episode_lengths']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['timesteps'], self.metrics['episode_lengths'])
            plt.title(f'{self.agent_name} - Episode Lengths')
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Length')
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"{self.agent_name}_lengths.png"))
            plt.close()


class ModelCheckpointCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints during training.
    """
    
    def __init__(self, agent_name: str, save_dir: str, save_freq: int = 10000, 
                 save_best: bool = True, verbose: int = 1):
        """
        Initialize the model checkpoint callback.
        
        Parameters:
        -----------
        agent_name : str
            Name of the agent being trained
        save_dir : str
            Directory to save model checkpoints
        save_freq : int
            Frequency (in steps) for saving checkpoints
        save_best : bool
            Whether to track and save the best model based on performance
        verbose : int
            Verbosity level
        """
        super(ModelCheckpointCallback, self).__init__(verbose)
        self.agent_name = agent_name
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_best = save_best
        self.best_mean_reward = -float('inf')
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Method called at each step of training.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        # Save regular checkpoint at specified frequency
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        
        # Track and save best model if enabled
        if self.save_best and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_best_model(mean_reward)
        
        return True
    
    def _save_checkpoint(self):
        """Save a regular model checkpoint."""
        checkpoint_path = os.path.join(
            self.save_dir, 
            f"{self.agent_name}_checkpoint_{self.num_timesteps}.zip"
        )
        self.model.save(checkpoint_path)
        
        if self.verbose > 0:
            logger.info(f"Saved checkpoint at {checkpoint_path}")
    
    def _save_best_model(self, mean_reward: float):
        """
        Save the best model based on performance.
        
        Parameters:
        -----------
        mean_reward : float
            Current mean reward that triggered the save
        """
        best_path = os.path.join(self.save_dir, f"{self.agent_name}_best_model.zip")
        self.model.save(best_path)
        
        # Save info about the best model
        info_path = os.path.join(self.save_dir, f"{self.agent_name}_best_model_info.json")
        with open(info_path, 'w') as f:
            json.dump({
                'timesteps': self.num_timesteps,
                'mean_reward': float(mean_reward),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        
        if self.verbose > 0:
            logger.info(f"Saved best model with mean reward: {mean_reward:.2f}")


class EarlyStoppingCallback(BaseCallback):
    """
    Custom callback for early stopping during training based on reward threshold or plateau.
    """
    
    def __init__(self, reward_threshold: Optional[float] = None, patience: int = 10, 
                 check_freq: int = 1000, min_delta: float = 0.01, verbose: int = 1):
        """
        Initialize the early stopping callback.
        
        Parameters:
        -----------
        reward_threshold : float, optional
            Threshold to stop training once reached
        patience : int
            Number of checks with no improvement after which to stop training
        check_freq : int
            Frequency (in steps) to check for early stopping conditions
        min_delta : float
            Minimum change in mean reward to qualify as an improvement
        verbose : int
            Verbosity level
        """
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.check_freq = check_freq
        self.min_delta = min_delta
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
    
    def _on_step(self) -> bool:
        """
        Method called at each step of training.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        if self.n_calls % self.check_freq == 0 and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            # Stop if we reach the reward threshold
            if self.reward_threshold is not None and mean_reward >= self.reward_threshold:
                if self.verbose > 0:
                    logger.info(f"Stopping training - reward threshold {self.reward_threshold} reached")
                return False
            
            # Check for improvement
            if mean_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                if self.verbose > 0:
                    logger.info(f"No improvement in {self.no_improvement_count} checks")
            
            # Stop if no improvement for several checks
            if self.no_improvement_count >= self.patience:
                if self.verbose > 0:
                    logger.info(f"Stopping training - no improvement for {self.patience} checks")
                return False
        
        return True


class HierarchicalTrainingCallback(BaseCallback):
    """
    Custom callback for coordinating training across multiple agents in the hierarchical architecture.
    """
    
    def __init__(self, agent_names: List[str], agents_callbacks: Dict[str, List[BaseCallback]], 
                 log_dir: str, verbose: int = 1):
        """
        Initialize the hierarchical training callback.
        
        Parameters:
        -----------
        agent_names : List[str]
            Names of agents in the hierarchical architecture
        agents_callbacks : Dict[str, List[BaseCallback]]
            Dictionary mapping agent names to their callback lists
        log_dir : str
            Directory to save logs
        verbose : int
            Verbosity level
        """
        super(HierarchicalTrainingCallback, self).__init__(verbose)
        self.agent_names = agent_names
        self.agents_callbacks = agents_callbacks
        self.log_dir = log_dir
        self.hierarchy_metrics = {
            'timesteps': [],
            'training_sequence': [],
            'active_agent': [],
            'hierarchy_rewards': []
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Method called at each step of training. This is used to track hierarchy-level metrics.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        # Record hierarchy-level metrics
        self.hierarchy_metrics['timesteps'].append(self.num_timesteps)
        
        # Identify current active agent (this will depend on your training implementation)
        active_agent = getattr(self.model, 'current_agent_name', self.agent_names[0])
        self.hierarchy_metrics['active_agent'].append(active_agent)
        
        # Record overall hierarchy performance
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.hierarchy_metrics['hierarchy_rewards'].append(mean_reward)
        
        return True
    
    def on_training_end(self):
        """Method called at the end of training to generate hierarchy-level reports."""
        # Save metrics
        metrics_df = pd.DataFrame({k: v for k, v in self.hierarchy_metrics.items() if v})
        metrics_path = os.path.join(self.log_dir, "hierarchy_training_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Generate visualization of hierarchy performance
        self._plot_hierarchy_performance()
    
    def _plot_hierarchy_performance(self):
        """Generate visualization of the hierarchical training performance."""
        if not self.hierarchy_metrics['hierarchy_rewards']:
            return
        
        # Plot hierarchy rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.hierarchy_metrics['timesteps'], self.hierarchy_metrics['hierarchy_rewards'])
        plt.title('Hierarchical Agent System - Overall Performance')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, "hierarchy_performance.png"))
        plt.close()
        
        # Plot agent distribution during training
        if self.hierarchy_metrics['active_agent']:
            agents = list(set(self.hierarchy_metrics['active_agent']))
            agent_counts = {agent: self.hierarchy_metrics['active_agent'].count(agent) for agent in agents}
            
            plt.figure(figsize=(10, 6))
            plt.bar(agent_counts.keys(), agent_counts.values())
            plt.title('Agent Training Distribution')
            plt.xlabel('Agent')
            plt.ylabel('Training Steps')
            plt.grid(axis='y')
            plt.savefig(os.path.join(self.log_dir, "agent_distribution.png"))
            plt.close()


class AgentSpecificEvalCallback(EvalCallback):
    """
    Extends EvalCallback with agent-specific evaluation metrics and visualizations.
    """
    
    def __init__(self, eval_env, agent_name: str, log_dir: str, n_eval_episodes=5, 
                 eval_freq=10000, verbose=1, best_model_save_path=None, 
                 deterministic=True, render=False):
        """
        Initialize the agent-specific evaluation callback.
        
        Parameters:
        -----------
        eval_env : gym.Env
            The environment used for evaluation
        agent_name : str
            Name of the agent being evaluated
        log_dir : str
            Directory to save logs
        n_eval_episodes : int
            Number of episodes to evaluate
        eval_freq : int
            Frequency of evaluation
        verbose : int
            Verbosity level
        best_model_save_path : str, optional
            Path to save the best model
        deterministic : bool
            Whether to use deterministic actions for evaluation
        render : bool
            Whether to render the environment during evaluation
        """
        super(AgentSpecificEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render
        )
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.agent_specific_metrics = {
            'timesteps': [],
            'mean_reward': [],
            'std_reward': [],
            'success_rate': [],
            'avg_episode_length': []
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Method called at each step of training.
        
        Returns:
        --------
        bool
            Whether training should continue
        """
        # Call parent's _on_step to perform regular evaluation
        continue_training = super()._on_step()
        
        # Record agent-specific metrics after evaluation
        if self.evaluations_timesteps and len(self.evaluations_timesteps) > len(self.agent_specific_metrics['timesteps']):
            # Get latest evaluation results
            timestep = self.evaluations_timesteps[-1]
            mean_reward = self.evaluations_rewards[-1]
            std_reward = self.evaluations_rewards_std[-1]
            
            # Record basic metrics
            self.agent_specific_metrics['timesteps'].append(timestep)
            self.agent_specific_metrics['mean_reward'].append(mean_reward)
            self.agent_specific_metrics['std_reward'].append(std_reward)
            
            # Calculate additional metrics from the last evaluation
            if hasattr(self, 'last_episode_info') and self.last_episode_info:
                success_rate = sum(1 for info in self.last_episode_info if info.get('success', False)) / len(self.last_episode_info)
                avg_length = np.mean([info.get('l', 0) for info in self.last_episode_info])
                
                self.agent_specific_metrics['success_rate'].append(success_rate)
                self.agent_specific_metrics['avg_episode_length'].append(avg_length)
            
            # Save agent-specific metrics
            self._save_agent_metrics()
            
            # Generate agent-specific plot if enough data is available
            if len(self.agent_specific_metrics['timesteps']) >= 2:
                self._plot_agent_evaluation()
        
        return continue_training
    
    def _save_agent_metrics(self):
        """Save agent-specific metrics to file."""
        metrics_df = pd.DataFrame({k: v for k, v in self.agent_specific_metrics.items() if v})
        metrics_path = os.path.join(self.log_dir, f"{self.agent_name}_eval_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
    
    def _plot_agent_evaluation(self):
        """Generate and save agent-specific evaluation plots."""
        # Rewards plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.agent_specific_metrics['timesteps'], self.agent_specific_metrics['mean_reward'])
        plt.fill_between(
            self.agent_specific_metrics['timesteps'],
            np.array(self.agent_specific_metrics['mean_reward']) - np.array(self.agent_specific_metrics['std_reward']),
            np.array(self.agent_specific_metrics['mean_reward']) + np.array(self.agent_specific_metrics['std_reward']),
            alpha=0.2
        )
        plt.title(f'{self.agent_name} - Evaluation Rewards')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, f"{self.agent_name}_eval_rewards.png"))
        plt.close()
        
        # Success rate plot if available
        if self.agent_specific_metrics['success_rate']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.agent_specific_metrics['timesteps'], self.agent_specific_metrics['success_rate'])
            plt.title(f'{self.agent_name} - Success Rate')
            plt.xlabel('Timesteps')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"{self.agent_name}_success_rate.png"))
            plt.close()


def create_training_callbacks(agent_name: str, log_dir: str, save_dir: str, eval_env=None,
                             save_freq: int = 10000, log_freq: int = 1000, eval_freq: int = 10000,
                             early_stopping: bool = True, reward_threshold: Optional[float] = None) -> CallbackList:
    """
    Create a standard set of callbacks for agent training.
    
    Parameters:
    -----------
    agent_name : str
        Name of the agent being trained
    log_dir : str
        Directory to save logs
    save_dir : str
        Directory to save model checkpoints
    eval_env : gym.Env, optional
        Environment for evaluation
    save_freq : int
        Frequency for saving model checkpoints
    log_freq : int
        Frequency for logging metrics
    eval_freq : int
        Frequency for evaluation
    early_stopping : bool
        Whether to use early stopping
    reward_threshold : float, optional
        Reward threshold for early stopping
        
    Returns:
    --------
    CallbackList
        Combined list of callbacks
    """
    # Create subdirectories
    agent_log_dir = os.path.join(log_dir, agent_name)
    agent_save_dir = os.path.join(save_dir, agent_name)
    os.makedirs(agent_log_dir, exist_ok=True)
    os.makedirs(agent_save_dir, exist_ok=True)
    
    # Create list of callbacks
    callbacks = [
        # Metrics logging
        MetricsLoggerCallback(
            agent_name=agent_name,
            log_dir=agent_log_dir,
            verbose=1,
            log_freq=log_freq
        ),
        
        # Model checkpointing
        ModelCheckpointCallback(
            agent_name=agent_name,
            save_dir=agent_save_dir,
            save_freq=save_freq,
            save_best=True,
            verbose=1
        )
    ]
    
    # Add evaluation callback if eval environment is provided
    if eval_env is not None:
        callbacks.append(
            AgentSpecificEvalCallback(
                eval_env=eval_env,
                agent_name=agent_name,
                log_dir=agent_log_dir,
                eval_freq=eval_freq,
                best_model_save_path=agent_save_dir,
                verbose=1
            )
        )
    
    # Add early stopping if requested
    if early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                reward_threshold=reward_threshold,
                patience=10,
                check_freq=eval_freq,
                verbose=1
            )
        )
    
    return CallbackList(callbacks)


def create_hierarchical_callback_system(agent_config: Dict[str, Dict], log_dir: str, save_dir: str) -> Dict:
    """
    Create a complete callback system for the hierarchical agent architecture.
    
    Parameters:
    -----------
    agent_config : Dict[str, Dict]
        Configuration dictionary for all agents in the hierarchy
    log_dir : str
        Base directory for logs
    save_dir : str
        Base directory for model checkpoints
        
    Returns:
    --------
    Dict
        Dictionary mapping agent names to their callback lists
    """
    callback_system = {}
    
    # Create callback lists for each agent
    for agent_name, config in agent_config.items():
        eval_env = config.get('eval_env', None)
        reward_threshold = config.get('reward_threshold', None)
        
        callback_system[agent_name] = create_training_callbacks(
            agent_name=agent_name,
            log_dir=log_dir,
            save_dir=save_dir,
            eval_env=eval_env,
            save_freq=config.get('save_freq', 10000),
            log_freq=config.get('log_freq', 1000),
            eval_freq=config.get('eval_freq', 10000),
            early_stopping=config.get('early_stopping', True),
            reward_threshold=reward_threshold
        )
    
    # Create a hierarchical callback to coordinate across agents
    hierarchical_callback = HierarchicalTrainingCallback(
        agent_names=list(agent_config.keys()),
        agents_callbacks=callback_system,
        log_dir=os.path.join(log_dir, 'hierarchy'),
        verbose=1
    )
    
    # Add the hierarchical callback to each agent's callback list
    for callbacks in callback_system.values():
        callbacks.callbacks.append(hierarchical_callback)
    
    return callback_system