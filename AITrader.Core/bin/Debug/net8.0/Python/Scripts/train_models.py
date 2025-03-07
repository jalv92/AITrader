"""
Train Models Script

This script loads market data from the data directory and trains the hierarchical RL agents.
"""

import os
import sys
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO

# Adjust Python path to find AITrader modules
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(root_dir)

# Import AITrader modules
from AITrader.Core.Python.Agents.no_trade_filter_agent import NoTradeFilterAgent
from AITrader.Core.Python.Agents.meta_agent_selector import MetaAgentSelector, MarketRegime
from AITrader.Core.Python.Agents.base_agent import BaseAgent
from AITrader.Core.Python.Agents.bullish_agent import BullishAgent
from AITrader.Core.Python.Agents.bearish_agent import BearishAgent
from AITrader.Core.Python.Agents.range_agent import RangeAgent
from AITrader.Core.Python.Agents.data_preparation import DataPreparationPipeline, FeatureSet
from AITrader.Core.Python.Agents.callback_handlers import create_training_callbacks
from AITrader.Core.Python.Scripts.data_loader import MarketDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for the hierarchical RL agent models using historical market data.
    """
    
    def __init__(self, data_dir: str, models_dir: str):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the market data CSV files
        models_dir : str
            Directory to save the trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = MarketDataLoader(data_dir)
        
        # Initialize data preparation pipeline
        self.data_pipeline = DataPreparationPipeline()
        
        # Set training parameters
        self.training_timesteps = {
            "no_trade_filter": 100000,
            "meta_selector": 100000,
            "bullish_agent": 100000,
            "bearish_agent": 100000,
            "range_agent": 100000
        }
        
        self.observation_spaces = {}
        self.action_spaces = {}
        self.training_envs = {}
        
    def load_and_prepare_data(self, dataset_name: str = None):
        """
        Load and prepare data for training.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Specific dataset to use. If None, the first available dataset is used.
        """
        logger.info("Loading market data...")
        
        # List available datasets
        datasets = self.data_loader.list_available_datasets()
        if not datasets:
            raise ValueError(f"No datasets found in {self.data_dir}")
        
        # Use specified dataset or first available
        if dataset_name:
            if dataset_name not in datasets:
                raise ValueError(f"Dataset {dataset_name} not found in {self.data_dir}")
            selected_dataset = dataset_name
        else:
            selected_dataset = datasets[0]
        
        logger.info(f"Using dataset: {selected_dataset}")
        
        # Load and prepare data
        prepared_data = self.data_loader.load_and_prepare_for_agents(selected_dataset)
        
        # Prepare data for each agent type
        logger.info("Preparing data for agent training...")
        
        # Get raw data with regimes
        raw_data = prepared_data["data_with_regimes"]
        
        # Prepare data for different feature sets
        data_sets = {}
        
        for feature_set in FeatureSet:
            try:
                # Prepare data
                data = self.data_pipeline.prepare_data(raw_data, feature_set)
                
                # Split into training and testing
                train_data, test_data = self.data_pipeline.split_train_test(data)
                
                # Store data sets
                data_sets[feature_set] = {
                    "train": train_data,
                    "test": test_data
                }
                
                logger.info(f"Prepared {feature_set.name} data: train={len(train_data)}, test={len(test_data)}")
            except Exception as e:
                logger.error(f"Error preparing data for {feature_set.name}: {e}")
        
        # Create observation spaces
        self.observation_spaces = self.data_pipeline.create_observation_spaces()
        
        # Create action spaces
        self.action_spaces = {
            FeatureSet.NO_TRADE: spaces.Discrete(2),  # 0 = No-Trade, 1 = Trade
            FeatureSet.META: spaces.Discrete(3),      # 0 = Bullish, 1 = Bearish, 2 = Range
            FeatureSet.BULLISH: spaces.Discrete(3),   # 0 = Buy, 1 = Sell, 2 = Hold
            FeatureSet.BEARISH: spaces.Discrete(3),   # 0 = Buy, 1 = Sell, 2 = Hold
            FeatureSet.RANGE: spaces.Discrete(3)      # 0 = Buy, 1 = Sell, 2 = Hold
        }
        
        # Save normalization parameters
        norm_params_path = os.path.join(self.models_dir, "normalization_params.pkl")
        self.data_pipeline.save_normalization_params(norm_params_path)
        
        # Create training environments
        logger.info("Creating training environments...")
        self.training_envs = self._create_training_environments(data_sets)
        
        logger.info("Data preparation completed")
        
    def _create_training_environments(self, data_sets):
        """
        Create training environments for each agent type.
        
        Parameters:
        -----------
        data_sets : dict
            Dictionary of data sets for each feature set
            
        Returns:
        --------
        dict
            Dictionary of training environments
        """
        envs = {}
        
        # Create environment for No-Trade Filter
        envs["no_trade_filter"] = self._create_no_trade_filter_env(
            data_sets[FeatureSet.NO_TRADE]["train"],
            self.observation_spaces[FeatureSet.NO_TRADE],
            self.action_spaces[FeatureSet.NO_TRADE]
        )
        
        # Create environment for Meta-Agent Selector
        envs["meta_selector"] = self._create_meta_selector_env(
            data_sets[FeatureSet.META]["train"],
            self.observation_spaces[FeatureSet.META],
            self.action_spaces[FeatureSet.META]
        )
        
        # Create environments for specialized agents
        for agent_type, feature_set in [
            ("bullish_agent", FeatureSet.BULLISH),
            ("bearish_agent", FeatureSet.BEARISH),
            ("range_agent", FeatureSet.RANGE)
        ]:
            envs[agent_type] = self._create_specialized_agent_env(
                data_sets[feature_set]["train"],
                self.observation_spaces[feature_set],
                self.action_spaces[feature_set],
                agent_type
            )
        
        return envs
    
    def _create_no_trade_filter_env(self, data, observation_space, action_space):
        """
        Create a training environment for the No-Trade Filter agent.
        
        This is a simplified placeholder implementation. In practice, you would create
        a proper Gym environment with appropriate reward structure based on the trading outcomes.
        """
        class NoTradeFilterEnv(gym.Env):
            def __init__(self, data, observation_space, action_space):
                self.data = data.copy().reset_index(drop=True)
                self.current_step = 0
                self.observation_space = observation_space
                self.action_space = action_space
                
            def reset(self):
                self.current_step = 0
                return self._get_observation()
                
            def step(self, action):
                # Move to next step
                self.current_step += 1
                
                # Check if done
                done = self.current_step >= len(self.data) - 1
                
                # Calculate reward
                # In a real implementation, this would be based on the trading outcome
                reward = 0
                if action == 1:  # Trade
                    # Simplified reward: positive if next price goes up, negative if down
                    if self.current_step < len(self.data) - 1:
                        price_change = self.data.iloc[self.current_step + 1]["close"] - self.data.iloc[self.current_step]["close"]
                        reward = np.sign(price_change)
                
                # Return observation, reward, done, info
                return self._get_observation(), reward, done, {}
                
            def _get_observation(self):
                # Get the current observation
                return self.data.iloc[self.current_step].values
        
        return NoTradeFilterEnv(data, observation_space, action_space)
    
    def _create_meta_selector_env(self, data, observation_space, action_space):
        """
        Create a training environment for the Meta-Agent Selector.
        
        This is a simplified placeholder implementation. In practice, you would create
        a proper Gym environment with appropriate reward structure based on regime detection accuracy.
        """
        class MetaSelectorEnv(gym.Env):
            def __init__(self, data, observation_space, action_space):
                self.data = data.copy().reset_index(drop=True)
                self.current_step = 0
                self.observation_space = observation_space
                self.action_space = action_space
                
                # Map regime labels
                self.regime_map = {
                    "bullish": 0,
                    "bearish": 1,
                    "range": 2
                }
                
            def reset(self):
                self.current_step = 0
                return self._get_observation()
                
            def step(self, action):
                # Move to next step
                self.current_step += 1
                
                # Check if done
                done = self.current_step >= len(self.data) - 1
                
                # Calculate reward
                # In a real implementation, this would be based on regime detection accuracy
                reward = 0
                if "regime" in self.data.columns:
                    true_regime = self.regime_map.get(self.data.iloc[self.current_step]["regime"], 2)
                    reward = 1 if action == true_regime else -0.5
                
                # Return observation, reward, done, info
                return self._get_observation(), reward, done, {}
                
            def _get_observation(self):
                # Get the current observation
                return self.data.iloc[self.current_step].values
        
        return MetaSelectorEnv(data, observation_space, action_space)
    
    def _create_specialized_agent_env(self, data, observation_space, action_space, agent_type):
        """
        Create a training environment for a specialized agent.
        
        This is a simplified placeholder implementation. In practice, you would create
        a proper Gym environment with appropriate reward structure based on trading outcomes.
        """
        class SpecializedAgentEnv(gym.Env):
            def __init__(self, data, observation_space, action_space, agent_type):
                self.data = data.copy().reset_index(drop=True)
                self.current_step = 0
                self.observation_space = observation_space
                self.action_space = action_space
                self.agent_type = agent_type
                self.position = 0  # 0 = flat, 1 = long, -1 = short
                
            def reset(self):
                self.current_step = 0
                self.position = 0
                return self._get_observation()
                
            def step(self, action):
                # Current price
                current_price = self.data.iloc[self.current_step]["close"]
                
                # Determine trading action
                # 0 = Buy, 1 = Sell, 2 = Hold
                new_position = self.position
                if action == 0:  # Buy
                    new_position = 1
                elif action == 1:  # Sell
                    new_position = -1
                # Hold action leaves position unchanged
                
                # Move to next step
                self.current_step += 1
                
                # Check if done
                done = self.current_step >= len(self.data) - 1
                
                # Calculate reward
                reward = 0
                if not done:
                    # Next price
                    next_price = self.data.iloc[self.current_step]["close"]
                    
                    # Price change
                    price_change = next_price - current_price
                    
                    # Position-based reward
                    if new_position == 1:  # Long
                        reward = price_change * 100  # Scale for better learning
                    elif new_position == -1:  # Short
                        reward = -price_change * 100  # Scale for better learning
                    
                    # Adjust rewards based on agent type
                    if self.agent_type == "bullish_agent":
                        # Bullish agent should prefer long positions in uptrends
                        if "regime" in self.data.columns and self.data.iloc[self.current_step]["regime"] == "bullish":
                            if new_position == 1:
                                reward *= 1.5  # Boost reward for correct behavior
                    
                    elif self.agent_type == "bearish_agent":
                        # Bearish agent should prefer short positions in downtrends
                        if "regime" in self.data.columns and self.data.iloc[self.current_step]["regime"] == "bearish":
                            if new_position == -1:
                                reward *= 1.5  # Boost reward for correct behavior
                    
                    elif self.agent_type == "range_agent":
                        # Range agent should buy low and sell high
                        if "regime" in self.data.columns and self.data.iloc[self.current_step]["regime"] == "range":
                            if (new_position == 1 and price_change > 0) or (new_position == -1 and price_change < 0):
                                reward *= 1.5  # Boost reward for correct behavior
                
                # Update position
                self.position = new_position
                
                # Return observation, reward, done, info
                return self._get_observation(), reward, done, {}
                
            def _get_observation(self):
                # Get the current observation
                return self.data.iloc[self.current_step].values
        
        return SpecializedAgentEnv(data, observation_space, action_space, agent_type)
    
    def train_agents(self):
        """
        Train the hierarchical agents.
        """
        # Ensure environments are created
        if not self.training_envs:
            raise ValueError("Training environments not created. Call load_and_prepare_data() first.")
        
        # Create log and checkpoints directories
        log_dir = os.path.join(self.models_dir, "logs")
        checkpoint_dir = os.path.join(self.models_dir, "checkpoints")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Train No-Trade Filter
        logger.info("Training No-Trade Filter Agent...")
        no_trade_filter = NoTradeFilterAgent(
            observation_space=self.observation_spaces[FeatureSet.NO_TRADE],
            action_space=self.action_spaces[FeatureSet.NO_TRADE]
        )
        
        ntf_callbacks = create_training_callbacks(
            agent_name="no_trade_filter",
            log_dir=log_dir,
            save_dir=checkpoint_dir
        )
        
        no_trade_filter.train(
            env=self.training_envs["no_trade_filter"],
            total_timesteps=self.training_timesteps["no_trade_filter"],
            callback=ntf_callbacks
        )
        
        # Save No-Trade Filter model
        no_trade_filter.save(os.path.join(self.models_dir, "no_trade_filter.zip"))
        
        # Train Meta-Agent Selector
        logger.info("Training Meta-Agent Selector...")
        meta_selector = MetaAgentSelector(
            observation_space=self.observation_spaces[FeatureSet.META],
            action_space=self.action_spaces[FeatureSet.META]
        )
        
        meta_callbacks = create_training_callbacks(
            agent_name="meta_selector",
            log_dir=log_dir,
            save_dir=checkpoint_dir
        )
        
        meta_selector.train(
            env=self.training_envs["meta_selector"],
            total_timesteps=self.training_timesteps["meta_selector"],
            callback=meta_callbacks
        )
        
        # Save Meta-Agent Selector model
        meta_selector.save(os.path.join(self.models_dir, "meta_selector.zip"))
        
        # Train specialized agents
        for agent_name, agent_class, feature_set in [
            ("bullish_agent", BullishAgent, FeatureSet.BULLISH),
            ("bearish_agent", BearishAgent, FeatureSet.BEARISH),
            ("range_agent", RangeAgent, FeatureSet.RANGE)
        ]:
            logger.info(f"Training {agent_name}...")
            agent = agent_class(
                observation_space=self.observation_spaces[feature_set],
                action_space=self.action_spaces[feature_set]
            )
            
            agent_callbacks = create_training_callbacks(
                agent_name=agent_name,
                log_dir=log_dir,
                save_dir=checkpoint_dir
            )
            
            agent.train(
                env=self.training_envs[agent_name],
                total_timesteps=self.training_timesteps[agent_name],
                callback=agent_callbacks
            )
            
            # Save agent model
            agent.save(os.path.join(self.models_dir, f"{agent_name}.zip"))
        
        logger.info("Agent training completed")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train AITrader RL agent models')
    parser.add_argument('--data-dir', type=str, default=None, help='Directory containing market data CSV files')
    parser.add_argument('--models-dir', type=str, default=None, help='Directory to save trained models')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to use')
    args = parser.parse_args()
    
    # Get directory paths
    if args.data_dir is None:
        data_dir = os.path.abspath(os.path.join(root_dir, "data"))
    else:
        data_dir = os.path.abspath(args.data_dir)
    
    if args.models_dir is None:
        models_dir = os.path.abspath(os.path.join(root_dir, "models"))
    else:
        models_dir = os.path.abspath(args.models_dir)
    
    # Create model trainer
    trainer = ModelTrainer(data_dir, models_dir)
    
    try:
        # Load and prepare data
        trainer.load_and_prepare_data(args.dataset)
        
        # Train agents
        trainer.train_agents()
        
        logger.info(f"Models saved to {models_dir}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()