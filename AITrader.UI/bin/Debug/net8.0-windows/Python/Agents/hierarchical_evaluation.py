"""
Hierarchical Evaluation Module

This module provides comprehensive evaluation tools for the hierarchical trading agent system,
measuring performance metrics across all agent levels and creating visualizations for analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union

from AITrader.Core.Python.Agents.no_trade_filter_agent import NoTradeFilterAgent
from AITrader.Core.Python.Agents.meta_agent_selector import MetaAgentSelector, MarketRegime
from AITrader.Core.Python.Agents.base_agent import BaseAgent
from AITrader.Core.Python.Agents.bullish_agent import BullishAgent
from AITrader.Core.Python.Agents.bearish_agent import BearishAgent
from AITrader.Core.Python.Agents.range_agent import RangeAgent


class HierarchicalEvaluator:
    """
    Evaluates the performance of the complete hierarchical agent system.
    
    This class provides methods to assess the performance of:
    1. The No-Trade Filter Agent's accuracy in identifying suitable trading conditions
    2. The Meta-Agent Selector's accuracy in market regime classification
    3. Each specialized agent's trading performance in its respective regime
    4. The overall system performance in terms of profitability and risk metrics
    """
    
    def __init__(self, no_trade_filter: NoTradeFilterAgent, meta_selector: MetaAgentSelector,
                 bullish_agent: BullishAgent, bearish_agent: BearishAgent, range_agent: RangeAgent):
        """
        Initialize the HierarchicalEvaluator.
        
        Parameters:
        -----------
        no_trade_filter : NoTradeFilterAgent
            The top-level No-Trade Filter Agent
        meta_selector : MetaAgentSelector
            The Meta-Agent Selector for market regime classification
        bullish_agent : BullishAgent
            The specialized agent for bullish markets
        bearish_agent : BearishAgent
            The specialized agent for bearish markets
        range_agent : RangeAgent
            The specialized agent for range-bound markets
        """
        self.no_trade_filter = no_trade_filter
        self.meta_selector = meta_selector
        self.bullish_agent = bullish_agent
        self.bearish_agent = bearish_agent
        self.range_agent = range_agent
        
        # Register the agents with the meta selector
        self.meta_selector.register_specialized_agent(MarketRegime.BULLISH, self.bullish_agent)
        self.meta_selector.register_specialized_agent(MarketRegime.BEARISH, self.bearish_agent)
        self.meta_selector.register_specialized_agent(MarketRegime.RANGE, self.range_agent)
        
        # Performance tracking
        self.evaluation_results = {
            'overall': {},
            'no_trade_filter': {},
            'meta_selector': {},
            'specialized_agents': {
                'bullish': {},
                'bearish': {},
                'range': {}
            }
        }
        
        # Tracking decision history
        self.decisions = []
    
    def evaluate(self, test_env, episodes=10, render=False, log_frequency=1):
        """
        Evaluate the complete hierarchical agent system on a test environment.
        
        Parameters:
        -----------
        test_env : gym.Env
            The test environment for evaluation
        episodes : int
            Number of episodes to evaluate
        render : bool
            Whether to render the environment during evaluation
        log_frequency : int
            Frequency for logging evaluation progress
            
        Returns:
        --------
        dict
            Dictionary containing comprehensive evaluation metrics
        """
        # Reset evaluation results
        self.evaluation_results = {
            'overall': {
                'total_return': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trades_executed': 0,
                'trades_skipped': 0,
                'episode_returns': []
            },
            'no_trade_filter': {
                'accuracy': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            },
            'meta_selector': {
                'accuracy': 0.0,
                'confusion_matrix': np.zeros((3, 3)),  # 3 regimes
                'regime_durations': {
                    MarketRegime.BULLISH: [],
                    MarketRegime.BEARISH: [],
                    MarketRegime.RANGE: []
                },
                'regime_transitions': {}
            },
            'specialized_agents': {
                'bullish': {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'trades': 0
                },
                'bearish': {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'trades': 0
                },
                'range': {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'trades': 0
                }
            }
        }
        
        # Clear decisions history
        self.decisions = []
        
        # Performance tracking variables
        all_returns = []
        all_trades = []
        regime_counts = {
            MarketRegime.BULLISH: 0,
            MarketRegime.BEARISH: 0,
            MarketRegime.RANGE: 0,
            MarketRegime.UNKNOWN: 0
        }
        
        # Run evaluation episodes
        for episode in range(episodes):
            observation = test_env.reset()
            done = False
            episode_return = 0
            episode_steps = 0
            
            current_regime = None
            regime_start_step = 0
            
            while not done:
                # Step 1: No-Trade Filter decision
                should_trade, ntf_confidence = self.no_trade_filter.predict(observation)
                should_trade = bool(should_trade)
                
                # Step 2: If should trade, identify market regime
                if should_trade:
                    regime, regime_confidence = self.meta_selector.predict(observation)
                    
                    # Track regime durations and transitions
                    if current_regime != regime:
                        if current_regime is not None:
                            regime_duration = episode_steps - regime_start_step
                            self.evaluation_results['meta_selector']['regime_durations'][current_regime].append(regime_duration)
                            
                            # Track transition
                            transition_key = f"{current_regime.name}_to_{regime.name}"
                            if transition_key not in self.evaluation_results['meta_selector']['regime_transitions']:
                                self.evaluation_results['meta_selector']['regime_transitions'][transition_key] = 0
                            self.evaluation_results['meta_selector']['regime_transitions'][transition_key] += 1
                        
                        current_regime = regime
                        regime_start_step = episode_steps
                    
                    # Step 3: Select specialized agent based on regime
                    if regime == MarketRegime.BULLISH:
                        specialized_agent = self.bullish_agent
                        agent_key = 'bullish'
                        regime_counts[MarketRegime.BULLISH] += 1
                    elif regime == MarketRegime.BEARISH:
                        specialized_agent = self.bearish_agent
                        agent_key = 'bearish'
                        regime_counts[MarketRegime.BEARISH] += 1
                    elif regime == MarketRegime.RANGE:
                        specialized_agent = self.range_agent
                        agent_key = 'range'
                        regime_counts[MarketRegime.RANGE] += 1
                    else:
                        specialized_agent = None
                        agent_key = None
                        regime_counts[MarketRegime.UNKNOWN] += 1
                    
                    # Step 4: Execute trading action using specialized agent
                    if specialized_agent is not None:
                        action, action_info = specialized_agent.predict(observation)
                        
                        # Record trade
                        trade_info = {
                            'episode': episode,
                            'step': episode_steps,
                            'agent': agent_key,
                            'action': action,
                            'confidence': action_info['action_probs'][action],
                            'regime': regime.name,
                            'ntf_confidence': ntf_confidence
                        }
                        all_trades.append(trade_info)
                        
                        # Update specialized agent metrics
                        self.evaluation_results['specialized_agents'][agent_key]['trades'] += 1
                    else:
                        # No suitable agent for the regime
                        action = 2  # Assuming 2 is "hold" or "do nothing"
                else:
                    # No-Trade Filter decided not to trade
                    action = 2  # Assuming 2 is "hold" or "do nothing"
                
                # Record decision
                decision = {
                    'episode': episode,
                    'step': episode_steps,
                    'should_trade': should_trade,
                    'ntf_confidence': ntf_confidence,
                    'action': action
                }
                
                if should_trade and specialized_agent is not None:
                    decision.update({
                        'regime': regime.name,
                        'regime_confidence': regime_confidence[regime],
                        'agent': agent_key
                    })
                
                self.decisions.append(decision)
                
                # Take action in environment
                next_observation, reward, done, info = test_env.step(action)
                
                # Update episode return
                episode_return += reward
                
                # Check for trade completion for specialized agent metrics
                if info.get('trade_completed', False) and should_trade and specialized_agent is not None:
                    self.update_agent_metrics(agent_key, info)
                
                # Render if requested
                if render:
                    test_env.render()
                
                # Update observation
                observation = next_observation
                episode_steps += 1
            
            # Record episode return
            all_returns.append(episode_return)
            self.evaluation_results['overall']['episode_returns'].append(episode_return)
            
            # Log progress
            if episode % log_frequency == 0:
                print(f"Episode {episode}/{episodes} - Return: {episode_return:.2f}")
        
        # Calculate overall metrics
        if all_returns:
            self.evaluation_results['overall']['total_return'] = sum(all_returns)
            self.evaluation_results['overall']['trades_executed'] = len(all_trades)
            self.evaluation_results['overall']['trades_skipped'] = len(self.decisions) - len(all_trades)
            
            # Calculate Sharpe ratio (simplified)
            returns_array = np.array(all_returns)
            if len(returns_array) > 1 and returns_array.std() > 0:
                self.evaluation_results['overall']['sharpe_ratio'] = (
                    returns_array.mean() / returns_array.std()) * np.sqrt(252)
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumsum(returns_array)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns)
            self.evaluation_results['overall']['max_drawdown'] = drawdown.max() if len(drawdown) > 0 else 0
        
        # Calculate regime distribution
        total_regimes = sum(regime_counts.values())
        if total_regimes > 0:
            for regime, count in regime_counts.items():
                regime_pct = (count / total_regimes) * 100
                self.evaluation_results['meta_selector'][f'{regime.name.lower()}_percentage'] = regime_pct
        
        return self.evaluation_results
    
    def update_agent_metrics(self, agent_key, trade_info):
        """
        Update metrics for a specialized agent based on completed trade information.
        
        Parameters:
        -----------
        agent_key : str
            Key identifying the specialized agent ('bullish', 'bearish', 'range')
        trade_info : dict
            Dictionary containing trade information
        """
        # Extract trade result
        profit = trade_info.get('profit', 0)
        
        # Update agent metrics
        agent_metrics = self.evaluation_results['specialized_agents'][agent_key]
        agent_metrics['total_return'] += profit
        
        # Update win/loss counter
        if 'win_count' not in agent_metrics:
            agent_metrics['win_count'] = 0
            agent_metrics['loss_count'] = 0
        
        if profit > 0:
            agent_metrics['win_count'] += 1
        else:
            agent_metrics['loss_count'] += 1
        
        # Calculate win rate
        total_trades = agent_metrics['win_count'] + agent_metrics['loss_count']
        if total_trades > 0:
            agent_metrics['win_rate'] = agent_metrics['win_count'] / total_trades
    
    def calculate_no_trade_filter_accuracy(self, ground_truth):
        """
        Calculate the accuracy of the No-Trade Filter Agent.
        
        Parameters:
        -----------
        ground_truth : list
            List of boolean values indicating if each decision point was actually
            a good trading opportunity (True) or not (False)
            
        Returns:
        --------
        dict
            Dictionary containing accuracy metrics
        """
        if len(self.decisions) != len(ground_truth):
            raise ValueError("Length of decisions and ground truth must match")
        
        tp = fp = tn = fn = 0
        
        for decision, truth in zip(self.decisions, ground_truth):
            if decision['should_trade'] and truth:
                tp += 1  # True positive
            elif decision['should_trade'] and not truth:
                fp += 1  # False positive
            elif not decision['should_trade'] and not truth:
                tn += 1  # True negative
            else:  # not decision['should_trade'] and truth
                fn += 1  # False negative
        
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        self.evaluation_results['no_trade_filter'].update(metrics)
        return metrics
    
    def calculate_regime_classification_accuracy(self, ground_truth_regimes):
        """
        Calculate the accuracy of the Meta-Agent Selector in regime classification.
        
        Parameters:
        -----------
        ground_truth_regimes : list
            List of MarketRegime values indicating the true market regime for each decision
            
        Returns:
        --------
        dict
            Dictionary containing accuracy metrics
        """
        # Filter decisions where trading was allowed
        trading_decisions = [d for d in self.decisions if d.get('should_trade', False)]
        
        if len(trading_decisions) != len(ground_truth_regimes):
            raise ValueError("Length of trading decisions and ground truth regimes must match")
        
        correct_classifications = 0
        confusion_matrix = np.zeros((3, 3))  # 3x3 for BULLISH, BEARISH, RANGE
        
        for decision, true_regime in zip(trading_decisions, ground_truth_regimes):
            if 'regime' not in decision:
                continue
                
            predicted_regime = MarketRegime[decision['regime']]
            
            # Update confusion matrix (excluding UNKNOWN)
            if predicted_regime != MarketRegime.UNKNOWN and true_regime != MarketRegime.UNKNOWN:
                confusion_matrix[predicted_regime.value, true_regime.value] += 1
            
            if predicted_regime == true_regime:
                correct_classifications += 1
        
        accuracy = correct_classifications / len(trading_decisions) if len(trading_decisions) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix
        }
        
        self.evaluation_results['meta_selector'].update(metrics)
        return metrics
    
    def generate_performance_report(self, output_path=None, include_plots=True):
        """
        Generate a comprehensive performance report.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report
        include_plots : bool
            Whether to include visualizations in the report
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the performance metrics
        """
        # Ensure evaluation has been run
        if not self.evaluation_results['overall']:
            raise ValueError("Must run evaluate() before generating report")
        
        # Create a report summary
        report = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Overall': {
                'Total Return': self.evaluation_results['overall']['total_return'],
                'Sharpe Ratio': self.evaluation_results['overall']['sharpe_ratio'],
                'Max Drawdown': self.evaluation_results['overall']['max_drawdown'],
                'Total Trades Executed': self.evaluation_results['overall']['trades_executed'],
                'Total Trades Skipped': self.evaluation_results['overall']['trades_skipped']
            },
            'No-Trade Filter': {
                'Accuracy': self.evaluation_results['no_trade_filter'].get('accuracy', 'N/A'),
                'Precision': self.evaluation_results['no_trade_filter'].get('precision', 'N/A'),
                'Recall': self.evaluation_results['no_trade_filter'].get('recall', 'N/A'),
                'F1 Score': self.evaluation_results['no_trade_filter'].get('f1_score', 'N/A')
            },
            'Market Regime Classification': {
                'Accuracy': self.evaluation_results['meta_selector'].get('accuracy', 'N/A'),
                'Bullish %': self.evaluation_results['meta_selector'].get('bullish_percentage', 'N/A'),
                'Bearish %': self.evaluation_results['meta_selector'].get('bearish_percentage', 'N/A'),
                'Range %': self.evaluation_results['meta_selector'].get('range_percentage', 'N/A')
            },
            'Specialized Agents': {
                'Bullish': {
                    'Return': self.evaluation_results['specialized_agents']['bullish']['total_return'],
                    'Win Rate': self.evaluation_results['specialized_agents']['bullish'].get('win_rate', 'N/A'),
                    'Trades': self.evaluation_results['specialized_agents']['bullish']['trades']
                },
                'Bearish': {
                    'Return': self.evaluation_results['specialized_agents']['bearish']['total_return'],
                    'Win Rate': self.evaluation_results['specialized_agents']['bearish'].get('win_rate', 'N/A'),
                    'Trades': self.evaluation_results['specialized_agents']['bearish']['trades']
                },
                'Range': {
                    'Return': self.evaluation_results['specialized_agents']['range']['total_return'],
                    'Win Rate': self.evaluation_results['specialized_agents']['range'].get('win_rate', 'N/A'),
                    'Trades': self.evaluation_results['specialized_agents']['range']['trades']
                }
            }
        }
        
        # Create a flat report for DataFrame conversion
        flat_report = {}
        for section, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            flat_report[f"{section} - {metric} - {sub_metric}"] = sub_value
                    else:
                        flat_report[f"{section} - {metric}"] = value
            else:
                flat_report[section] = metrics
        
        # Convert to DataFrame
        report_df = pd.DataFrame([flat_report])
        
        # Save to CSV if output path provided
        if output_path:
            report_df.to_csv(f"{output_path}/performance_report.csv", index=False)
            
            # Generate plots if requested
            if include_plots:
                self.generate_visualizations(output_path)
        
        return report_df
    
    def generate_visualizations(self, output_path):
        """
        Generate visualizations of the evaluation results.
        
        Parameters:
        -----------
        output_path : str
            Path to save the visualizations
        """
        # Ensure directory exists
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Plot 1: Cumulative returns
        if self.evaluation_results['overall']['episode_returns']:
            plt.figure(figsize=(12, 6))
            cumulative_returns = np.cumsum(self.evaluation_results['overall']['episode_returns'])
            plt.plot(cumulative_returns)
            plt.title('Cumulative Returns')
            plt.xlabel('Episode')
            plt.ylabel('Return')
            plt.grid(True)
            plt.savefig(f"{output_path}/cumulative_returns.png")
            plt.close()
        
        # Plot 2: Agent Performance Comparison
        agent_returns = [
            self.evaluation_results['specialized_agents']['bullish']['total_return'],
            self.evaluation_results['specialized_agents']['bearish']['total_return'],
            self.evaluation_results['specialized_agents']['range']['total_return']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(['Bullish', 'Bearish', 'Range'], agent_returns)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            bar.set_color('green' if agent_returns[i] > 0 else 'red')
            
        plt.title('Specialized Agent Performance')
        plt.ylabel('Total Return')
        plt.grid(axis='y')
        plt.savefig(f"{output_path}/agent_performance.png")
        plt.close()
        
        # Plot 3: Market Regime Distribution
        if hasattr(self.evaluation_results['meta_selector'], 'get') and all(
                self.evaluation_results['meta_selector'].get(f'{regime.name.lower()}_percentage', None) is not None
                for regime in [MarketRegime.BULLISH, MarketRegime.BEARISH, MarketRegime.RANGE]):
            
            regime_percentages = [
                self.evaluation_results['meta_selector'].get('bullish_percentage', 0),
                self.evaluation_results['meta_selector'].get('bearish_percentage', 0),
                self.evaluation_results['meta_selector'].get('range_percentage', 0)
            ]
            
            plt.figure(figsize=(10, 6))
            plt.pie(regime_percentages, labels=['Bullish', 'Bearish', 'Range'], 
                    autopct='%1.1f%%', startangle=90)
            plt.title('Market Regime Distribution')
            plt.savefig(f"{output_path}/regime_distribution.png")
            plt.close()
        
        # Plot 4: No-Trade Filter Confusion Matrix
        if all(metric in self.evaluation_results['no_trade_filter'] 
                for metric in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']):
            
            confusion_data = np.array([
                [self.evaluation_results['no_trade_filter']['true_positives'], 
                 self.evaluation_results['no_trade_filter']['false_positives']],
                [self.evaluation_results['no_trade_filter']['false_negatives'], 
                 self.evaluation_results['no_trade_filter']['true_negatives']]
            ])
            
            plt.figure(figsize=(8, 6))
            plt.imshow(confusion_data, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('No-Trade Filter Confusion Matrix')
            plt.colorbar()
            
            classes = ['Trade', 'No Trade']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            threshold = confusion_data.max() / 2
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, format(confusion_data[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if confusion_data[i, j] > threshold else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(f"{output_path}/no_trade_filter_confusion.png")
            plt.close()
        
        # Plot 5: Meta-Agent Confusion Matrix if available
        if 'confusion_matrix' in self.evaluation_results['meta_selector']:
            cm = self.evaluation_results['meta_selector']['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Market Regime Classification Confusion Matrix')
            plt.colorbar()
            
            classes = ['Bullish', 'Bearish', 'Range']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            threshold = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > threshold else "black")
            
            plt.ylabel('True regime')
            plt.xlabel('Predicted regime')
            plt.tight_layout()
            plt.savefig(f"{output_path}/regime_confusion_matrix.png")
            plt.close()


class BacktestingEngine:
    """
    Engine for backtesting the hierarchical agent system on historical data.
    
    This class provides methods to:
    1. Run the hierarchical agent system on historical market data
    2. Track and analyze performance metrics
    3. Visualize trading decisions and performance
    """
    
    def __init__(self, evaluator: HierarchicalEvaluator):
        """
        Initialize the BacktestingEngine.
        
        Parameters:
        -----------
        evaluator : HierarchicalEvaluator
            The hierarchical evaluator instance
        """
        self.evaluator = evaluator
        self.backtest_results = None
        self.trade_history = []
        self.equity_curve = []
    
    def run_backtest(self, historical_data, initial_capital=10000.0, 
                     transaction_cost=0.001, risk_free_rate=0.02):
        """
        Run a backtest of the hierarchical agent system on historical data.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame containing historical market data
        initial_capital : float
            Initial capital for the backtest
        transaction_cost : float
            Transaction cost as a percentage of trade value
        risk_free_rate : float
            Annualized risk-free rate for calculating risk-adjusted metrics
            
        Returns:
        --------
        dict
            Dictionary containing comprehensive backtest results
        """
        # Initialize backtest
        capital = initial_capital
        position = 0
        self.trade_history = []
        self.equity_curve = [capital]
        
        # Prepare result structure
        self.backtest_results = {
            'initial_capital': initial_capital,
            'final_capital': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'trades': 0,
            'regime_performance': {
                'bullish': {'return': 0.0, 'trades': 0},
                'bearish': {'return': 0.0, 'trades': 0},
                'range': {'return': 0.0, 'trades': 0}
            },
            'monthly_returns': []
        }
        
        # Run backtest
        for i in range(len(historical_data) - 1):
            current_data = historical_data.iloc[i]
            next_data = historical_data.iloc[i + 1]
            
            # Prepare observation (this would depend on your specific data format)
            observation = self._prepare_observation(current_data)
            
            # Step 1: No-Trade Filter decision
            should_trade, ntf_confidence = self.evaluator.no_trade_filter.predict(observation)
            should_trade = bool(should_trade)
            
            # Record equity at this step
            self.equity_curve.append(capital + position * current_data['close'])
            
            # Step 2: If should trade, identify market regime and execute strategy
            if should_trade:
                regime, regime_confidence = self.evaluator.meta_selector.predict(observation)
                
                # Select appropriate agent based on regime
                if regime == MarketRegime.BULLISH:
                    agent = self.evaluator.bullish_agent
                    regime_key = 'bullish'
                elif regime == MarketRegime.BEARISH:
                    agent = self.evaluator.bearish_agent
                    regime_key = 'bearish'
                elif regime == MarketRegime.RANGE:
                    agent = self.evaluator.range_agent
                    regime_key = 'range'
                else:
                    # Unknown regime, skip trading
                    continue
                
                # Get action from specialized agent
                action, _ = agent.predict(observation)
                
                # Execute action (assuming action space: 0=Buy, 1=Sell, 2=Hold)
                if action == 0 and position <= 0:  # Buy
                    if position < 0:  # Close short first
                        profit = position * (current_data['close'] - current_data['open'])
                        capital += profit - abs(position * current_data['close'] * transaction_cost)
                        
                        # Record closing trade
                        self._record_trade('close_short', i, current_data, profit, regime_key)
                    
                    # Open long position
                    position_size = self._calculate_position_size(capital, agent, observation)
                    position = position_size / current_data['close']
                    capital -= position * current_data['close'] * (1 + transaction_cost)
                    
                    # Record opening trade
                    self._record_trade('open_long', i, current_data, 0, regime_key)
                    
                elif action == 1 and position >= 0:  # Sell
                    if position > 0:  # Close long first
                        profit = position * (current_data['close'] - current_data['open'])
                        capital += position * current_data['close'] - (position * current_data['close'] * transaction_cost)
                        
                        # Record closing trade
                        self._record_trade('close_long', i, current_data, profit, regime_key)
                    
                    # Open short position
                    position_size = self._calculate_position_size(capital, agent, observation)
                    position = -position_size / current_data['close']
                    capital += abs(position * current_data['close']) - (abs(position * current_data['close']) * transaction_cost)
                    
                    # Record opening trade
                    self._record_trade('open_short', i, current_data, 0, regime_key)
                
                elif action == 2:  # Hold
                    pass  # Do nothing
            
            # Calculate daily return (for metrics)
            if i % 20 == 0:  # Approximate monthly return (assuming 20 trading days per month)
                monthly_return = (self.equity_curve[-1] / self.equity_curve[-20] - 1) if i >= 20 else 0
                self.backtest_results['monthly_returns'].append(monthly_return)
        
        # Close any open position at the end of backtest
        final_data = historical_data.iloc[-1]
        if position != 0:
            if position > 0:  # Long position
                profit = position * (final_data['close'] - final_data['open'])
                capital += position * final_data['close'] - (position * final_data['close'] * transaction_cost)
                self._record_trade('close_long', len(historical_data)-1, final_data, profit, 'end_of_backtest')
            else:  # Short position
                profit = abs(position) * (final_data['open'] - final_data['close'])
                capital += abs(position * final_data['close']) - (abs(position * final_data['close']) * transaction_cost)
                self._record_trade('close_short', len(historical_data)-1, final_data, profit, 'end_of_backtest')
        
        # Final equity value
        final_equity = capital
        self.equity_curve.append(final_equity)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(initial_capital, final_equity, risk_free_rate)
        
        return self.backtest_results
    
    def _prepare_observation(self, data_row):
        """
        Prepare observation vector from a data row for agent input.
        This should be customized based on the expected observation space of your agents.
        
        Parameters:
        -----------
        data_row : pd.Series
            A row from the historical data DataFrame
            
        Returns:
        --------
        np.ndarray
            Observation vector for agent input
        """
        # This is a placeholder implementation
        # In a real system, this would extract and normalize relevant features
        
        # Example: extract basic price and volume features
        features = [
            data_row.get('open', 0),
            data_row.get('high', 0),
            data_row.get('low', 0),
            data_row.get('close', 0),
            data_row.get('volume', 0)
        ]
        
        # Add technical indicators if available
        for indicator in ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal']:
            if indicator in data_row:
                features.append(data_row[indicator])
        
        # Convert to numpy array and normalize if needed
        obs = np.array(features, dtype=np.float32)
        
        return obs
    
    def _calculate_position_size(self, capital, agent, observation):
        """
        Calculate appropriate position size based on agent and capital.
        
        Parameters:
        -----------
        capital : float
            Current capital
        agent : BaseAgent
            The agent making the trading decision
        observation : np.ndarray
            Current market observation
            
        Returns:
        --------
        float
            Position size in currency units
        """
        # Get confidence level for the action
        confidence = agent.get_action_confidence(observation)
        
        # Use agent's position sizing logic if available
        if hasattr(agent, 'calculate_position_size'):
            position_pct = agent.calculate_position_size(confidence, capital)
        else:
            # Default position sizing: more confident = larger position
            position_pct = 0.02 + (confidence * 0.08)  # Between 2% and 10% of capital
        
        # Ensure position size is reasonable
        position_pct = min(max(position_pct, 0.01), 0.1)  # Between 1% and 10%
        
        return capital * position_pct
    
    def _record_trade(self, trade_type, index, data, profit, regime):
        """
        Record a trade for analysis.
        
        Parameters:
        -----------
        trade_type : str
            Type of trade ('open_long', 'close_long', 'open_short', 'close_short')
        index : int
            Index in the historical data
        data : pd.Series
            Data row where trade occurred
        profit : float
            Profit/loss from the trade (for closing trades)
        regime : str
            Market regime during the trade
        """
        trade = {
            'index': index,
            'timestamp': data.name if hasattr(data, 'name') else index,
            'type': trade_type,
            'price': data['close'],
            'profit': profit,
            'regime': regime
        }
        
        self.trade_history.append(trade)
        
        # Update regime-specific metrics for closing trades
        if trade_type in ['close_long', 'close_short'] and regime in self.backtest_results['regime_performance']:
            self.backtest_results['regime_performance'][regime]['return'] += profit
            self.backtest_results['regime_performance'][regime]['trades'] += 1
    
    def _calculate_performance_metrics(self, initial_capital, final_equity, risk_free_rate):
        """
        Calculate comprehensive performance metrics from backtest results.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital
        final_equity : float
            Final equity
        risk_free_rate : float
            Annualized risk-free rate
        """
        # Basic return metrics
        total_return = (final_equity / initial_capital) - 1
        trading_days = len(self.equity_curve) - 1
        annual_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
        
        # Calculate daily returns
        daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            sharpe_ratio = (np.mean(daily_returns) - daily_risk_free) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_returns = np.array(self.equity_curve) / initial_capital
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        closing_trades = [t for t in self.trade_history if t['type'] in ['close_long', 'close_short']]
        winning_trades = sum(1 for t in closing_trades if t['profit'] > 0)
        total_trades = len(closing_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Update results
        self.backtest_results.update({
            'final_capital': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': total_trades
        })
    
    def plot_equity_curve(self, title="Backtest Equity Curve", save_path=None):
        """
        Plot the equity curve from the backtest.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        save_path : str, optional
            Path to save the plot, if provided
        """
        if not self.equity_curve:
            raise ValueError("No backtest results available")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title(title)
        plt.xlabel('Trading Days')
        plt.ylabel('Equity')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_monthly_returns(self, title="Monthly Returns", save_path=None):
        """
        Plot monthly returns from the backtest.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        save_path : str, optional
            Path to save the plot, if provided
        """
        if not self.backtest_results or not self.backtest_results['monthly_returns']:
            raise ValueError("No backtest results available")
        
        monthly_returns = self.backtest_results['monthly_returns']
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(monthly_returns)), monthly_returns)
        
        # Color positive returns green, negative red
        for i, bar in enumerate(bars):
            bar.set_color('green' if monthly_returns[i] >= 0 else 'red')
        
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Return')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_trade_distribution(self, by_regime=True, title="Trade Distribution", save_path=None):
        """
        Plot trade distribution by outcome and optionally by market regime.
        
        Parameters:
        -----------
        by_regime : bool
            Whether to break down trades by market regime
        title : str
            Title for the plot
        save_path : str, optional
            Path to save the plot, if provided
        """
        if not self.trade_history:
            raise ValueError("No trade history available")
        
        # Filter closing trades only
        closing_trades = [t for t in self.trade_history if t['type'] in ['close_long', 'close_short']]
        
        if by_regime:
            # Group trades by regime and outcome
            regime_data = {}
            
            for regime in ['bullish', 'bearish', 'range']:
                regime_trades = [t for t in closing_trades if t['regime'] == regime]
                if not regime_trades:
                    continue
                    
                winning = sum(1 for t in regime_trades if t['profit'] > 0)
                losing = len(regime_trades) - winning
                
                regime_data[regime] = (winning, losing)
            
            # Create grouped bar chart
            regimes = list(regime_data.keys())
            winning_counts = [regime_data[r][0] for r in regimes]
            losing_counts = [regime_data[r][1] for r in regimes]
            
            x = np.arange(len(regimes))
            width = 0.35
            
            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, winning_counts, width, label='Winning', color='green')
            plt.bar(x + width/2, losing_counts, width, label='Losing', color='red')
            
            plt.xlabel('Market Regime')
            plt.ylabel('Number of Trades')
            plt.title(title)
            plt.xticks(x, regimes)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
        else:
            # Simple winning vs losing trades
            winning = sum(1 for t in closing_trades if t['profit'] > 0)
            losing = len(closing_trades) - winning
            
            plt.figure(figsize=(10, 6))
            plt.bar(['Winning', 'Losing'], [winning, losing], color=['green', 'red'])
            plt.title(title)
            plt.ylabel('Number of Trades')
            plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def generate_backtest_report(self, output_path=None):
        """
        Generate a comprehensive backtest report.
        
        Parameters:
        -----------
        output_path : str, optional
            Directory path to save report files
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing key backtest metrics
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available")
        
        # Create report DataFrame
        report_data = {
            'Metric': [
                'Initial Capital',
                'Final Capital',
                'Total Return (%)',
                'Annualized Return (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
                'Win Rate (%)',
                'Total Trades',
                'Bullish Regime Return (%)',
                'Bearish Regime Return (%)',
                'Range Regime Return (%)'
            ],
            'Value': [
                self.backtest_results['initial_capital'],
                self.backtest_results['final_capital'],
                self.backtest_results['total_return'] * 100,
                self.backtest_results['annual_return'] * 100,
                self.backtest_results['sharpe_ratio'],
                self.backtest_results['max_drawdown'] * 100,
                self.backtest_results['win_rate'] * 100,
                self.backtest_results['trades'],
                self.backtest_results['regime_performance']['bullish']['return'] * 100 / self.backtest_results['initial_capital'],
                self.backtest_results['regime_performance']['bearish']['return'] * 100 / self.backtest_results['initial_capital'],
                self.backtest_results['regime_performance']['range']['return'] * 100 / self.backtest_results['initial_capital']
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        
        # Save report and plots if output path provided
        if output_path:
            import os
            os.makedirs(output_path, exist_ok=True)
            
            # Save report to CSV
            report_df.to_csv(f"{output_path}/backtest_report.csv", index=False)
            
            # Generate and save plots
            self.plot_equity_curve(save_path=f"{output_path}/equity_curve.png")
            self.plot_monthly_returns(save_path=f"{output_path}/monthly_returns.png")
            self.plot_trade_distribution(by_regime=True, save_path=f"{output_path}/trade_distribution.png")
            
            # Save trade history
            trade_df = pd.DataFrame(self.trade_history)
            trade_df.to_csv(f"{output_path}/trade_history.csv", index=False)
        
        return report_df