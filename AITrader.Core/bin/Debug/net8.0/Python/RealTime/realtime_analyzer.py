"""
Real-Time Market Analyzer Module

This module implements the real-time market analyzer that processes live market data
from NinjaTrader 8 and makes trading decisions using the hierarchical agent architecture.
"""

import os
import time
import threading
import datetime
import logging
import numpy as np
import pandas as pd
from queue import Queue
from typing import Dict, List, Tuple, Optional, Union, Callable

# Import socket client for NinjaTrader communication
from AITrader.Core.Python.RealTime.socket_client import NinjaTraderSocketClient

# Import hierarchical agent architecture
from AITrader.Core.Python.Agents.no_trade_filter_agent import NoTradeFilterAgent
from AITrader.Core.Python.Agents.meta_agent_selector import MetaAgentSelector, MarketRegime
from AITrader.Core.Python.Agents.bullish_agent import BullishAgent
from AITrader.Core.Python.Agents.bearish_agent import BearishAgent
from AITrader.Core.Python.Agents.range_agent import RangeAgent

# Import data preparation
from AITrader.Core.Python.Agents.data_preparation import DataPreparationPipeline, FeatureSet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealTimeMarketAnalyzer:
    """
    Real-time market analyzer that processes live market data and makes trading decisions
    using the hierarchical agent architecture.
    """
    
    def __init__(self, 
                 models_dir: str,
                 data_host: str = "127.0.0.1", 
                 data_port: int = 5000,
                 order_host: str = "127.0.0.1", 
                 order_port: int = 5001):
        """
        Initialize the real-time market analyzer.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing the trained agent models
        data_host : str
            Host for the data socket connection
        data_port : int
            Port for the data socket connection
        order_host : str
            Host for the order socket connection
        order_port : int
            Port for the order socket connection
        """
        # Socket client for NinjaTrader communication
        self.socket_client = NinjaTraderSocketClient(
            data_host=data_host,
            data_port=data_port,
            order_host=order_host,
            order_port=order_port
        )
        
        # Models directory
        self.models_dir = models_dir
        
        # Data preparation pipeline
        self.data_pipeline = DataPreparationPipeline()
        
        # Load normalization parameters
        norm_params_path = os.path.join(models_dir, "normalization_params.pkl")
        if os.path.exists(norm_params_path):
            self.data_pipeline.load_normalization_params(norm_params_path)
        else:
            logger.warning(f"Normalization parameters not found at {norm_params_path}")
        
        # Observation spaces
        self.observation_spaces = self.data_pipeline.create_observation_spaces()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Historical data buffer
        self.market_data_buffer = []
        self.max_buffer_size = 200  # Store up to 200 data points
        
        # Current market state
        self.current_position = 0  # 0 = flat, 1 = long, -1 = short
        self.last_signal_time = None
        self.signal_cooldown = 60  # Seconds between signals
        
        # Analysis thread
        self.analysis_thread = None
        self.running = False
        self.analysis_interval = 1.0  # seconds between analyses
        self.last_analysis_time = 0
        
        # Trading parameters
        self.trading_enabled = True
        self.position_sizing = 1.0
        self.stop_loss_ticks = 10
        self.take_profit_ticks = 20
        
        # Register callbacks
        self.socket_client.register_data_callback(self._on_market_data_received)
        self.socket_client.register_order_callback(self._on_order_update_received)
        self.socket_client.register_connection_callback(self._on_connection_status_changed)
    
    def start(self) -> bool:
        """
        Start the real-time market analyzer.
        
        Returns:
        --------
        bool
            True if started successfully, False otherwise
        """
        # Start socket client
        if not self.socket_client.start():
            logger.error("Failed to start socket client")
            return False
        
        # Start analysis thread
        self.running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        logger.info("Real-time market analyzer started")
        return True
    
    def stop(self) -> None:
        """
        Stop the real-time market analyzer.
        """
        self.running = False
        
        # Stop socket client
        self.socket_client.stop()
        
        # Wait for analysis thread to terminate
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(2)
        
        logger.info("Real-time market analyzer stopped")
    
    def set_trading_parameters(self, enabled: bool = True, position_sizing: float = 1.0,
                              stop_loss_ticks: int = 10, take_profit_ticks: int = 20) -> None:
        """
        Set trading parameters.
        
        Parameters:
        -----------
        enabled : bool
            Enable/disable trading
        position_sizing : float
            Position sizing multiplier
        stop_loss_ticks : int
            Stop loss in ticks
        take_profit_ticks : int
            Take profit in ticks
        """
        self.trading_enabled = enabled
        self.position_sizing = position_sizing
        self.stop_loss_ticks = stop_loss_ticks
        self.take_profit_ticks = take_profit_ticks
        logger.info(f"Trading parameters updated: enabled={enabled}, position_sizing={position_sizing}, "
                   f"stop_loss={stop_loss_ticks}, take_profit={take_profit_ticks}")
    
    def get_current_market_state(self) -> Dict:
        """
        Get the current market state.
        
        Returns:
        --------
        Dict
            Current market state
        """
        if not self.market_data_buffer:
            return {"status": "No data received"}
        
        # Get the latest market data
        latest_data = self.market_data_buffer[-1]
        
        # Get connection status
        data_connected, order_connected = self.socket_client.is_connected()
        
        return {
            "data_connected": data_connected,
            "order_connected": order_connected,
            "timestamp": latest_data.get("timestamp", "Unknown"),
            "last_price": latest_data.get("close", 0.0),
            "current_position": self.current_position,
            "trading_enabled": self.trading_enabled,
            "data_points": len(self.market_data_buffer)
        }
    
    def get_historical_data(self) -> pd.DataFrame:
        """
        Get the historical market data buffer.
        
        Returns:
        --------
        pd.DataFrame
            Historical market data
        """
        if not self.market_data_buffer:
            return pd.DataFrame()
        
        return pd.DataFrame(self.market_data_buffer)
    
    def _initialize_agents(self) -> Dict:
        """
        Initialize the hierarchical agents.
        
        Returns:
        --------
        Dict
            Dictionary of initialized agents
        """
        # Check if observation spaces are available
        if not self.observation_spaces:
            logger.error("Observation spaces not available")
            raise ValueError("Observation spaces not available, data pipeline not properly initialized")
        
        # No-Trade Filter Agent
        no_trade_filter = NoTradeFilterAgent(
            observation_space=self.observation_spaces[FeatureSet.NO_TRADE],
            action_space=gym.spaces.Discrete(2),  # 0 = No-Trade, 1 = Trade
            model_path=os.path.join(self.models_dir, "no_trade_filter.zip")
        )
        
        # Meta-Agent Selector
        meta_selector = MetaAgentSelector(
            observation_space=self.observation_spaces[FeatureSet.META],
            action_space=gym.spaces.Discrete(3),  # 0 = Bullish, 1 = Bearish, 2 = Range
            model_path=os.path.join(self.models_dir, "meta_selector.zip")
        )
        
        # Specialized Agents
        bullish_agent = BullishAgent(
            observation_space=self.observation_spaces[FeatureSet.BULLISH],
            action_space=gym.spaces.Discrete(3),  # 0 = Buy, 1 = Sell, 2 = Hold
            model_path=os.path.join(self.models_dir, "bullish_agent.zip")
        )
        
        bearish_agent = BearishAgent(
            observation_space=self.observation_spaces[FeatureSet.BEARISH],
            action_space=gym.spaces.Discrete(3),  # 0 = Buy, 1 = Sell, 2 = Hold
            model_path=os.path.join(self.models_dir, "bearish_agent.zip")
        )
        
        range_agent = RangeAgent(
            observation_space=self.observation_spaces[FeatureSet.RANGE],
            action_space=gym.spaces.Discrete(3),  # 0 = Buy, 1 = Sell, 2 = Hold
            model_path=os.path.join(self.models_dir, "range_agent.zip")
        )
        
        # Register specialized agents with meta-selector
        meta_selector.register_specialized_agent(MarketRegime.BULLISH, bullish_agent)
        meta_selector.register_specialized_agent(MarketRegime.BEARISH, bearish_agent)
        meta_selector.register_specialized_agent(MarketRegime.RANGE, range_agent)
        
        return {
            "no_trade_filter": no_trade_filter,
            "meta_selector": meta_selector,
            "bullish_agent": bullish_agent,
            "bearish_agent": bearish_agent,
            "range_agent": range_agent
        }
    
    def _analysis_loop(self) -> None:
        """
        Main loop for market analysis and trading decisions.
        """
        while self.running:
            current_time = time.time()
            
            # Check if enough time has passed since last analysis
            if current_time - self.last_analysis_time < self.analysis_interval:
                time.sleep(0.1)  # Sleep briefly to prevent high CPU usage
                continue
            
            # Update last analysis time
            self.last_analysis_time = current_time
            
            # Check if we have enough data
            if len(self.market_data_buffer) < 20:
                continue
            
            # Check if trading is enabled
            if not self.trading_enabled:
                continue
            
            # Check connection status
            data_connected, order_connected = self.socket_client.is_connected()
            if not data_connected or not order_connected:
                logger.warning("Cannot analyze - not fully connected")
                continue
            
            try:
                # Process market data and make trading decision
                self._process_market_data()
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    def _process_market_data(self) -> None:
        """
        Process market data and make trading decisions.
        """
        # Convert market data buffer to DataFrame
        df = pd.DataFrame(self.market_data_buffer)
        
        # Prepare data for each agent type
        try:
            no_trade_data = self.data_pipeline.prepare_data(df, feature_set=FeatureSet.NO_TRADE)
            meta_data = self.data_pipeline.prepare_data(df, feature_set=FeatureSet.META)
            bullish_data = self.data_pipeline.prepare_data(df, feature_set=FeatureSet.BULLISH)
            bearish_data = self.data_pipeline.prepare_data(df, feature_set=FeatureSet.BEARISH)
            range_data = self.data_pipeline.prepare_data(df, feature_set=FeatureSet.RANGE)
            
            # Normalize data
            no_trade_features = self.data_pipeline.normalize_data(no_trade_data, FeatureSet.NO_TRADE)
            meta_features = self.data_pipeline.normalize_data(meta_data, FeatureSet.META)
            bullish_features = self.data_pipeline.normalize_data(bullish_data, FeatureSet.BULLISH)
            bearish_features = self.data_pipeline.normalize_data(bearish_data, FeatureSet.BEARISH)
            range_features = self.data_pipeline.normalize_data(range_data, FeatureSet.RANGE)
            
            # Get the latest observation for each agent
            no_trade_obs = self.data_pipeline.create_observation_vector(no_trade_features.iloc[-1], FeatureSet.NO_TRADE)
            meta_obs = self.data_pipeline.create_observation_vector(meta_features.iloc[-1], FeatureSet.META)
            bullish_obs = self.data_pipeline.create_observation_vector(bullish_features.iloc[-1], FeatureSet.BULLISH)
            bearish_obs = self.data_pipeline.create_observation_vector(bearish_features.iloc[-1], FeatureSet.BEARISH)
            range_obs = self.data_pipeline.create_observation_vector(range_features.iloc[-1], FeatureSet.RANGE)
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return
        
        # Step 1: No-Trade Filter decision
        should_trade, ntf_confidence = self.agents["no_trade_filter"].predict(no_trade_obs)
        should_trade = bool(should_trade)
        
        if not should_trade:
            logger.info(f"No-Trade filter decided not to trade (confidence: {ntf_confidence:.2f})")
            return
        
        # Step 2: Meta-Agent regime classification
        regime, regime_confidences = self.agents["meta_selector"].predict(meta_obs)
        regime_confidence = regime_confidences[regime]
        
        logger.info(f"Market regime: {regime.name} (confidence: {regime_confidence:.2f})")
        
        # Step 3: Select specialized agent based on regime
        if regime == MarketRegime.BULLISH:
            agent = self.agents["bullish_agent"]
            agent_obs = bullish_obs
            agent_name = "bullish"
        elif regime == MarketRegime.BEARISH:
            agent = self.agents["bearish_agent"]
            agent_obs = bearish_obs
            agent_name = "bearish"
        elif regime == MarketRegime.RANGE:
            agent = self.agents["range_agent"]
            agent_obs = range_obs
            agent_name = "range"
        else:
            logger.warning(f"Unknown market regime: {regime}")
            return
        
        # Step 4: Get trading action from specialized agent
        action, action_info = agent.predict(agent_obs)
        action_confidence = action_info["action_probs"][action]
        
        logger.info(f"{agent_name} agent action: {action} (confidence: {action_confidence:.2f})")
        
        # Step 5: Determine trading signal
        trading_signal = self._determine_trading_signal(action, agent_name, self.current_position)
        
        # Step 6: Execute trading signal
        if trading_signal != 0 or self.current_position != 0:  # 0 = no change
            self._execute_trading_signal(trading_signal)
    
    def _determine_trading_signal(self, action: int, agent_name: str, current_position: int) -> int:
        """
        Determine trading signal based on agent action and current position.
        
        Parameters:
        -----------
        action : int
            Agent action (0 = Buy, 1 = Sell, 2 = Hold)
        agent_name : str
            Name of the agent making the decision
        current_position : int
            Current position (0 = flat, 1 = long, -1 = short)
            
        Returns:
        --------
        int
            Trading signal (-1 = short, 0 = flat/no change, 1 = long)
        """
        # Check signal cooldown
        if self.last_signal_time is not None and time.time() - self.last_signal_time < self.signal_cooldown:
            return 0  # No change during cooldown
        
        # Map action to signal
        if action == 0:  # Buy
            # If bullish regime and we're not already long, go long
            # If range regime and we're short, go flat
            if agent_name == "bullish" or agent_name == "range":
                return 1 if current_position <= 0 else 0
            # If bearish regime, generally don't buy but can close shorts
            else:
                return 0 if current_position < 0 else 0
                
        elif action == 1:  # Sell
            # If bearish regime and we're not already short, go short
            # If range regime and we're long, go flat
            if agent_name == "bearish" or agent_name == "range":
                return -1 if current_position >= 0 else 0
            # If bullish regime, generally don't sell but can close longs
            else:
                return 0 if current_position > 0 else 0
                
        else:  # Hold - no change
            return 0
    
    def _execute_trading_signal(self, signal: int) -> None:
        """
        Execute a trading signal.
        
        Parameters:
        -----------
        signal : int
            Trading signal (-1 = short, 0 = flat, 1 = long)
        """
        # If signal is the same as current position, no need to act
        if signal == self.current_position:
            return
        
        # If signal is 0 (flat) and we're already flat, no need to act
        if signal == 0 and self.current_position == 0:
            return
        
        # Apply position sizing
        position_size = self.position_sizing
        
        # Set EMA choice (simplified for now)
        ema_choice = 0  # No EMA
        
        # Set stop loss and take profit
        stop_loss = self.stop_loss_ticks
        take_profit = self.take_profit_ticks
        
        # Send trading signal
        success = self.socket_client.send_trading_signal(
            signal=signal,
            ema_choice=ema_choice,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if success:
            logger.info(f"Sent trading signal: {signal} with size={position_size}, SL={stop_loss}, TP={take_profit}")
            self.last_signal_time = time.time()
            self.current_position = signal
        else:
            logger.error("Failed to send trading signal")
    
    def _on_market_data_received(self, data: Dict) -> None:
        """
        Handle received market data.
        
        Parameters:
        -----------
        data : Dict
            Market data
        """
        # Add to buffer
        self.market_data_buffer.append(data)
        
        # Trim buffer if too large
        if len(self.market_data_buffer) > self.max_buffer_size:
            self.market_data_buffer = self.market_data_buffer[-self.max_buffer_size:]
    
    def _on_order_update_received(self, data: Dict) -> None:
        """
        Handle received order updates.
        
        Parameters:
        -----------
        data : Dict
            Order update data
        """
        if data.get("type") == "order_confirmation":
            logger.info(f"Order confirmed: {data}")
        elif data.get("type") == "trade_execution":
            logger.info(f"Trade executed: {data}")
            
            # Update current position based on trade execution
            action = data.get("action", "")
            if action.startswith("Exit"):
                # Position closed
                self.current_position = 0
            elif action.startswith("Enter") or action.startswith("Open"):
                # New position opened
                if "Long" in action:
                    self.current_position = 1
                elif "Short" in action:
                    self.current_position = -1
    
    def _on_connection_status_changed(self, data_connected: bool, order_connected: bool) -> None:
        """
        Handle connection status changes.
        
        Parameters:
        -----------
        data_connected : bool
            Data socket connection status
        order_connected : bool
            Order socket connection status
        """
        logger.info(f"Connection status changed: data={data_connected}, order={order_connected}")
        
        # If disconnected, clear buffer
        if not data_connected:
            self.market_data_buffer = []


# Main function for testing
def main():
    # Directory containing the trained models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models")
    
    # Create real-time analyzer
    analyzer = RealTimeMarketAnalyzer(
        models_dir=models_dir,
        data_host="127.0.0.1",
        data_port=5000,
        order_host="127.0.0.1",
        order_port=5001
    )
    
    # Set trading parameters
    analyzer.set_trading_parameters(
        enabled=True,
        position_sizing=1.0,
        stop_loss_ticks=10,
        take_profit_ticks=20
    )
    
    # Start analyzer
    if analyzer.start():
        logger.info("Analyzer started successfully")
        
        try:
            # Run for 1 hour
            time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Stop analyzer
            analyzer.stop()
            logger.info("Analyzer stopped")
    else:
        logger.error("Failed to start analyzer")


if __name__ == "__main__":
    import gym  # Import gym here to avoid circular import
    import select  # Import select for socket operations
    main()