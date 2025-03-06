# 🤖 AITrader - Intelligent Trading with Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stable Baselines3](https://img.shields.io/badge/stable--baselines3-latest-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![Project Status: Active](https://img.shields.io/badge/project%20status-active-brightgreen.svg)](https://www.repostatus.org/#active)
<!-- Add more badges as needed, e.g., testing coverage, documentation status -->

<!-- Add project logo here (suggested size: 200x200px) -->
<!-- ![AITrader Logo](path/to/logo.png) -->

A sophisticated algorithmic trading system leveraging hierarchical reinforcement learning to adapt to changing market conditions and optimize trading decisions.

## 📑 Table of Contents

- [Overview](#-overview)
- [Why AITrader?](#-why-aitrader)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Usage Examples](#-usage-examples)
- [Roadmap](#-roadmap)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🔍 Overview

AITrader is an advanced trading system that uses reinforcement learning to make intelligent trading decisions based on dynamic market conditions. It implements a hierarchical agent architecture that combines specialized trading strategies optimized for different market regimes.

By leveraging the power of deep reinforcement learning through Stable Baselines3 and a custom-designed hierarchical decision-making process, AITrader can identify market regimes, adapt to changing conditions, and select the most appropriate trading strategy for the current environment.

## 🚀 Why AITrader?

Traditional algorithmic trading systems often excel in specific market conditions but struggle when market regimes change. AITrader addresses this fundamental challenge with:

- **Adaptive Strategy Selection** - Automatically identifies market conditions and applies the most suitable trading approach
- **Risk-Aware Trading** - Includes a No-Trade Filter to prevent trading in unfavorable conditions
- **Specialized Expertise** - Uses dedicated agents optimized for bullish, bearish, and range-bound markets
- **Continuous Learning** - Agents improve over time as they encounter new market scenarios
- **Hierarchical Architecture** - Modular design allows for easy extension and customization

## ✨ Features

### Core Components

- **No-Trade Filter Agent** 🛑
  - Top-level gatekeeper that determines if market conditions are suitable for trading
  - Analyzes volatility, liquidity, and risk factors to make a binary decision

- **Meta-Agent Selector** 🔍
  - Identifies the current market regime (Bullish, Bearish, or Range-bound)
  - Selects the most appropriate specialized agent for the detected regime

- **Specialized Trading Agents** 📈
  - **Bullish Agent**: Optimized for upward trending markets
  - **Bearish Agent**: Optimized for downward trending markets
  - **Range Agent**: Optimized for sideways or range-bound markets

### Additional Modules

- **Data Preparation Pipeline**: Tailored feature engineering for each agent type
- **Hierarchical Evaluation**: Comprehensive performance assessment tools
- **Custom Callbacks**: Training monitors with checkpoint management and early stopping
- **Backtesting Engine**: Detailed analysis of trading strategy performance

## 🏗️ Architecture

AITrader implements a three-level hierarchical agent architecture for optimal trading decisions:

<!-- Add architecture diagram here. Consider creating a diagram using tools like draw.io, mermaid, or lucidchart -->
<!-- ![Architecture Diagram](path/to/architecture_diagram.png) -->

```
                        ┌───────────────────┐
                        │   No-Trade Filter │
                        │       Agent       │
                        └────────┬──────────┘
                                 │
                                 │ (If trading conditions are favorable)
                                 ▼
                        ┌───────────────────┐
                        │   Meta-Agent      │
                        │    Selector       │
                        └────────┬──────────┘
                                 │
                  ┌──────────────┴───────────────┐
                  │                              │
┌─────────────────▼───┐    ┌──────────────▼───┐  │  ┌──────────────▼───┐
│   Bullish Agent     │    │  Bearish Agent   │  │  │   Range Agent     │
└─────────────────────┘    └──────────────────┘  └──└──────────────────┘
```

**Decision Process**:

1. The No-Trade Filter Agent evaluates market conditions
2. If conditions are favorable, the Meta-Agent Selector identifies the market regime
3. Based on the regime, the appropriate Specialized Agent executes the trading strategy

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Sufficient RAM for training RL models (8GB+ recommended)
- GPU acceleration recommended for faster training

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AITrader.git
   cd AITrader
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv env
   env\Scripts\activate

   # Linux/macOS
   python -m venv env
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is not available, install the core dependencies:
   ```bash
   pip install stable-baselines3 gym numpy pandas matplotlib tensorflow talib
   ```

4. Set up environment variables (if needed):
   ```bash
   # Example
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## 🏁 Getting Started

### Creating a Simple Trading Environment

```python
import gym
from gym import spaces
import numpy as np

class TradingEnvironment(gym.Env):
    """Simple trading environment for AITrader."""
    
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy (0), Sell (1), Hold (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
    
    def reset(self):
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action):
        # Execute action, calculate reward, etc.
        # ...
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        # Return the current market observation
        # ...
        return observation
```

### Training the No-Trade Filter Agent

```python
from gym import spaces
from AITrader.Core.Python.Agents.no_trade_filter_agent import NoTradeFilterAgent
from AITrader.Core.Python.Agents.callback_handlers import create_training_callbacks

# Define observation and action spaces
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
action_space = spaces.Discrete(2)  # Trade (1) or No-Trade (0)

# Initialize the agent
no_trade_filter = NoTradeFilterAgent(
    observation_space=observation_space,
    action_space=action_space,
    learning_rate=0.0003
)

# Create environment
env = TradingEnvironment(data)

# Set up callbacks
callbacks = create_training_callbacks(
    agent_name="no_trade_filter",
    log_dir="logs",
    save_dir="models",
    eval_env=env
)

# Train the agent
no_trade_filter.train(
    env=env,
    total_timesteps=100000,
    callback=callbacks
)

# Save the trained model
no_trade_filter.save("models/no_trade_filter.zip")
```

### Running the Complete Hierarchical System

```python
# Full integration code will be available once implementation is complete
# This is a conceptual example

from AITrader.Core.Python.Agents.hierarchical_evaluation import HierarchicalEvaluator
from AITrader.Core.Python.Agents.meta_agent_selector import MetaAgentSelector, MarketRegime
from AITrader.Core.Python.Agents.bullish_agent import BullishAgent
from AITrader.Core.Python.Agents.bearish_agent import BearishAgent
from AITrader.Core.Python.Agents.range_agent import RangeAgent

# Initialize all agents
# ...

# Create the hierarchical evaluator
evaluator = HierarchicalEvaluator(
    no_trade_filter=no_trade_filter,
    meta_selector=meta_selector,
    bullish_agent=bullish_agent,
    bearish_agent=bearish_agent,
    range_agent=range_agent
)

# Run evaluation
results = evaluator.evaluate(
    test_env=test_env,
    episodes=10,
    render=True
)

# Generate performance report
evaluator.generate_performance_report(
    output_path="reports",
    include_plots=True
)
```

## 📊 Usage Examples

### Data Preparation

```python
from AITrader.Core.Python.Agents.data_preparation import DataPreparationPipeline, FeatureSet
import pandas as pd

# Load historical market data
data = pd.read_csv("market_data.csv")

# Initialize data preparation pipeline
pipeline = DataPreparationPipeline(
    use_talib=True,
    window_sizes=(5, 10, 20, 50, 100),
    log_transform_volume=True
)

# Prepare data for each agent type
no_trade_data = pipeline.prepare_data(data, feature_set=FeatureSet.NO_TRADE)
meta_data = pipeline.prepare_data(data, feature_set=FeatureSet.META)
bullish_data = pipeline.prepare_data(data, feature_set=FeatureSet.BULLISH)

# Split data into training and testing sets
train_sets, test_sets = pipeline.prepare_training_data(
    data=data,
    train_test_split=0.8,
    shuffle=False
)

# Save normalization parameters for later use
pipeline.save_normalization_params("normalization_params.pkl")
```

### Backtesting

```python
from AITrader.Core.Python.Agents.hierarchical_evaluation import BacktestingEngine
import pandas as pd

# Load historical data
data = pd.read_csv("historical_data.csv")

# Initialize backtesting engine with the hierarchical evaluator
backtester = BacktestingEngine(evaluator)

# Run backtest
results = backtester.run_backtest(
    historical_data=data,
    initial_capital=10000.0,
    transaction_cost=0.001,
    risk_free_rate=0.02
)

# Generate backtest report
report = backtester.generate_backtest_report(output_path="backtest_results")

# Plot results
backtester.plot_equity_curve(title="AITrader Performance")
backtester.plot_monthly_returns()
backtester.plot_trade_distribution(by_regime=True)
```

## 🗺️ Roadmap

The following tasks are planned for future development:

- [ ] **Integration Testing**: Comprehensive testing of the complete hierarchical system
- [ ] **Hyperparameter Optimization**: Fine-tuning each agent's parameters for optimal performance
- [ ] **Extended Documentation**: Expanded usage examples and API documentation
- [ ] **Web Interface**: Development of a web-based dashboard for visualization and control
- [ ] **Alternative ML Algorithms**: Support for PPO, SAC, and other RL algorithms
- [ ] **Real-time Trading Connectivity**: Integration with broker APIs for live trading

## ❓ Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'talib'`  
**Solution**: TA-Lib requires special installation:
```bash
# For Windows
pip install --no-cache-dir TA-Lib-binary

# For Linux
apt-get install ta-lib
pip install TA-Lib

# For macOS
brew install ta-lib
pip install TA-Lib
```

**Issue**: `Memory Error` during training  
**Solution**: Reduce batch size or use smaller replay buffer:
```python
model = PPO("MlpPolicy", 
            env, 
            learning_rate=0.0003,
            batch_size=64,  # Reduced from default
            buffer_size=10000,  # Reduced from default
            verbose=1)
```

**Issue**: `GPU not detected` warning  
**Solution**: Ensure TensorFlow can access your GPU:
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**:
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

Please make sure your code follows the project's coding style and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

**Project Maintainer**: Your Name - [@yourgithub](https://github.com/yourgithub)

**Project Link**: [https://github.com/yourusername/AITrader](https://github.com/yourusername/AITrader)

---

<div align="center">
  
  **AITrader** — Making intelligent trading decisions with reinforcement learning
  
  <!-- If you have a small logo, you can add it here too -->
  
</div>