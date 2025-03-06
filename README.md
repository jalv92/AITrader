# AITrader - Reinforcement Learning Trading System

## Project Overview

AITrader is an advanced trading system that leverages reinforcement learning (RL) techniques to make intelligent trading decisions based on market conditions. The system implements a hierarchical agent architecture with specialized components:

1. **No-Trade Filter Agent**: A top-level agent that analyzes market conditions to determine if the current environment is suitable for trading. This serves as a gatekeeper to prevent trading in unfavorable or highly volatile market conditions.

2. **Meta-Agent Selector**: An intermediate agent that classifies the current market regime (Bullish, Bearish, or Range-bound) and selects the most appropriate specialized trading agent for that regime.

3. **Specialized Trading Agents**: A collection of agents optimized for specific market conditions:
   - **Bullish Agent**: Optimized for upward trending markets
   - **Bearish Agent**: Optimized for downward trending markets
   - **Range Agent**: Optimized for sideways or range-bound markets

This hierarchical approach allows the system to adapt to changing market conditions by employing specialized strategies tailored to each market regime, improving overall trading performance and risk management.

## Current Implementation Status

### Implemented Components

- **Base Agent (`base_agent.py`)**: The foundation class that provides common functionality for all specialized agents, including training, prediction, performance tracking, and model saving/loading capabilities.

- **Specialized Trading Agents**:
  - **Bullish Agent (`bullish_agent.py`)**: Agent optimized for bullish market conditions with specialized methods for trend strength calculation, support level identification, and breakout detection.
  - **Bearish Agent (`bearish_agent.py`)**: Agent optimized for bearish market conditions.
  - **Range Agent (`range_agent.py`)**: Agent optimized for range-bound market conditions.

- **Meta-Agent Selector (`meta_agent_selector.py`)**: Component that analyzes market conditions to identify the current regime and select the appropriate specialized agent.

- **No-Trade Filter Agent (`no_trade_filter_agent.py`)**: Top-level decision maker that determines if market conditions are suitable for trading.

### Pending Components

The following components have been identified in the original project plan but have not yet been implemented:

- **Hierarchical Evaluation Module (`hierarchical_evaluation.py`)**: For evaluating the performance of the complete agent system.
  
- **Data Preparation Module (`data_preparation.py`)**: For preprocessing and preparing data tailored to each agent type.
  
- **Callback Handlers (`callback_handlers.py`)**: For monitoring training and managing model checkpoints.

## Future Development Tasks

Based on the current implementation status, the following tasks are still pending:

1. **Develop Hierarchical Evaluation Module**: Create a comprehensive evaluation framework to assess individual agent performance and system-wide effectiveness. This should include metrics for overall system accuracy, regime classification accuracy, and trading performance. [Status: Not started]

2. **Implement Data Preparation Scripts**: Create data processing pipelines that preprocess raw market data for each agent type, including feature engineering specific to different market regimes. [Status: Not started]

3. **Create Callback Handlers**: Implement callback mechanisms to monitor training progress, save checkpoints, and facilitate early stopping based on performance metrics. [Status: Not started]

4. **Integration Testing**: Test the complete hierarchical system with all components working together, including handling transitions between different market regimes. [Status: Not started]

5. **Optimization and Hyperparameter Tuning**: Fine-tune the hyperparameters of each agent to maximize performance. [Status: Not started]

6. **Documentation Enhancement**: Expand documentation with usage examples, training guides, and performance benchmarks. [Status: In progress]

## Setup Instructions

### Python Environment

1. Create a virtual environment:
   ```
   python -m venv env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     env\Scripts\activate
     ```
   - On Linux/Mac:
     ```
     source env/bin/activate
     ```

3. Install dependencies:
   ```
   pip install stable-baselines3 gym numpy pandas matplotlib tensorflow
   ```

### System Requirements

- Python 3.8 or higher
- Sufficient RAM for training RL models (8GB+ recommended)
- GPU acceleration recommended for faster training

## Usage Guidelines

While the system is still under development, basic usage will involve:

1. **Training the Agents**:
   ```python
   # Code example for training will be provided once integration is complete
   ```

2. **Making Trading Predictions**:
   ```python
   # Code example for prediction will be provided once integration is complete
   ```

3. **Evaluating Performance**:
   ```python
   # Code example for evaluation will be provided once integration is complete
   ```

Full usage documentation will be expanded as the system reaches completion.

## Architecture Diagram

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

## Integration with C# Application

This Python module is designed to be integrated with a C# application using Python.NET. The C# application provides a user interface and handles data management, while these Python modules perform the core reinforcement learning tasks.
