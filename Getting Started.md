# Getting Started with AITrader

This guide will help you set up and run the AITrader application for the first time.

## Prerequisites

Before starting, make sure you have:

1. **.NET 8.0 SDK** installed on your system
2. **Visual Studio 2022** or **Visual Studio Code** with C# extension
3. **NinjaTrader 8** installed and configured (see "NinjaTrader Setup Guide.md")
4. **Python environment** set up (see "Python Environment Setup.md")

## Setting Up the Project

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/AITrader.git
   cd AITrader
   ```

2. **Build the solution**:
   ```
   dotnet build
   ```

3. **Install the RLExecutor strategy in NinjaTrader 8**:
   - Follow the instructions in "NinjaTrader Setup Guide.md"

## Running the Test Connection Script

Before launching the full application, it's a good idea to test the connection to NinjaTrader:

1. **Start NinjaTrader 8** with the RLExecutor strategy applied to a chart
2. **Run the test connection script**:
   ```
   python AITrader.Core/Python/Scripts/test_connection.py
   ```
3. **Verify** that the script connects successfully and you see confirmation messages

## Running AITrader

### Option 1: Using Visual Studio

1. **Open the solution** in Visual Studio
2. **Set AITrader.UI as the startup project**:
   - Right-click on "AITrader.UI" in Solution Explorer
   - Select "Set as Startup Project"
3. **Run the application** (F5 or click the Start button)

### Option 2: Using Command Line

1. **Build and run the application**:
   ```
   dotnet run --project AITrader.UI
   ```

## First-Time Setup

When you first launch AITrader, you'll need to:

1. **Connect to NinjaTrader**:
   - AITrader will automatically attempt to connect to NinjaTrader
   - Ensure NinjaTrader is running with the RLExecutor strategy loaded
   - You should see connection status in the application

2. **Configure trading parameters**:
   - Set position sizing
   - Configure stop loss and take profit parameters
   - Enable/disable trading as needed

## Testing the Full System

To verify that the entire system is working correctly:

1. **Start NinjaTrader** with the RLExecutor strategy applied
2. **Launch AITrader**
3. **Navigate to Real-Time Trading** in the application
4. **Click "Start"** to begin real-time analysis
5. **Monitor** the status messages and connection indicators
6. **Execute a test trade**:
   - Adjust parameters as needed
   - Click "Update Parameters" to apply changes

## Troubleshooting

### Application Doesn't Start

- Check the build output for errors
- Verify that .NET 8.0 SDK is installed
- Ensure all required packages are installed

### Connection Issues

- Verify NinjaTrader is running with the RLExecutor strategy
- Check that the ports (5000 and 5001) are not blocked by firewall
- Try running the test connection script to isolate the issue

### Python Engine Issues

- Check the application logs for Python initialization errors
- Verify that all required Python packages are installed
- Try setting up a manual Python environment (see "Python Environment Setup.md")

## Next Steps

Once you have the system running correctly:

1. **Train your AI models**:
   - Use the provided scripts in `AITrader.Core/Python/Scripts`
   - Configure the parameters to match your trading strategy

2. **Test with historical data**:
   - Use the backtesting functionality to evaluate model performance

3. **Fine-tune your strategy**:
   - Adjust parameters based on backtest results
   - Iterate and improve your models