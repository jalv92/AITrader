using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Reflection;
using Python.Runtime;
using Microsoft.Extensions.Logging;
using AITrader.Core.Models;
using AITrader.Core.Services.Python;

namespace AITrader.Core.Services.RealTimeTrading
{
    /// <summary>
    /// Service for managing real-time trading with NinjaTrader using RL agents
    /// </summary>
    public class RealTimeTradingService : IDisposable
    {
        private readonly ILogger<RealTimeTradingService> _logger;
        private readonly IPythonEngineService _pythonEngine;
        private dynamic _realTimeAnalyzer;
        private bool _isRunning = false;
        private readonly object _lock = new object();
        private Timer _statusUpdateTimer;
        private CancellationTokenSource _cts;

        /// <summary>
        /// Connection status for the data and order sockets
        /// </summary>
        public bool IsDataConnected { get; private set; }
        public bool IsOrderConnected { get; private set; }

        /// <summary>
        /// Status update event
        /// </summary>
        public event EventHandler<RealTimeStatusEventArgs> StatusUpdated;

        /// <summary>
        /// Constructor
        /// </summary>
        public RealTimeTradingService(ILogger<RealTimeTradingService> logger, IPythonEngineService pythonEngine)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _pythonEngine = pythonEngine ?? throw new ArgumentNullException(nameof(pythonEngine));
        }

        /// <summary>
        /// Initialize the real-time trading service
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("Initializing real-time trading service...");
                
                // Initialize Python engine
                await _pythonEngine.InitializeAsync();

                // Use default configuration for now
                string dataHost = "127.0.0.1";
                int dataPort = 5000;
                string orderHost = "127.0.0.1";
                int orderPort = 5001;
                
                _logger.LogInformation($"Using socket configuration: DataServer={dataHost}:{dataPort}, OrderServer={orderHost}:{orderPort}");

                // Get models directory
                string modelsDir = GetModelsDirectory();
                _logger.LogInformation($"Using models directory: {modelsDir}");

                // Create real-time analyzer
                using (Py.GIL())
                {
                    // Import the module
                    dynamic realtimeModule = Py.Import("AITrader.Core.Python.RealTime.realtime_analyzer");
                    
                    // Create analyzer instance
                    _realTimeAnalyzer = realtimeModule.RealTimeMarketAnalyzer(
                        models_dir: modelsDir,
                        data_host: dataHost,
                        data_port: dataPort,
                        order_host: orderHost,
                        order_port: orderPort
                    );

                    // Set initial trading parameters - these can be updated later
                    _realTimeAnalyzer.set_trading_parameters(
                        enabled: true,
                        position_sizing: 1.0,
                        stop_loss_ticks: 10,
                        take_profit_ticks: 20
                    );
                    
                    _logger.LogInformation("Real-time analyzer created successfully");
                }
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing real-time trading service");
                return false;
            }
        }

        /// <summary>
        /// Start the real-time trading service
        /// </summary>
        public async Task<bool> StartAsync()
        {
            if (_isRunning)
            {
                _logger.LogWarning("Real-time trading service already running");
                return true;
            }

            try
            {
                _cts = new CancellationTokenSource();

                using (Py.GIL())
                {
                    bool started = _realTimeAnalyzer.start();
                    if (!started)
                    {
                        _logger.LogError("Failed to start real-time analyzer");
                        return false;
                    }
                }

                // Start status update timer
                _statusUpdateTimer = new Timer(UpdateStatus, null, 1000, 5000); // Update every 5 seconds

                _isRunning = true;
                _logger.LogInformation("Real-time trading service started");
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting real-time trading service");
                return false;
            }
        }

        /// <summary>
        /// Stop the real-time trading service
        /// </summary>
        public async Task StopAsync()
        {
            if (!_isRunning)
                return;

            try
            {
                // Stop status update timer
                _statusUpdateTimer?.Change(Timeout.Infinite, Timeout.Infinite);
                _statusUpdateTimer?.Dispose();
                _statusUpdateTimer = null;

                // Cancel any ongoing operations
                _cts?.Cancel();
                _cts?.Dispose();
                _cts = null;

                // Stop analyzer
                using (Py.GIL())
                {
                    _realTimeAnalyzer.stop();
                }

                _isRunning = false;
                IsDataConnected = false;
                IsOrderConnected = false;
                
                _logger.LogInformation("Real-time trading service stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping real-time trading service");
            }
        }

        /// <summary>
        /// Update trading parameters
        /// </summary>
        public void UpdateTradingParameters(bool enabled, double positionSizing, int stopLossTicks, int takeProfitTicks)
        {
            try
            {
                using (Py.GIL())
                {
                    _realTimeAnalyzer.set_trading_parameters(
                        enabled: enabled,
                        position_sizing: positionSizing,
                        stop_loss_ticks: stopLossTicks,
                        take_profit_ticks: takeProfitTicks
                    );
                }
                
                _logger.LogInformation($"Trading parameters updated: enabled={enabled}, position_sizing={positionSizing}, " +
                                     $"stop_loss={stopLossTicks}, take_profit={takeProfitTicks}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating trading parameters");
            }
        }

        /// <summary>
        /// Get the current state of the market
        /// </summary>
        public RealTimeMarketState GetCurrentMarketState()
        {
            try
            {
                using (Py.GIL())
                {
                    dynamic state = _realTimeAnalyzer.get_current_market_state();
                    
                    // Update connection states
                    IsDataConnected = state["data_connected"];
                    IsOrderConnected = state["order_connected"];
                    
                    // Map to C# object
                    return new RealTimeMarketState
                    {
                        IsDataConnected = state["data_connected"],
                        IsOrderConnected = state["order_connected"],
                        Timestamp = state.get("timestamp", DateTime.Now.ToString()),
                        LastPrice = state.get("last_price", 0.0),
                        CurrentPosition = state.get("current_position", 0),
                        TradingEnabled = state.get("trading_enabled", false),
                        DataPointsAvailable = state.get("data_points", 0)
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting current market state");
                return new RealTimeMarketState
                {
                    IsDataConnected = false,
                    IsOrderConnected = false,
                    Timestamp = DateTime.Now.ToString(),
                    LastPrice = 0.0,
                    CurrentPosition = 0,
                    TradingEnabled = false,
                    DataPointsAvailable = 0,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Get historical data from the buffer
        /// </summary>
        public List<MarketDataPoint> GetHistoricalData()
        {
            try
            {
                using (Py.GIL())
                {
                    dynamic df = _realTimeAnalyzer.get_historical_data();
                    
                    // Convert to C# list
                    var result = new List<MarketDataPoint>();
                    
                    // Check if DataFrame is empty
                    if (df.__len__() == 0)
                        return result;
                    
                    // Get list of column names
                    var columns = new List<string>();
                    foreach (var col in df.columns)
                    {
                        columns.Add(col.ToString());
                    }
                    
                    // Extract data
                    for (int i = 0; i < df.__len__(); i++)
                    {
                        var dataPoint = new MarketDataPoint();
                        
                        // Extract standard OHLCV values if available
                        if (columns.Contains("open"))
                            dataPoint.Open = Convert.ToDouble(df.iloc[i]["open"]);
                        
                        if (columns.Contains("high"))
                            dataPoint.High = Convert.ToDouble(df.iloc[i]["high"]);
                        
                        if (columns.Contains("low"))
                            dataPoint.Low = Convert.ToDouble(df.iloc[i]["low"]);
                        
                        if (columns.Contains("close"))
                            dataPoint.Close = Convert.ToDouble(df.iloc[i]["close"]);
                        
                        if (columns.Contains("volume"))
                            dataPoint.Volume = Convert.ToDouble(df.iloc[i]["volume"]);
                        
                        if (columns.Contains("timestamp"))
                            dataPoint.Timestamp = df.iloc[i]["timestamp"].ToString();
                        
                        // Add other available indicators as custom properties
                        foreach (var col in columns)
                        {
                            if (col != "open" && col != "high" && col != "low" && col != "close" && 
                                col != "volume" && col != "timestamp")
                            {
                                if (!dataPoint.Indicators.ContainsKey(col))
                                {
                                    dataPoint.Indicators[col] = Convert.ToDouble(df.iloc[i][col]);
                                }
                            }
                        }
                        
                        result.Add(dataPoint);
                    }
                    
                    return result;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting historical data");
                return new List<MarketDataPoint>();
            }
        }

        /// <summary>
        /// Update status and raise StatusUpdated event
        /// </summary>
        private void UpdateStatus(object state)
        {
            if (!_isRunning)
                return;

            try
            {
                var marketState = GetCurrentMarketState();
                
                // Update connection status
                lock (_lock)
                {
                    IsDataConnected = marketState.IsDataConnected;
                    IsOrderConnected = marketState.IsOrderConnected;
                }
                
                // Raise event
                StatusUpdated?.Invoke(this, new RealTimeStatusEventArgs
                {
                    MarketState = marketState
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating status");
            }
        }

        /// <summary>
        /// Get the models directory
        /// </summary>
        private string GetModelsDirectory()
        {
            // Get the base directory of the assembly
            string baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            
            // Navigate to the models directory
            string modelsDir = Path.Combine(baseDir, "models");
            
            // Create directory if it doesn't exist
            if (!Directory.Exists(modelsDir))
            {
                Directory.CreateDirectory(modelsDir);
            }
            
            return modelsDir;
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            StopAsync().Wait();
            _statusUpdateTimer?.Dispose();
            _cts?.Dispose();
        }
    }

    /// <summary>
    /// Real-time market state data model
    /// </summary>
    public class RealTimeMarketState
    {
        public bool IsDataConnected { get; set; }
        public bool IsOrderConnected { get; set; }
        public string Timestamp { get; set; }
        public double LastPrice { get; set; }
        public int CurrentPosition { get; set; }
        public bool TradingEnabled { get; set; }
        public int DataPointsAvailable { get; set; }
        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// Market data point model
    /// </summary>
    public class MarketDataPoint
    {
        public string Timestamp { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public double Volume { get; set; }
        public Dictionary<string, double> Indicators { get; set; } = new Dictionary<string, double>();
    }

    /// <summary>
    /// Event args for real-time status updates
    /// </summary>
    public class RealTimeStatusEventArgs : EventArgs
    {
        public RealTimeMarketState MarketState { get; set; }
    }
}