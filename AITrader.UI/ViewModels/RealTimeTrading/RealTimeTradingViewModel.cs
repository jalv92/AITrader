using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Extensions.Logging;
using AITrader.Core.Services.RealTimeTrading;
using AITrader.UI.Models;
using AITrader.UI.Commands;

namespace AITrader.UI.ViewModels.RealTimeTrading
{
    /// <summary>
    /// ViewModel for the real-time trading UI
    /// </summary>
    public class RealTimeTradingViewModel : ViewModelBase
    {
        private readonly ILogger<RealTimeTradingViewModel> _logger;
        private readonly RealTimeTradingService _tradingService;
        private readonly DispatcherTimer _uiUpdateTimer;
        private bool _isInitialized;
        private bool _isServiceRunning;

        // Commands
        public ICommand StartTradingCommand { get; }
        public ICommand StopTradingCommand { get; }
        public ICommand UpdateParametersCommand { get; }

        // Trading parameters
        private bool _tradingEnabled;
        public bool TradingEnabled
        {
            get => _tradingEnabled;
            set => SetProperty(ref _tradingEnabled, value);
        }

        private double _positionSizing;
        public double PositionSizing
        {
            get => _positionSizing;
            set => SetProperty(ref _positionSizing, value);
        }

        private int _stopLossTicks;
        public int StopLossTicks
        {
            get => _stopLossTicks;
            set => SetProperty(ref _stopLossTicks, value);
        }

        private int _takeProfitTicks;
        public int TakeProfitTicks
        {
            get => _takeProfitTicks;
            set => SetProperty(ref _takeProfitTicks, value);
        }

        // Connection status
        private bool _isDataConnected;
        public bool IsDataConnected
        {
            get => _isDataConnected;
            set => SetProperty(ref _isDataConnected, value);
        }

        private bool _isOrderConnected;
        public bool IsOrderConnected
        {
            get => _isOrderConnected;
            set => SetProperty(ref _isOrderConnected, value);
        }

        // Market data
        private string _lastTimestamp;
        public string LastTimestamp
        {
            get => _lastTimestamp;
            set => SetProperty(ref _lastTimestamp, value);
        }

        private double _lastPrice;
        public double LastPrice
        {
            get => _lastPrice;
            set => SetProperty(ref _lastPrice, value);
        }

        private int _currentPosition;
        public int CurrentPosition
        {
            get => _currentPosition;
            set => SetProperty(ref _currentPosition, value);
        }

        private string _currentPositionText;
        public string CurrentPositionText
        {
            get => _currentPositionText;
            set => SetProperty(ref _currentPositionText, value);
        }

        private int _dataPointsAvailable;
        public int DataPointsAvailable
        {
            get => _dataPointsAvailable;
            set => SetProperty(ref _dataPointsAvailable, value);
        }

        // Market data collection for chart
        public ObservableCollection<MarketDataPointModel> MarketData { get; } = new ObservableCollection<MarketDataPointModel>();

        // Status messages
        private ObservableCollection<string> _statusMessages = new ObservableCollection<string>();
        public ObservableCollection<string> StatusMessages
        {
            get => _statusMessages;
            set => SetProperty(ref _statusMessages, value);
        }

        // Service status
        private bool _isLoading;
        public bool IsLoading
        {
            get => _isLoading;
            set => SetProperty(ref _isLoading, value);
        }

        private string _errorMessage;
        public string ErrorMessage
        {
            get => _errorMessage;
            set => SetProperty(ref _errorMessage, value);
        }

        /// <summary>
        /// Constructor
        /// </summary>
        public RealTimeTradingViewModel(ILogger<RealTimeTradingViewModel> logger, RealTimeTradingService tradingService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _tradingService = tradingService ?? throw new ArgumentNullException(nameof(tradingService));

            // Initialize commands
            StartTradingCommand = new AsyncRelayCommand(StartTradingAsync, CanStartTrading);
            StopTradingCommand = new AsyncRelayCommand(StopTradingAsync, CanStopTrading);
            UpdateParametersCommand = new RelayCommand(UpdateTradingParameters, CanUpdateParameters);

            // Initialize default values
            TradingEnabled = true;
            PositionSizing = 1.0;
            StopLossTicks = 10;
            TakeProfitTicks = 20;

            // Setup UI update timer
            _uiUpdateTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            _uiUpdateTimer.Tick += UpdateUI;

            // Subscribe to service events
            _tradingService.StatusUpdated += OnTradingServiceStatusUpdated;
        }

        /// <summary>
        /// Initialize the ViewModel
        /// </summary>
        public async Task InitializeAsync()
        {
            if (_isInitialized)
                return;

            try
            {
                IsLoading = true;
                AddStatusMessage("Initializing real-time trading service...");

                bool initialized = await _tradingService.InitializeAsync();
                if (!initialized)
                {
                    ErrorMessage = "Failed to initialize real-time trading service";
                    AddStatusMessage("Failed to initialize real-time trading service");
                    return;
                }

                _isInitialized = true;
                AddStatusMessage("Real-time trading service initialized successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing real-time trading ViewModel");
                ErrorMessage = $"Error initializing: {ex.Message}";
                AddStatusMessage($"Error initializing: {ex.Message}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Start the trading service
        /// </summary>
        private async Task StartTradingAsync()
        {
            try
            {
                IsLoading = true;
                ErrorMessage = null;
                AddStatusMessage("Starting real-time trading...");

                bool started = await _tradingService.StartAsync();
                if (!started)
                {
                    ErrorMessage = "Failed to start real-time trading";
                    AddStatusMessage("Failed to start real-time trading");
                    return;
                }

                _isServiceRunning = true;
                AddStatusMessage("Real-time trading started successfully");

                // Start UI update timer
                _uiUpdateTimer.Start();

                // Update command can execute status
                (StartTradingCommand as AsyncRelayCommand)?.RaiseCanExecuteChanged();
                (StopTradingCommand as AsyncRelayCommand)?.RaiseCanExecuteChanged();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting real-time trading");
                ErrorMessage = $"Error starting: {ex.Message}";
                AddStatusMessage($"Error starting: {ex.Message}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Stop the trading service
        /// </summary>
        private async Task StopTradingAsync()
        {
            try
            {
                IsLoading = true;
                AddStatusMessage("Stopping real-time trading...");

                await _tradingService.StopAsync();
                _isServiceRunning = false;
                
                // Stop UI update timer
                _uiUpdateTimer.Stop();

                AddStatusMessage("Real-time trading stopped");

                // Reset connection status
                IsDataConnected = false;
                IsOrderConnected = false;

                // Update command can execute status
                (StartTradingCommand as AsyncRelayCommand)?.RaiseCanExecuteChanged();
                (StopTradingCommand as AsyncRelayCommand)?.RaiseCanExecuteChanged();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping real-time trading");
                ErrorMessage = $"Error stopping: {ex.Message}";
                AddStatusMessage($"Error stopping: {ex.Message}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Update trading parameters
        /// </summary>
        private void UpdateTradingParameters()
        {
            try
            {
                AddStatusMessage($"Updating trading parameters: enabled={TradingEnabled}, " +
                               $"position_sizing={PositionSizing}, " +
                               $"stop_loss={StopLossTicks}, " +
                               $"take_profit={TakeProfitTicks}");

                _tradingService.UpdateTradingParameters(
                    TradingEnabled,
                    PositionSizing,
                    StopLossTicks,
                    TakeProfitTicks);

                AddStatusMessage("Trading parameters updated successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating trading parameters");
                ErrorMessage = $"Error updating parameters: {ex.Message}";
                AddStatusMessage($"Error updating parameters: {ex.Message}");
            }
        }

        /// <summary>
        /// Update UI from the trading service
        /// </summary>
        private void UpdateUI(object sender, EventArgs e)
        {
            try
            {
                // Get current market state
                var marketState = _tradingService.GetCurrentMarketState();

                // Update UI properties
                IsDataConnected = marketState.IsDataConnected;
                IsOrderConnected = marketState.IsOrderConnected;
                LastTimestamp = marketState.Timestamp;
                LastPrice = marketState.LastPrice;
                CurrentPosition = marketState.CurrentPosition;
                DataPointsAvailable = marketState.DataPointsAvailable;

                // Update position text
                CurrentPositionText = CurrentPosition switch
                {
                    1 => "LONG",
                    -1 => "SHORT",
                    _ => "FLAT"
                };

                // Fetch historical data for chart (limit updates to avoid UI freezing)
                if (DateTime.Now.Second % 5 == 0) // Update every 5 seconds
                {
                    UpdateMarketDataChart();
                }

                // Check for errors
                if (!string.IsNullOrEmpty(marketState.ErrorMessage))
                {
                    ErrorMessage = marketState.ErrorMessage;
                    AddStatusMessage($"Error: {marketState.ErrorMessage}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating UI");
                ErrorMessage = $"Error updating UI: {ex.Message}";
            }
        }

        /// <summary>
        /// Update market data chart
        /// </summary>
        private void UpdateMarketDataChart()
        {
            try
            {
                var historicalData = _tradingService.GetHistoricalData();
                if (historicalData == null || historicalData.Count == 0)
                    return;

                // Clear existing data and add new data
                App.Current.Dispatcher.Invoke(() =>
                {
                    MarketData.Clear();
                    foreach (var data in historicalData)
                    {
                        MarketData.Add(new MarketDataPointModel
                        {
                            Timestamp = data.Timestamp,
                            Open = data.Open,
                            High = data.High,
                            Low = data.Low,
                            Close = data.Close,
                            Volume = data.Volume,
                            Indicators = new Dictionary<string, double>(data.Indicators)
                        });
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating market data chart");
            }
        }

        /// <summary>
        /// Handle trading service status updates
        /// </summary>
        private void OnTradingServiceStatusUpdated(object sender, RealTimeStatusEventArgs e)
        {
            try
            {
                // Update connection status
                IsDataConnected = e.MarketState.IsDataConnected;
                IsOrderConnected = e.MarketState.IsOrderConnected;

                // Log connection status changes
                if (IsDataConnected && IsOrderConnected)
                {
                    AddStatusMessage("Connected to NinjaTrader");
                }
                else if (!IsDataConnected && !IsOrderConnected)
                {
                    AddStatusMessage("Disconnected from NinjaTrader");
                }
                else
                {
                    AddStatusMessage($"Connection status: Data={IsDataConnected}, Order={IsOrderConnected}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling trading service status update");
            }
        }

        /// <summary>
        /// Add a status message
        /// </summary>
        private void AddStatusMessage(string message)
        {
            string timestamp = DateTime.Now.ToString("HH:mm:ss");
            string formattedMessage = $"[{timestamp}] {message}";

            App.Current.Dispatcher.Invoke(() =>
            {
                StatusMessages.Insert(0, formattedMessage);

                // Limit the number of messages to avoid memory issues
                if (StatusMessages.Count > 100)
                {
                    StatusMessages.RemoveAt(StatusMessages.Count - 1);
                }
            });
        }

        // Command can execute methods
        private bool CanStartTrading() => _isInitialized && !_isServiceRunning && !IsLoading;
        private bool CanStopTrading() => _isServiceRunning && !IsLoading;
        private bool CanUpdateParameters() => _isInitialized && !IsLoading;

        /// <summary>
        /// Clean up resources
        /// </summary>
        public override void Dispose()
        {
            _uiUpdateTimer.Stop();
            _tradingService.StatusUpdated -= OnTradingServiceStatusUpdated;
            base.Dispose();
        }
    }

    /// <summary>
    /// Market data point model for UI
    /// </summary>
    public class MarketDataPointModel : ModelBase
    {
        private string _timestamp;
        public string Timestamp
        {
            get => _timestamp;
            set => SetProperty(ref _timestamp, value);
        }

        private double _open;
        public double Open
        {
            get => _open;
            set => SetProperty(ref _open, value);
        }

        private double _high;
        public double High
        {
            get => _high;
            set => SetProperty(ref _high, value);
        }

        private double _low;
        public double Low
        {
            get => _low;
            set => SetProperty(ref _low, value);
        }

        private double _close;
        public double Close
        {
            get => _close;
            set => SetProperty(ref _close, value);
        }

        private double _volume;
        public double Volume
        {
            get => _volume;
            set => SetProperty(ref _volume, value);
        }

        public Dictionary<string, double> Indicators { get; set; } = new Dictionary<string, double>();
    }
}