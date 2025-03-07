using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Extensions.Logging;
using AITrader.Core.Models;
using AITrader.Core.Services.RealTimeTrading;
using AITrader.UI.Commands;
using MaterialDesignThemes.Wpf;

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
        
        private DateTime _timestamp;
        public DateTime Timestamp
        {
            get => _timestamp;
            set => SetProperty(ref _timestamp, value);
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

        private string _errorMessage = string.Empty;
        public string ErrorMessage
        {
            get => _errorMessage;
            set => SetProperty(ref _errorMessage, value);
        }
        
        // Snackbar message queue for error messages
        private MaterialDesignThemes.Wpf.SnackbarMessageQueue _snackbarMessageQueue = new MaterialDesignThemes.Wpf.SnackbarMessageQueue(TimeSpan.FromSeconds(3));
        public MaterialDesignThemes.Wpf.SnackbarMessageQueue SnackbarMessageQueue
        {
            get => _snackbarMessageQueue;
            set => SetProperty(ref _snackbarMessageQueue, value);
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
            UpdateParametersCommand = new AsyncRelayCommand(UpdateTradingParameters, CanUpdateParameters);

            // Initialize default values
            TradingEnabled = true;
            PositionSizing = 1.0;
            StopLossTicks = 10;
            TakeProfitTicks = 20;
            IsLoading = false;
            CurrentPositionText = "FLAT";

            // Create UI update timer
            _uiUpdateTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            _uiUpdateTimer.Tick += (sender, e) => UpdateUI();
        }

        /// <summary>
        /// Initialize the ViewModel
        /// </summary>
        public override void Initialize()
        {
            base.Initialize();
            
            try
            {
                _logger.LogInformation("Inicializando RealTimeTradingViewModel");
                
                // En lugar de bloquear con GetAwaiter().GetResult(), iniciamos la inicialización
                // de forma asíncrona para no bloquear el hilo principal de la UI
                IsLoading = true;
                AddStatusMessage("Initializing real-time trading module...");
                
                // Iniciamos la inicialización asíncrona sin bloquear
                Task.Run(async () => {
                    try
                    {
                        bool success = await InitializeAsync();
                        
                        if (!success)
                        {
                            _logger.LogWarning("La inicialización de RealTimeTradingViewModel no fue exitosa");
                            // Actualizar la UI desde el hilo correcto
                            App.Current.Dispatcher.Invoke(() => {
                                ErrorMessage = "Inicialización incompleta. El sistema puede funcionar con capacidades limitadas.";
                                AddStatusMessage("Error: " + ErrorMessage);
                                SnackbarMessageQueue.Enqueue(ErrorMessage);
                                IsLoading = false;
                            });
                        }
                        else
                        {
                            _logger.LogInformation("RealTimeTradingViewModel inicializado correctamente");
                            // Actualizar la UI desde el hilo correcto
                            App.Current.Dispatcher.Invoke(() => {
                                AddStatusMessage("Inicialización completada correctamente.");
                                IsLoading = false;
                            });
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error durante la inicialización asíncrona de RealTimeTradingViewModel");
                        // Actualizar la UI desde el hilo correcto
                        App.Current.Dispatcher.Invoke(() => {
                            ErrorMessage = $"Error de inicialización: {ex.Message}";
                            AddStatusMessage("Error: " + ErrorMessage);
                            SnackbarMessageQueue.Enqueue(ErrorMessage);
                            IsLoading = false;
                        });
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error en Initialize() de RealTimeTradingViewModel");
                ErrorMessage = $"Error de inicialización: {ex.Message}";
                IsLoading = false;
            }
        }

        /// <summary>
        /// Initialize the ViewModel
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            if (_isInitialized)
                return true;

            try
            {
                IsLoading = true;
                AddStatusMessage("Initializing real-time trading module...");

                // Initialize trading service - con manejo adicional de errores
                bool success;
                try
                {
                    _logger.LogInformation("Iniciando inicialización del servicio RealTimeTradingService");
                    success = await _tradingService.InitializeAsync();
                    _logger.LogInformation($"Inicialización del servicio RealTimeTradingService completada: {success}");
                    
                    if (!success)
                    {
                        ErrorMessage = "Failed to initialize trading service.";
                        AddStatusMessage("Error: " + ErrorMessage);
                        return false;
                    }
                }
                catch (Exception svcEx)
                {
                    _logger.LogError(svcEx, "Error crítico al inicializar RealTimeTradingService");
                    ErrorMessage = $"Error crítico al inicializar el servicio de trading: {svcEx.Message}";
                    AddStatusMessage("Error crítico: " + ErrorMessage);
                    // Continuamos con inicialización parcial en lugar de fallar completamente
                    // para que la UI al menos sea visible
                    success = false;
                }

                // Subscribe to trading service events
                _tradingService.StatusUpdated += OnTradingServiceStatusUpdated;

                // Set initial parameters
                await UpdateTradingParameters();

                _isInitialized = true;
                AddStatusMessage("Real-time trading module initialized successfully.");

                // Start UI update timer
                _uiUpdateTimer.Start();

                return true;
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Error initializing view model: {ex.Message}";
                _logger.LogError(ex, "Error initializing RealTimeTradingViewModel");
                AddStatusMessage("Error: " + ErrorMessage);
                return false;
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
                AddStatusMessage("Starting real-time trading system...");

                // Start the trading service
                var success = await _tradingService.StartAsync();
                if (!success)
                {
                    ErrorMessage = "Failed to start trading service.";
                    AddStatusMessage("Error: " + ErrorMessage);
                    SnackbarMessageQueue.Enqueue(ErrorMessage);
                    return;
                }

                _isServiceRunning = true;
                AddStatusMessage("Real-time trading system started.");
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Error starting trading: {ex.Message}";
                _logger.LogError(ex, "Error starting trading service");
                AddStatusMessage("Error: " + ErrorMessage);
                SnackbarMessageQueue.Enqueue(ErrorMessage);
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
                AddStatusMessage("Stopping real-time trading system...");

                // Stop the trading service
                await _tradingService.StopAsync();
                _isServiceRunning = false;
                AddStatusMessage("Real-time trading system stopped.");
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Error stopping trading: {ex.Message}";
                _logger.LogError(ex, "Error stopping trading service");
                AddStatusMessage("Error: " + ErrorMessage);
                SnackbarMessageQueue.Enqueue(ErrorMessage);
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Update trading parameters
        /// </summary>
        private async Task UpdateTradingParameters()
        {
            try
            {
                // Creamos un objeto de parámetros de trading
                var parameters = new TradingParameters
                {
                    Instrument = "ES", // Por defecto usamos el E-mini S&P 500
                    Quantity = (int)PositionSizing,
                    StopLoss = StopLossTicks,
                    TakeProfit = TakeProfitTicks
                };

                // Update trading parameters
                bool success = await _tradingService.UpdateTradingParameters(parameters);

                if (success)
                {
                    AddStatusMessage($"Trading parameters updated: Enabled={TradingEnabled}, " +
                        $"Position Size={PositionSizing}, " +
                        $"Stop Loss={StopLossTicks} ticks, " +
                        $"Take Profit={TakeProfitTicks} ticks");
                }
                else
                {
                    AddStatusMessage("Failed to update trading parameters");
                }
            }
            catch (Exception ex)
            {
                AddStatusMessage($"Error updating trading parameters: {ex.Message}");
                _logger.LogError(ex, "Error updating trading parameters");
            }
        }

        /// <summary>
        /// Update UI from the trading service
        /// </summary>
        private void UpdateUI()
        {
            try
            {
                // Update connection status
                IsDataConnected = _tradingService.IsDataConnected;
                IsOrderConnected = _tradingService.IsOrderConnected;

                // Update market data
                var marketState = _tradingService.GetCurrentMarketState();
                if (marketState != null)
                {
                    Timestamp = marketState.Timestamp;
                    LastTimestamp = marketState.Timestamp.ToString("HH:mm:ss.fff");
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

                    // Check for errors
                    if (!string.IsNullOrEmpty(marketState.ErrorMessage))
                    {
                        ErrorMessage = marketState.ErrorMessage;
                        AddStatusMessage($"Error: {marketState.ErrorMessage}");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating UI");
                ErrorMessage = $"Error updating UI: {ex.Message}";
            }
        }

        /// <summary>
        /// Handle trading service status updates
        /// </summary>
        private void OnTradingServiceStatusUpdated(object? sender, RealTimeStatusEventArgs e)
        {
            AddStatusMessage($"Status update: Data connected={e.IsDataConnected}, Order connected={e.IsOrderConnected}");

            // Update connection status
            IsDataConnected = e.IsDataConnected;
            IsOrderConnected = e.IsOrderConnected;

            // Notify UI of property changes
            OnPropertyChanged(nameof(IsDataConnected));
            OnPropertyChanged(nameof(IsOrderConnected));
        }

        /// <summary>
        /// Add a status message
        /// </summary>
        private void AddStatusMessage(string message)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss");
            var fullMessage = $"[{timestamp}] {message}";

            // Check if we're on the UI thread
            if (App.Current.Dispatcher.CheckAccess())
            {
                StatusMessages.Add(fullMessage);

                // Limit the number of messages
                while (StatusMessages.Count > 100)
                {
                    StatusMessages.RemoveAt(0);
                }
            }
            else
            {
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessages.Add(fullMessage);

                    // Limit the number of messages
                    while (StatusMessages.Count > 100)
                    {
                        StatusMessages.RemoveAt(0);
                    }
                });
            }
        }

        // Command can execute methods
        private bool CanStartTrading() => _isInitialized && !_isServiceRunning;
        private bool CanStopTrading() => _isInitialized && _isServiceRunning;
        private bool CanUpdateParameters() => _isInitialized;

        /// <summary>
        /// Clean up resources
        /// </summary>
        public override void Dispose()
        {
            // Stop the UI update timer
            if (_uiUpdateTimer != null)
            {
                _uiUpdateTimer.Stop();
            }

            // Unsubscribe from events to avoid memory leaks
            if (_tradingService != null)
            {
                _tradingService.StatusUpdated -= OnTradingServiceStatusUpdated;
            }

            base.Dispose();
        }
    }
}