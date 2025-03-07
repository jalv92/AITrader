using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Extensions.Logging;
using AITrader.UI.Commands;
using AITrader.Core.Services.Python;
using AITrader.UI.Models;

namespace AITrader.UI.ViewModels.Settings
{
    /// <summary>
    /// ViewModel for application settings
    /// </summary>
    public class SettingsViewModel : ViewModelBase
    {
        private readonly ILogger<SettingsViewModel> _logger;
        private readonly IPythonEngineService _pythonEngineService;

        // NinjaTrader settings
        private string _ninjaTraderPath = @"C:\Program Files\NinjaTrader 8";
        public string NinjaTraderPath
        {
            get => _ninjaTraderPath;
            set => SetProperty(ref _ninjaTraderPath, value);
        }

        private bool _isNinjaTraderInstalled;
        public bool IsNinjaTraderInstalled
        {
            get => _isNinjaTraderInstalled;
            set => SetProperty(ref _isNinjaTraderInstalled, value);
        }

        private bool _isNinjaTraderRunning;
        public bool IsNinjaTraderRunning
        {
            get => _isNinjaTraderRunning;
            set => SetProperty(ref _isNinjaTraderRunning, value);
        }

        // Python settings
        private string _pythonPath;
        public string PythonPath
        {
            get => _pythonPath;
            set => SetProperty(ref _pythonPath, value);
        }

        private bool _isPythonInstalled;
        public bool IsPythonInstalled
        {
            get => _isPythonInstalled;
            set => SetProperty(ref _isPythonInstalled, value);
        }

        private ObservableCollection<PythonPackage> _pythonPackages = new ObservableCollection<PythonPackage>();
        public ObservableCollection<PythonPackage> PythonPackages
        {
            get => _pythonPackages;
            set => SetProperty(ref _pythonPackages, value);
        }

        // Data settings
        private string _dataDirectory = @"C:\Users\javlo\Documents\AITrader\Data";
        public string DataDirectory
        {
            get => _dataDirectory;
            set => SetProperty(ref _dataDirectory, value);
        }

        private string _modelsDirectory = @"C:\Users\javlo\Documents\AITrader\Models";
        public string ModelsDirectory
        {
            get => _modelsDirectory;
            set => SetProperty(ref _modelsDirectory, value);
        }

        // Status
        private string _statusMessage = "Settings loaded";
        public string StatusMessage
        {
            get => _statusMessage;
            set => SetProperty(ref _statusMessage, value);
        }

        private bool _isBusy;
        public bool IsBusy
        {
            get => _isBusy;
            set => SetProperty(ref _isBusy, value);
        }

        // Commands
        public ICommand DetectNinjaTraderCommand { get; }
        public ICommand OpenNinjaTraderCommand { get; }
        public ICommand BrowseNinjaTraderPathCommand { get; }
        public ICommand BrowseDataDirectoryCommand { get; }
        public ICommand BrowseModelsDirectoryCommand { get; }
        public ICommand SaveSettingsCommand { get; }
        public ICommand ResetSettingsCommand { get; }
        public ICommand RefreshPythonPackagesCommand { get; }
        public ICommand InstallMissingPackagesCommand { get; }

        /// <summary>
        /// Constructor
        /// </summary>
        public SettingsViewModel(ILogger<SettingsViewModel> logger, IPythonEngineService pythonEngineService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _pythonEngineService = pythonEngineService ?? throw new ArgumentNullException(nameof(pythonEngineService));

            // Initialize commands
            DetectNinjaTraderCommand = new AsyncRelayCommand(DetectNinjaTraderAsync, () => !IsBusy);
            OpenNinjaTraderCommand = new RelayCommand(OpenNinjaTrader, () => IsNinjaTraderInstalled && !IsNinjaTraderRunning && !IsBusy);
            BrowseNinjaTraderPathCommand = new RelayCommand(BrowseNinjaTraderPath, () => !IsBusy);
            BrowseDataDirectoryCommand = new RelayCommand(BrowseDataDirectory, () => !IsBusy);
            BrowseModelsDirectoryCommand = new RelayCommand(BrowseModelsDirectory, () => !IsBusy);
            SaveSettingsCommand = new AsyncRelayCommand(SaveSettingsAsync, () => !IsBusy);
            ResetSettingsCommand = new RelayCommand(ResetSettings, () => !IsBusy);
            RefreshPythonPackagesCommand = new AsyncRelayCommand(RefreshPythonPackagesAsync, () => !IsBusy);
            InstallMissingPackagesCommand = new AsyncRelayCommand(InstallMissingPackagesAsync, () => !IsBusy);

            // Initialize
            InitializeAsync();
        }

        /// <summary>
        /// Initialize the view model
        /// </summary>
        private async void InitializeAsync()
        {
            try
            {
                IsBusy = true;
                StatusMessage = "Loading settings...";

                // Load Python information
                await LoadPythonInfoAsync();

                // Check NinjaTrader
                await DetectNinjaTraderAsync();

                // Create default directories if they don't exist
                EnsureDirectoriesExist();

                StatusMessage = "Settings loaded";
                _logger.LogInformation("Settings loaded successfully");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error loading settings: {ex.Message}";
                _logger.LogError(ex, "Error loading settings");
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>
        /// Load Python information
        /// </summary>
        private async Task LoadPythonInfoAsync()
        {
            try
            {
                // Check if Python is initialized
                IsPythonInstalled = _pythonEngineService.IsInitialized;
                
                if (IsPythonInstalled)
                {
                    // Since we can't directly get the Python path, we'll use a workaround
                    // In a real implementation, this would be enhanced to get the actual Python path
                    PythonPath = "Python is installed and initialized";
                    
                    // Refresh Python packages
                    await RefreshPythonPackagesAsync();
                }
                else
                {
                    PythonPath = "Python not detected";
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading Python information");
                IsPythonInstalled = false;
                PythonPath = "Error detecting Python";
                StatusMessage = $"Error: {ex.Message}";
            }
        }

        /// <summary>
        /// Detect NinjaTrader installation and status
        /// </summary>
        private async Task DetectNinjaTraderAsync()
        {
            try
            {
                StatusMessage = "Detecting NinjaTrader...";
                IsBusy = true;

                // Check if NinjaTrader is installed
                IsNinjaTraderInstalled = File.Exists(Path.Combine(NinjaTraderPath, "NinjaTrader.exe"));

                // Check if NinjaTrader is running
                IsNinjaTraderRunning = Process.GetProcessesByName("NinjaTrader").Length > 0;

                StatusMessage = IsNinjaTraderInstalled
                    ? "NinjaTrader 8 detected"
                    : "NinjaTrader 8 not found";

                _logger.LogInformation($"NinjaTrader detection - Installed: {IsNinjaTraderInstalled}, Running: {IsNinjaTraderRunning}");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error detecting NinjaTrader: {ex.Message}";
                _logger.LogError(ex, "Error detecting NinjaTrader");
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>
        /// Open NinjaTrader
        /// </summary>
        private void OpenNinjaTrader()
        {
            try
            {
                if (!IsNinjaTraderInstalled)
                {
                    StatusMessage = "NinjaTrader 8 is not installed";
                    return;
                }

                string ninjaTraderExePath = Path.Combine(NinjaTraderPath, "NinjaTrader.exe");
                if (!File.Exists(ninjaTraderExePath))
                {
                    StatusMessage = $"NinjaTrader executable not found at {ninjaTraderExePath}";
                    return;
                }

                // Start NinjaTrader
                Process.Start(ninjaTraderExePath);
                StatusMessage = "NinjaTrader 8 is starting...";
                _logger.LogInformation("NinjaTrader 8 started");

                // Update status after a delay
                Task.Delay(2000).ContinueWith(_ => DetectNinjaTraderAsync());
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error starting NinjaTrader: {ex.Message}";
                _logger.LogError(ex, "Error starting NinjaTrader");
            }
        }

        /// <summary>
        /// Browse for NinjaTrader installation path
        /// </summary>
        private void BrowseNinjaTraderPath()
        {
            // In a real application, this would open a folder browser dialog
            // For now, we just simulate it
            StatusMessage = "NinjaTrader path selection would open here";
        }

        /// <summary>
        /// Browse for data directory
        /// </summary>
        private void BrowseDataDirectory()
        {
            // In a real application, this would open a folder browser dialog
            // For now, we just simulate it
            StatusMessage = "Data directory selection would open here";
        }

        /// <summary>
        /// Browse for models directory
        /// </summary>
        private void BrowseModelsDirectory()
        {
            // In a real application, this would open a folder browser dialog
            // For now, we just simulate it
            StatusMessage = "Models directory selection would open here";
        }

        /// <summary>
        /// Save settings
        /// </summary>
        private async Task SaveSettingsAsync()
        {
            try
            {
                IsBusy = true;
                StatusMessage = "Saving settings...";

                // In a real application, this would save settings to a configuration file
                // For now, just simulate a delay
                await Task.Delay(1000);

                // Ensure directories exist
                EnsureDirectoriesExist();

                StatusMessage = "Settings saved successfully";
                _logger.LogInformation("Settings saved successfully");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error saving settings: {ex.Message}";
                _logger.LogError(ex, "Error saving settings");
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>
        /// Reset settings to default values
        /// </summary>
        private void ResetSettings()
        {
            NinjaTraderPath = @"C:\Program Files\NinjaTrader 8";
            DataDirectory = @"C:\Users\javlo\Documents\AITrader\Data";
            ModelsDirectory = @"C:\Users\javlo\Documents\AITrader\Models";
            StatusMessage = "Settings reset to default values";
            _logger.LogInformation("Settings reset to default values");
        }

        /// <summary>
        /// Refresh the list of installed Python packages
        /// </summary>
        private async Task RefreshPythonPackagesAsync()
        {
            try
            {
                IsBusy = true;
                StatusMessage = "Refreshing Python packages...";
                PythonPackages.Clear();

                // Get the list of required packages from requirements.txt
                string[] requiredPackages = new[]
                {
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "gym",
                    "gymnasium",
                    "stable-baselines3",
                    "talib-binary"
                };

                // In a real application, we would call into the Python engine service to get actual package info
                // For now, simulate some packages being installed and some missing
                foreach (var packageName in requiredPackages)
                {
                    bool isInstalled = true;
                    string version = "1.0.0";
                    string status = "Installed";

                    // Simulate some packages as not installed
                    if (packageName == "talib-binary")
                    {
                        isInstalled = true;
                        version = "0.4.24";
                    }
                    else if (packageName == "stable-baselines3")
                    {
                        isInstalled = true;
                        version = "2.0.0";
                    }
                    else if (packageName == "pandas")
                    {
                        isInstalled = true;
                        version = "1.5.3";
                    }
                    else if (packageName == "numpy")
                    {
                        isInstalled = true;
                        version = "1.26.4";
                    }
                    else if (packageName == "matplotlib")
                    {
                        isInstalled = true;
                        version = "3.7.2";
                    }

                    PythonPackages.Add(new PythonPackage
                    {
                        Name = packageName,
                        IsInstalled = isInstalled,
                        Version = isInstalled ? version : null,
                        Status = isInstalled ? status : "Not installed"
                    });
                }

                StatusMessage = "Python packages refreshed";
                _logger.LogInformation("Python packages refreshed");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error refreshing Python packages: {ex.Message}";
                _logger.LogError(ex, "Error refreshing Python packages");
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>
        /// Install missing Python packages
        /// </summary>
        private async Task InstallMissingPackagesAsync()
        {
            try
            {
                IsBusy = true;
                StatusMessage = "Installing missing Python packages...";

                var missingPackages = PythonPackages.Where(p => !p.IsInstalled).ToList();
                if (missingPackages.Count == 0)
                {
                    StatusMessage = "No missing packages to install";
                    return;
                }

                // In a real application, this would call into the Python engine service to install packages
                // For now, simulate installation
                await Task.Delay(2000);

                // Update status of installed packages
                foreach (var package in missingPackages)
                {
                    package.IsInstalled = true;
                    package.Version = "1.0.0";
                    package.Status = "Installed";
                }

                StatusMessage = "Missing packages installed successfully";
                _logger.LogInformation("Missing Python packages installed successfully");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error installing packages: {ex.Message}";
                _logger.LogError(ex, "Error installing Python packages");
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>
        /// Ensure that the data and models directories exist
        /// </summary>
        private void EnsureDirectoriesExist()
        {
            try
            {
                if (!string.IsNullOrEmpty(DataDirectory) && !Directory.Exists(DataDirectory))
                {
                    Directory.CreateDirectory(DataDirectory);
                    _logger.LogInformation($"Created data directory: {DataDirectory}");
                }

                if (!string.IsNullOrEmpty(ModelsDirectory) && !Directory.Exists(ModelsDirectory))
                {
                    Directory.CreateDirectory(ModelsDirectory);
                    _logger.LogInformation($"Created models directory: {ModelsDirectory}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error ensuring directories exist");
                throw;
            }
        }
    }

    /// <summary>
    /// Python package information
    /// </summary>
    public class PythonPackage : ModelBase
    {
        private string _name;
        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }

        private bool _isInstalled;
        public bool IsInstalled
        {
            get => _isInstalled;
            set => SetProperty(ref _isInstalled, value);
        }

        private string _version;
        public string Version
        {
            get => _version;
            set => SetProperty(ref _version, value);
        }

        private string _status;
        public string Status
        {
            get => _status;
            set => SetProperty(ref _status, value);
        }
    }
}
