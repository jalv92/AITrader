using System;
using System.Collections.ObjectModel;
using System.Windows.Input;
using Microsoft.Extensions.Logging;
using AITrader.UI.Commands;
using AITrader.UI.Models;
using AITrader.UI.ViewModels.RealTimeTrading;

namespace AITrader.UI.ViewModels
{
    /// <summary>
    /// Main ViewModel for the application
    /// </summary>
    public class MainViewModel : ViewModelBase
    {
        private readonly ILogger<MainViewModel> _logger;
        private readonly IServiceProvider _serviceProvider;
        
        // Navigation
        private ViewModelBase _currentView;
        public ViewModelBase CurrentView
        {
            get => _currentView;
            set => SetProperty(ref _currentView, value);
        }
        
        // Navigation commands
        public ICommand NavigateToRealTimeTradingCommand { get; }
        public ICommand NavigateToBacktestingCommand { get; }
        public ICommand NavigateToModelTrainingCommand { get; }
        public ICommand NavigateToSettingsCommand { get; }
        
        // Menu items
        public ObservableCollection<MenuItem> MenuItems { get; } = new ObservableCollection<MenuItem>();
        
        // Selected menu item
        private MenuItem _selectedMenuItem;
        public MenuItem SelectedMenuItem
        {
            get => _selectedMenuItem;
            set
            {
                if (SetProperty(ref _selectedMenuItem, value) && value != null)
                {
                    // Execute the command
                    value.Command?.Execute(value.CommandParameter);
                }
            }
        }
        
        /// <summary>
        /// Constructor
        /// </summary>
        public MainViewModel(ILogger<MainViewModel> logger, IServiceProvider serviceProvider)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
            
            // Initialize commands
            NavigateToRealTimeTradingCommand = new RelayCommand(NavigateToRealTimeTrading);
            NavigateToBacktestingCommand = new RelayCommand(NavigateToBacktesting);
            NavigateToModelTrainingCommand = new RelayCommand(NavigateToModelTraining);
            NavigateToSettingsCommand = new RelayCommand(NavigateToSettings);
            
            // Initialize menu items
            InitializeMenuItems();
            
            // Set default view
            NavigateToRealTimeTrading();
        }
        
        private void InitializeMenuItems()
        {
            MenuItems.Add(new MenuItem
            {
                Name = "Real-Time Trading",
                Icon = "ChartLine",
                Command = NavigateToRealTimeTradingCommand
            });
            
            MenuItems.Add(new MenuItem
            {
                Name = "Backtesting",
                Icon = "ChartBar",
                Command = NavigateToBacktestingCommand
            });
            
            MenuItems.Add(new MenuItem
            {
                Name = "Model Training",
                Icon = "Brain",
                Command = NavigateToModelTrainingCommand
            });
            
            MenuItems.Add(new MenuItem
            {
                Name = "Settings",
                Icon = "Cog",
                Command = NavigateToSettingsCommand
            });
        }
        
        private void NavigateToRealTimeTrading()
        {
            try
            {
                // Create or get the view model from DI
                var viewModel = (RealTimeTradingViewModel)_serviceProvider.GetService(typeof(RealTimeTradingViewModel));
                if (viewModel == null)
                {
                    _logger.LogError("RealTimeTradingViewModel not registered in DI container");
                    return;
                }
                
                CurrentView = viewModel;
                _logger.LogInformation("Navigated to Real-Time Trading");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error navigating to Real-Time Trading");
            }
        }
        
        private void NavigateToBacktesting()
        {
            // Placeholder for backtesting navigation
            _logger.LogInformation("Navigated to Backtesting");
        }
        
        private void NavigateToModelTraining()
        {
            // Placeholder for model training navigation
            _logger.LogInformation("Navigated to Model Training");
        }
        
        private void NavigateToSettings()
        {
            // Placeholder for settings navigation
            _logger.LogInformation("Navigated to Settings");
        }
    }
    
    /// <summary>
    /// Menu item model
    /// </summary>
    public class MenuItem : ModelBase
    {
        private string _name;
        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }
        
        private string _icon;
        public string Icon
        {
            get => _icon;
            set => SetProperty(ref _icon, value);
        }
        
        public ICommand Command { get; set; }
        public object CommandParameter { get; set; }
    }
}