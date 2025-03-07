using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using AITrader.Core.Services;
using AITrader.UI.Commands;
using AITrader.UI.ViewModels.RealTimeTrading;
using AITrader.UI.ViewModels.Backtesting;
using AITrader.UI.ViewModels.ModelTraining;
using AITrader.UI.ViewModels.Settings;

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
            _logger = logger;
            _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
            
            // Initialize commands
            NavigateToRealTimeTradingCommand = new AsyncRelayCommand(ExecuteNavigateToRealTimeTrading);
            NavigateToBacktestingCommand = new AsyncRelayCommand(ExecuteNavigateToBacktesting);
            NavigateToModelTrainingCommand = new AsyncRelayCommand(ExecuteNavigateToModelTraining);
            NavigateToSettingsCommand = new AsyncRelayCommand(ExecuteNavigateToSettings);
            
            // Initialize menu items
            InitializeMenuItems();
            
            // Set default view to avoid null reference
            CurrentView = new PlaceholderViewModel("Iniciando AITrader...");
            
            // La navegación a la vista inicial se hará desde Initialize() para asegurarnos
            // de que la ventana ya está visible y establecida
        }
        
        /// <summary>
        /// Inicializa el ViewModel y navega a la vista inicial
        /// </summary>
        public override void Initialize()
        {
            base.Initialize();
            
            _logger.LogInformation("Inicializando MainViewModel");
            
            try
            {
                // Eliminamos la navegación automática aquí para evitar competencia con App.xaml.cs
                // La navegación se controlará exclusivamente desde App.xaml.cs
                _logger.LogInformation("MainViewModel inicializado - navegación diferida a App.xaml.cs");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error durante la inicialización del MainViewModel");
                MessageBox.Show($"Error al inicializar la aplicación: {ex.Message}", 
                               "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
            }
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
        
        private async Task ExecuteNavigateToRealTimeTrading()
        {
            try
            {
                _logger.LogInformation("Navegando a Real-Time Trading");
                
                // Obtener el ViewModel
                var viewModel = _serviceProvider.GetRequiredService<RealTimeTradingViewModel>();
                _logger.LogInformation("RealTimeTradingViewModel obtenido correctamente");
                
                // Establecer una vista temporal mientras se inicializa
                CurrentView = new PlaceholderViewModel("Cargando vista de trading en tiempo real...");
                _logger.LogInformation("Vista temporal establecida");

                // Inicializar el ViewModel
                try 
                {
                    _logger.LogInformation("Iniciando inicialización del RealTimeTradingViewModel");
                    
                    // SOLUCIÓN RADICAL: Manejo especial de excepciones durante la inicialización 
                    try
                    {
                        viewModel.Initialize();
                        _logger.LogInformation("RealTimeTradingViewModel inicializado correctamente");
                    }
                    catch (Exception initEx)
                    {
                        _logger.LogError(initEx, "Error durante inicialización de RealTimeTradingViewModel");
                        
                        // No lanzamos la excepción para permitir que la vista se muestre de todos modos
                        // En lugar de eso, mostramos el error en la vista
                        MessageBox.Show($"Se produjo un error durante la inicialización:\n{initEx.Message}\n\nLa aplicación continuará en modo limitado.",
                            "Error de inicialización", MessageBoxButton.OK, MessageBoxImage.Warning);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error grave inicializando RealTimeTradingViewModel");
                    // Aún así mostramos un mensaje pero no lanzamos la excepción
                    MessageBox.Show($"Error grave durante la inicialización:\n{ex.Message}\n\nAlgunas funciones podrían no estar disponibles.",
                        "Error de inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                
                // Actualizar la UI con el ViewModel (no con la vista)
                _logger.LogInformation("Actualizando CurrentView con el RealTimeTradingViewModel");
                CurrentView = viewModel;
                _logger.LogInformation("CurrentView actualizado correctamente a RealTimeTradingViewModel");
                
                // Seleccionar el menú correspondiente
                var menuItem = MenuItems.FirstOrDefault(m => m.Name == "Real-Time Trading");
                if (menuItem != null)
                {
                    _logger.LogInformation("Seleccionando menú Real-Time Trading");
                    SelectedMenuItem = menuItem;
                    _logger.LogInformation("Menú seleccionado correctamente");
                }
                else
                {
                    _logger.LogWarning("No se encontró el menú Real-Time Trading");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error navegando a Real-Time Trading");
                
                // Mostrar mensaje al usuario
                MessageBox.Show($"Error al cargar la vista de Trading en Tiempo Real: {ex.Message}", 
                               "Error de Navegación", MessageBoxButton.OK, MessageBoxImage.Error);
                
                // Establecer una vista de error
                CurrentView = new PlaceholderViewModel($"Error: {ex.Message}");
            }
        }
        
        private async Task ExecuteNavigateToBacktesting()
        {
            try
            {
                _logger.LogInformation("Navegando a Backtesting");
                
                // Obtener el ViewModel
                var viewModel = _serviceProvider.GetRequiredService<BacktestingViewModel>();
                _logger.LogInformation("BacktestingViewModel obtenido correctamente");
                
                // Establecer una vista temporal mientras se inicializa
                CurrentView = new PlaceholderViewModel("Cargando módulo de backtesting...");
                _logger.LogInformation("Vista temporal establecida");

                // Inicializar el ViewModel
                try 
                {
                    _logger.LogInformation("Iniciando inicialización del BacktestingViewModel");
                    viewModel.Initialize();
                    _logger.LogInformation("BacktestingViewModel inicializado correctamente");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error inicializando BacktestingViewModel");
                    // En lugar de lanzar la excepción, manejamos el error como en RealTimeTradingViewModel
                    MessageBox.Show($"Error durante la inicialización:\n{ex.Message}\n\nLa aplicación continuará en modo limitado.",
                        "Error de inicialización", MessageBoxButton.OK, MessageBoxImage.Warning);
                }
                
                // Actualizar la UI con el ViewModel
                _logger.LogInformation("Actualizando CurrentView con el BacktestingViewModel");
                CurrentView = viewModel;
                _logger.LogInformation("CurrentView actualizado correctamente a BacktestingViewModel");
                
                // Seleccionar el menú correspondiente
                var menuItem = MenuItems.FirstOrDefault(m => m.Name == "Backtesting");
                if (menuItem != null)
                {
                    _logger.LogInformation("Seleccionando menú Backtesting");
                    SelectedMenuItem = menuItem;
                    _logger.LogInformation("Menú seleccionado correctamente");
                }
                else
                {
                    _logger.LogWarning("No se encontró el menú Backtesting");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error navegando a Backtesting");
                
                // Mostrar mensaje al usuario
                MessageBox.Show($"Error al cargar la vista de Backtesting: {ex.Message}", 
                               "Error de Navegación", MessageBoxButton.OK, MessageBoxImage.Error);
                
                // Establecer una vista de error
                CurrentView = new PlaceholderViewModel($"Error: {ex.Message}");
            }
        }
        
        private async Task ExecuteNavigateToModelTraining()
        {
            try
            {
                _logger.LogInformation("Navegando a Model Training");
                
                // Obtener el ViewModel
                var viewModel = _serviceProvider.GetRequiredService<ModelTrainingViewModel>();
                _logger.LogInformation("ModelTrainingViewModel obtenido correctamente");
                
                // Establecer una vista temporal mientras se inicializa
                CurrentView = new PlaceholderViewModel("Cargando módulo de entrenamiento de modelos...");
                _logger.LogInformation("Vista temporal establecida");

                // Inicializar el ViewModel
                try 
                {
                    _logger.LogInformation("Iniciando inicialización del ModelTrainingViewModel");
                    viewModel.Initialize();
                    _logger.LogInformation("ModelTrainingViewModel inicializado correctamente");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error inicializando ModelTrainingViewModel");
                    // En lugar de lanzar la excepción, manejamos el error como en RealTimeTradingViewModel
                    MessageBox.Show($"Error durante la inicialización:\n{ex.Message}\n\nLa aplicación continuará en modo limitado.",
                        "Error de inicialización", MessageBoxButton.OK, MessageBoxImage.Warning);
                }
                
                // Actualizar la UI con el ViewModel
                _logger.LogInformation("Actualizando CurrentView con el ModelTrainingViewModel");
                CurrentView = viewModel;
                _logger.LogInformation("CurrentView actualizado correctamente a ModelTrainingViewModel");
                
                // Seleccionar el menú correspondiente
                var menuItem = MenuItems.FirstOrDefault(m => m.Name == "Model Training");
                if (menuItem != null)
                {
                    _logger.LogInformation("Seleccionando menú Model Training");
                    SelectedMenuItem = menuItem;
                    _logger.LogInformation("Menú seleccionado correctamente");
                }
                else
                {
                    _logger.LogWarning("No se encontró el menú Model Training");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error navegando a Model Training");
                
                // Mostrar mensaje al usuario
                MessageBox.Show($"Error al cargar la vista de Entrenamiento de Modelos: {ex.Message}", 
                               "Error de Navegación", MessageBoxButton.OK, MessageBoxImage.Error);
                
                // Establecer una vista de error
                CurrentView = new PlaceholderViewModel($"Error: {ex.Message}");
            }
        }
        
        private async Task ExecuteNavigateToSettings()
        {
            try
            {
                _logger.LogInformation("Navegando a Settings");
                
                // Obtener el ViewModel
                var viewModel = _serviceProvider.GetRequiredService<SettingsViewModel>();
                _logger.LogInformation("SettingsViewModel obtenido correctamente");
                
                // Establecer una vista temporal mientras se inicializa
                CurrentView = new PlaceholderViewModel("Cargando configuración...");
                _logger.LogInformation("Vista temporal establecida");

                // Inicializar el ViewModel
                try 
                {
                    _logger.LogInformation("Iniciando inicialización del SettingsViewModel");
                    viewModel.Initialize();
                    _logger.LogInformation("SettingsViewModel inicializado correctamente");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error inicializando SettingsViewModel");
                    // En lugar de lanzar la excepción, manejamos el error como en RealTimeTradingViewModel
                    MessageBox.Show($"Error durante la inicialización:\n{ex.Message}\n\nLa aplicación continuará en modo limitado.",
                        "Error de inicialización", MessageBoxButton.OK, MessageBoxImage.Warning);
                }
                
                // Actualizar la UI con el ViewModel
                _logger.LogInformation("Actualizando CurrentView con el SettingsViewModel");
                CurrentView = viewModel;
                _logger.LogInformation("CurrentView actualizado correctamente a SettingsViewModel");
                
                // Seleccionar el menú correspondiente
                var menuItem = MenuItems.FirstOrDefault(m => m.Name == "Settings");
                if (menuItem != null)
                {
                    _logger.LogInformation("Seleccionando menú Settings");
                    SelectedMenuItem = menuItem;
                    _logger.LogInformation("Menú seleccionado correctamente");
                }
                else
                {
                    _logger.LogWarning("No se encontró el menú Settings");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error navegando a Settings");
                
                // Mostrar mensaje al usuario
                MessageBox.Show($"Error al cargar la vista de Configuración: {ex.Message}", 
                               "Error de Navegación", MessageBoxButton.OK, MessageBoxImage.Error);
                
                // Establecer una vista de error
                CurrentView = new PlaceholderViewModel($"Error: {ex.Message}");
            }
        }
    }
    
    /// <summary>
    /// Menu item model
    /// </summary>
    public class MenuItem : ViewModelBase
    {
        private string _name = string.Empty;
        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }

        private string _icon = string.Empty;
        public string Icon
        {
            get => _icon;
            set => SetProperty(ref _icon, value);
        }

        public ICommand Command { get; set; } = null!;
        public object CommandParameter { get; set; } = string.Empty;
    }
}