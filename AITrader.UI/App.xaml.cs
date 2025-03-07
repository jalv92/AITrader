using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Debug;
using AITrader.Core;
using AITrader.Core.Services;
using AITrader.Core.Services.Python;
using AITrader.UI.ViewModels;
using AITrader.UI.ViewModels.RealTimeTrading;
using AITrader.UI.ViewModels.Backtesting;
using AITrader.UI.ViewModels.ModelTraining;
using AITrader.UI.ViewModels.Settings;
using AITrader.UI.Views.RealTimeTrading;
using AITrader.UI.Views.Backtesting;
using AITrader.UI.Views.ModelTraining;
using AITrader.UI.Views.Settings;

namespace AITrader.UI
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private ServiceProvider _serviceProvider;
        private ILogger<App> _logger;

        protected override void OnStartup(StartupEventArgs e)
        {
            try
            {
                base.OnStartup(e);
                
                // Configurar manejo global de excepciones no controladas
                AppDomain.CurrentDomain.UnhandledException += (sender, args) =>
                {
                    var exception = args.ExceptionObject as Exception;
                    MessageBox.Show($"Error no controlado: {exception?.Message}\n\nStackTrace: {exception?.StackTrace}",
                                   "Error Fatal", MessageBoxButton.OK, MessageBoxImage.Error);
                };

                // Configurar manejo de excepciones en el dispatcher
                DispatcherUnhandledException += (sender, args) =>
                {
                    MessageBox.Show($"Error en el Dispatcher: {args.Exception.Message}\n\nStackTrace: {args.Exception.StackTrace}",
                                   "Error de UI", MessageBoxButton.OK, MessageBoxImage.Error);
                    args.Handled = true; // Marcar como manejado para evitar el cierre de la aplicación
                };
                
                // Establecer cómo se cierra la aplicación
                ShutdownMode = ShutdownMode.OnMainWindowClose;
                
                try
                {
                    // Configuración de servicios
                    var serviceCollection = new ServiceCollection();
                    ConfigureServices(serviceCollection);
                    _serviceProvider = serviceCollection.BuildServiceProvider();
                    
                    // Obtener logger
                    _logger = _serviceProvider.GetRequiredService<ILogger<App>>();
                    _logger.LogInformation("Iniciando aplicación AITrader");
                    
                    // Inicializar Python asíncronamente
                    try 
                    {
                        _logger.LogInformation("Iniciando inicialización de Python");
                        var pythonService = _serviceProvider.GetRequiredService<IPythonEngineService>();
                        
                        // Usamos un DispatcherTimer para manejar la inicialización asíncrona
                        var initTimer = new DispatcherTimer();
                        initTimer.Interval = TimeSpan.FromMilliseconds(100);
                        initTimer.Tick += async (s, args) => 
                        {
                            initTimer.Stop();
                            bool success = await pythonService.InitializeAsync();
                            if (success)
                            {
                                _logger.LogInformation("Python inicializado correctamente");
                                InitializeMainWindow();
                            }
                            else
                            {
                                _logger.LogError("Python no se pudo inicializar");
                                MessageBox.Show("Error al inicializar Python, la aplicación funcionará con funcionalidad limitada.",
                                               "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Warning);
                                InitializeMainWindow();
                            }
                        };
                        initTimer.Start();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error al inicializar Python");
                        MessageBox.Show($"Error al inicializar Python: {ex.Message}", 
                                      "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
                        
                        // Aún así, intentamos mostrar la ventana principal
                        InitializeMainWindow();
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error al iniciar la aplicación: {ex.Message}", 
                                  "Error de Inicio", MessageBoxButton.OK, MessageBoxImage.Error);
                    Shutdown(-1);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al iniciar la aplicación: {ex.Message}", 
                              "Error de Inicio", MessageBoxButton.OK, MessageBoxImage.Error);
                Shutdown(-1);
            }
        }
        
        private void InitializeMainWindow()
        {
            try
            {
                _logger.LogInformation("Inicializando ventana principal");
                
                // Crear y configurar los DataTemplates para asociar ViewModels con Vistas
                var resourceDictionary = new ResourceDictionary();
                
                // DataTemplate para RealTimeTradingViewModel
                var realTimeTradingTemplate = new DataTemplate(typeof(RealTimeTradingViewModel));
                var realTimeTradingFactory = new FrameworkElementFactory(typeof(RealTimeTradingView));
                realTimeTradingTemplate.VisualTree = realTimeTradingFactory;
                resourceDictionary.Add(typeof(RealTimeTradingViewModel), realTimeTradingTemplate);
                
                // DataTemplate para BacktestingViewModel
                var backtestingTemplate = new DataTemplate(typeof(BacktestingViewModel));
                var backtestingFactory = new FrameworkElementFactory(typeof(BacktestingView));
                backtestingTemplate.VisualTree = backtestingFactory;
                resourceDictionary.Add(typeof(BacktestingViewModel), backtestingTemplate);
                
                // DataTemplate para ModelTrainingViewModel
                var modelTrainingTemplate = new DataTemplate(typeof(ModelTrainingViewModel));
                var modelTrainingFactory = new FrameworkElementFactory(typeof(ModelTrainingView));
                modelTrainingTemplate.VisualTree = modelTrainingFactory;
                resourceDictionary.Add(typeof(ModelTrainingViewModel), modelTrainingTemplate);
                
                // DataTemplate para SettingsViewModel
                var settingsTemplate = new DataTemplate(typeof(SettingsViewModel));
                var settingsFactory = new FrameworkElementFactory(typeof(SettingsView));
                settingsTemplate.VisualTree = settingsFactory;
                resourceDictionary.Add(typeof(SettingsViewModel), settingsTemplate);
                
                // Agregar el ResourceDictionary a los recursos de la aplicación
                Resources.MergedDictionaries.Add(resourceDictionary);
                
                // Obtener el MainViewModel
                var viewModel = _serviceProvider.GetRequiredService<MainViewModel>();
                viewModel.Initialize();
                
                // Crear una instancia de MainWindow desde XAML
                var mainWindow = new MainWindow();
                
                // Establecer el DataContext
                mainWindow.DataContext = viewModel;
                
                // Establecer como ventana principal
                MainWindow = mainWindow;
                
                // Manejar evento de carga
                mainWindow.Loaded += (s, e) =>
                {
                    _logger.LogInformation("Ventana principal cargada - evento Loaded");
                    
                    // Navegar a la vista inicial después de un pequeño retraso
                    var timer = new DispatcherTimer();
                    timer.Interval = TimeSpan.FromMilliseconds(500);
                    timer.Tick += (ts, te) =>
                    {
                        _logger.LogInformation("Iniciando navegación a RealTimeTrading");
                        viewModel.NavigateToRealTimeTradingCommand.Execute(null);
                        timer.Stop();
                    };
                    timer.Start();
                };
                
                // Mostrar la ventana
                mainWindow.Show();
                mainWindow.Activate();
                mainWindow.Focus();
                
                _logger.LogInformation("Ventana principal mostrada correctamente");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error al inicializar la ventana principal");
                MessageBox.Show($"Error al inicializar la ventana principal: {ex.Message}\n\nStackTrace: {ex.StackTrace}",
                               "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ConfigureServices(IServiceCollection services)
        {
            // Registrar servicios de Core
            services.RegisterServices();
            
            // ViewModels
            services.AddSingleton<MainViewModel>();
            services.AddTransient<RealTimeTradingViewModel>();
            services.AddTransient<BacktestingViewModel>();
            services.AddTransient<ModelTrainingViewModel>();
            services.AddTransient<SettingsViewModel>();
            
            // Logging
            services.AddLogging(configure =>
            {
                configure.AddConsole();
                configure.AddDebug(); // Ahora tenemos la referencia correcta
                configure.SetMinimumLevel(LogLevel.Debug);
            });
            
            // Vistas
            services.AddTransient<Views.RealTimeTrading.RealTimeTradingView>();
            services.AddTransient<Views.Backtesting.BacktestingView>();
            services.AddTransient<Views.ModelTraining.ModelTrainingView>();
            services.AddTransient<Views.Settings.SettingsView>();
        }

        protected override void OnExit(ExitEventArgs e)
        {
            // Limpieza de recursos
            _serviceProvider?.Dispose();
            
            base.OnExit(e);
        }
    }
}