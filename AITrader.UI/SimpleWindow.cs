using System;
using System.Windows;
using AITrader.UI.ViewModels;
using Microsoft.Extensions.Logging;
using System.Windows.Threading;
using System.ComponentModel;

namespace AITrader.UI
{
    /// <summary>
    /// Una ventana alternativa extremadamente simple para probar
    /// </summary>
    public class SimpleWindow : Window
    {
        private readonly ILogger<SimpleWindow> _logger;
        private readonly MainViewModel _viewModel;
        private bool _userConfirmedClose = false;
        private bool _viewModelInitialized = false;
        private bool _navigationExecuted = false;

        public SimpleWindow(MainViewModel viewModel, ILogger<SimpleWindow> logger)
        {
            _logger = logger;
            _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
            
            try
            {
                _logger.LogInformation("Inicializando SimpleWindow");
                
                // Propiedades básicas
                Title = "AITrader - Ventana Simple";
                Width = 1280;
                Height = 800;
                WindowStartupLocation = WindowStartupLocation.CenterScreen;
                
                // Crear un ContentControl en tiempo de ejecución y asignarlo
                var contentControl = new System.Windows.Controls.ContentControl
                {
                    HorizontalAlignment = HorizontalAlignment.Stretch,
                    VerticalAlignment = VerticalAlignment.Stretch,
                    HorizontalContentAlignment = HorizontalAlignment.Stretch,
                    VerticalContentAlignment = VerticalAlignment.Stretch
                };
                
                // Vincular el ContentControl al CurrentView del ViewModel
                contentControl.SetBinding(
                    System.Windows.Controls.ContentControl.ContentProperty,
                    new System.Windows.Data.Binding("CurrentView") { Source = _viewModel }
                );
                
                // Establecer el contenido de la ventana
                Content = contentControl;
                
                // Establecer el ViewModel como DataContext
                DataContext = _viewModel;
                
                // PROTECCIÓN: Interceptar los intentos de cierre de la aplicación
                Application.Current.ShutdownMode = ShutdownMode.OnExplicitShutdown;
                
                // Manejar eventos
                Loaded += SimpleWindow_Loaded;
                Closing += SimpleWindow_Closing;
                Closed += SimpleWindow_Closed;
                
                _logger.LogInformation("SimpleWindow inicializada correctamente");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error al inicializar SimpleWindow");
                MessageBox.Show($"Error al inicializar la ventana: {ex.Message}",
                               "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void SimpleWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _logger.LogInformation("SimpleWindow cargada - evento Loaded");
            
            try
            {
                // Asegurarse de que la ventana está visible
                Visibility = Visibility.Visible;
                Activate();
                Focus();
                
                // Inicializar el ViewModel de forma controlada si aún no lo hemos hecho
                if (!_viewModelInitialized)
                {
                    _logger.LogInformation("Iniciando inicialización del ViewModel");
                    _viewModel.Initialize();
                    _viewModelInitialized = true;
                    _logger.LogInformation("ViewModel inicializado correctamente");
                }
                
                // Retrasar la navegación usando un timer para asegurar que todo está cargado correctamente
                if (!_navigationExecuted)
                {
                    _logger.LogInformation("Programando navegación inicial con retraso");
                    var timer = new DispatcherTimer();
                    timer.Interval = TimeSpan.FromMilliseconds(500);
                    timer.Tick += (s, args) =>
                    {
                        _logger.LogInformation("Ejecutando navegación inicial");
                        try
                        {
                            _viewModel.NavigateToRealTimeTradingCommand.Execute(null);
                            _navigationExecuted = true;
                            _logger.LogInformation("Navegación inicial completada correctamente");
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Error en navegación inicial");
                            MessageBox.Show($"Error al navegar a la vista inicial: {ex.Message}",
                                           "Error de Navegación", MessageBoxButton.OK, MessageBoxImage.Error);
                        }
                        timer.Stop();
                    };
                    timer.Start();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error en el evento Loaded de SimpleWindow");
                MessageBox.Show($"Error al cargar la ventana: {ex.Message}",
                               "Error de Carga", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void SimpleWindow_Closing(object sender, CancelEventArgs e)
        {
            _logger.LogInformation("SimpleWindow cerrándose - evento Closing. UserConfirmedClose={UserConfirmedClose}", _userConfirmedClose);
            
            // Solo pedimos confirmación si aún no ha sido confirmado
            if (!_userConfirmedClose)
            {
                var result = MessageBox.Show("¿Está seguro que desea cerrar la aplicación?",
                                         "Confirmar Salida", MessageBoxButton.YesNo, MessageBoxImage.Question);
                
                if (result == MessageBoxResult.No)
                {
                    e.Cancel = true;
                    _logger.LogInformation("Cierre cancelado por el usuario");
                }
                else
                {
                    _userConfirmedClose = true;
                    _logger.LogInformation("Cierre confirmado por el usuario");
                }
            }
            
            // Siempre cancelamos si no ha sido confirmado
            if (!_userConfirmedClose)
            {
                e.Cancel = true;
                _logger.LogInformation("Cierre cancelado porque no ha sido confirmado por el usuario");
            }
        }
        
        private void SimpleWindow_Closed(object sender, EventArgs e)
        {
            _logger.LogInformation("SimpleWindow cerrada - evento Closed. UserConfirmedClose={UserConfirmedClose}", _userConfirmedClose);
            
            try
            {
                // Limpiar recursos si es necesario
                _viewModel?.Dispose();
                
                // Solo cerramos la aplicación si el usuario ha confirmado
                if (_userConfirmedClose)
                {
                    _logger.LogInformation("Cerrando la aplicación desde SimpleWindow (confirmado por usuario)");
                    Application.Current.Shutdown();
                }
                else
                {
                    _logger.LogWarning("¡ALERTA! La ventana se está cerrando sin confirmación del usuario");
                    
                    // Intento desesperado: crear una nueva ventana para reemplazar esta
                    try
                    {
                        _logger.LogInformation("Intentando crear una ventana de emergencia");
                        var emergencyWindow = new Window
                        {
                            Title = "¡EMERGENCIA! - AITrader",
                            Width = 600,
                            Height = 400,
                            WindowStartupLocation = WindowStartupLocation.CenterScreen,
                            Content = new System.Windows.Controls.TextBlock
                            {
                                Text = "La ventana principal se cerró inesperadamente.\nPor favor reinicie la aplicación.",
                                FontSize = 20,
                                HorizontalAlignment = HorizontalAlignment.Center,
                                VerticalAlignment = VerticalAlignment.Center
                            }
                        };
                        Application.Current.MainWindow = emergencyWindow;
                        emergencyWindow.Show();
                        _logger.LogInformation("Ventana de emergencia creada y mostrada");
                    }
                    catch (Exception emEx)
                    {
                        _logger.LogError(emEx, "Error al crear ventana de emergencia");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error en el evento Closed de SimpleWindow");
            }
        }
    }
}
