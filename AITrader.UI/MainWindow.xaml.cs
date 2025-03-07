using System;
using System.Windows;
using AITrader.UI.ViewModels;
using Microsoft.Extensions.Logging;
using System.Windows.Threading;
using System.Windows.Controls;

namespace AITrader.UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _viewModel;
        private readonly ILogger<MainWindow> _logger;
        private bool _isClosingProgrammatically = false;

        public MainWindow(MainViewModel viewModel, ILogger<MainWindow> logger)
        {
            _logger = logger;
            _logger.LogInformation("Inicializando MainWindow");
            
            try
            {
                // Inicializar componentes XAML
                InitializeComponent();
                
                // Configuración explícita de la ventana para asegurar visibilidad
                Title = "AITrader - Sistema de Trading Algorítmico";
                Width = 1280;
                Height = 800;
                WindowStartupLocation = WindowStartupLocation.CenterScreen;
                Visibility = Visibility.Visible;
                WindowState = WindowState.Normal;
                
                // Guardar referencias
                _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
                
                // Establecer DataContext
                DataContext = _viewModel;
                
                // Registrar manejadores de eventos
                Loaded += MainWindow_Loaded;
                Closing += MainWindow_Closing;
                Closed += MainWindow_Closed;
                
                _logger.LogInformation("MainWindow inicializada correctamente");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error al inicializar MainWindow");
                MessageBox.Show($"Error al inicializar la ventana principal: {ex.Message}",
                               "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _logger.LogInformation("Evento Loaded de MainWindow disparado");
            
            try
            {
                // Forzar la visibilidad otra vez después de que se cargue
                Visibility = Visibility.Visible;
                Activate();
                Focus();
                
                // Inicializar el ViewModel después de que todo está cargado correctamente
                _viewModel.Initialize();
                
                // Establecer un ViewModel inicial después de un breve retraso
                // para asegurar que todo se haya inicializado correctamente
                var timer = new DispatcherTimer();
                timer.Interval = TimeSpan.FromMilliseconds(1000);
                timer.Tick += (s, args) =>
                {
                    _logger.LogInformation("Ejecutando navegación inicial a RealTimeTrading");
                    _viewModel.NavigateToRealTimeTradingCommand.Execute(null);
                    timer.Stop();
                };
                timer.Start();
                
                _logger.LogInformation("MainWindow se ha cargado completamente");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error en el evento Loaded de MainWindow");
                MessageBox.Show($"Error durante la carga de la ventana: {ex.Message}",
                               "Error de Carga", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _logger.LogInformation("Evento Closing de MainWindow disparado");
            
            // Si no estamos cerrando programáticamente, confirmamos con el usuario
            if (!_isClosingProgrammatically)
            {
                var result = MessageBox.Show("¿Está seguro que desea cerrar la aplicación?",
                                          "Confirmar Salida", MessageBoxButton.YesNo, MessageBoxImage.Question);
                
                if (result == MessageBoxResult.No)
                {
                    e.Cancel = true;
                    _logger.LogInformation("Cierre de la aplicación cancelado por el usuario");
                }
                else
                {
                    _logger.LogInformation("Cierre de la aplicación confirmado por el usuario");
                }
            }
        }
        
        private void MainWindow_Closed(object sender, EventArgs e)
        {
            try
            {
                _logger.LogInformation("Evento Closed de MainWindow disparado");
                
                // Limpiar recursos
                _viewModel?.Dispose();
                
                // Solo cerramos la aplicación si este cierre fue iniciado correctamente
                if (!_isClosingProgrammatically)
                {
                    _logger.LogInformation("Cerrando la aplicación desde MainWindow");
                    Application.Current.Shutdown();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error al cerrar la aplicación");
            }
        }
        
        // Método para cerrar la ventana programáticamente
        public void CloseWindow()
        {
            _isClosingProgrammatically = true;
            Close();
        }
    }
}