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
        private MainViewModel _viewModel;
        private ILogger<MainWindow> _logger;
        private bool _isClosingProgrammatically = false;

        public MainWindow()
        {
            try
            {
                // Inicializar componentes XAML
                InitializeComponent();
                
                // Registrar manejadores de eventos
                Loaded += MainWindow_Loaded;
                Closing += MainWindow_Closing;
                Closed += MainWindow_Closed;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al inicializar la ventana principal: {ex.Message}",
                               "Error de Inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                // Forzar la visibilidad después de que se cargue
                Visibility = Visibility.Visible;
                Activate();
                Focus();
                
                // El ViewModel ya debe estar inicializado por App.xaml.cs
                _viewModel = DataContext as MainViewModel;
                if (_viewModel == null)
                {
                    MessageBox.Show("Error: No se ha configurado el ViewModel correctamente.",
                                   "Error de inicialización", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error durante la carga de la ventana: {ex.Message}",
                               "Error de Carga", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {            
            // Si no estamos cerrando programáticamente, confirmamos con el usuario
            if (!_isClosingProgrammatically)
            {
                var result = MessageBox.Show("¿Está seguro que desea cerrar la aplicación?",
                                          "Confirmar Salida", MessageBoxButton.YesNo, MessageBoxImage.Question);
                
                if (result == MessageBoxResult.No)
                {
                    e.Cancel = true;
                }
            }
        }
        
        private void MainWindow_Closed(object sender, EventArgs e)
        {
            try
            {
                // Limpiar recursos
                if (_viewModel != null)
                {
                    _viewModel.Dispose();
                }
                
                // Solo cerramos la aplicación si este cierre fue iniciado correctamente
                if (!_isClosingProgrammatically)
                {
                    Application.Current.Shutdown();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al cerrar la aplicación: {ex.Message}", 
                              "Error", MessageBoxButton.OK, MessageBoxImage.Error);
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