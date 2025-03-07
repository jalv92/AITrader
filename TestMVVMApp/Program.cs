// See https://aka.ms/new-console-template for more information
using System;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Extensions.DependencyInjection;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace TestMVVM
{
    // ViewModel Base
    public class ViewModelBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
    
    // Un ViewModel simple
    public class TestViewModel : ViewModelBase
    {
        private string _message = "Este es un ViewModel de prueba";
        
        public string Message
        {
            get => _message;
            set
            {
                _message = value;
                OnPropertyChanged();
            }
        }
    }
    
    // Ventana principal
    public class MainWindow : Window
    {
        public MainWindow()
        {
            Title = "Test MVVM Window";
            Width = 500;
            Height = 300;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            
            // Crear el grid principal
            var grid = new Grid();
            
            // Crear el ContentControl para mostrar el ViewModel
            var contentControl = new ContentControl
            {
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            
            // Añadir al grid
            grid.Children.Add(contentControl);
            
            // Crear el ViewModel y asignar como DataContext
            var viewModel = new TestViewModel();
            DataContext = viewModel;
            
            // Binding del ContentControl al ViewModel
            contentControl.Content = viewModel;
            
            // Definir explícitamente cómo se debe visualizar el ViewModel
            contentControl.ContentTemplate = new DataTemplate();
            var factory = new FrameworkElementFactory(typeof(TextBlock));
            factory.SetBinding(TextBlock.TextProperty, new System.Windows.Data.Binding("Message"));
            factory.SetValue(TextBlock.FontSizeProperty, 24.0);
            factory.SetValue(TextBlock.FontWeightProperty, FontWeights.Bold);
            factory.SetValue(TextBlock.ForegroundProperty, System.Windows.Media.Brushes.Red);
            (contentControl.ContentTemplate as DataTemplate).VisualTree = factory;
            
            // Asignar el grid como contenido de la ventana
            Content = grid;
            
            // Asegurarnos que la ventana está visible
            Topmost = true;
            
            Console.WriteLine("MainWindow creada");
        }
    }
    
    // Application
    public class App : Application
    {
        private ServiceProvider _serviceProvider;
        
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            
            // Configurar servicios
            var services = new ServiceCollection();
            ConfigureServices(services);
            _serviceProvider = services.BuildServiceProvider();
            
            // Mostrar ventana principal
            Console.WriteLine("Iniciando aplicación de prueba");
            
            // Intentar crear y mostrar la ventana
            try
            {
                var mainWindow = new MainWindow();
                Current.MainWindow = mainWindow;
                mainWindow.Show();
                mainWindow.Activate();
                mainWindow.Focus();
                
                Console.WriteLine("Ventana mostrada correctamente");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al mostrar ventana: {ex.Message}");
                Shutdown();
            }
        }
        
        private void ConfigureServices(IServiceCollection services)
        {
            // Registro simple de servicios para la aplicación de prueba
            services.AddSingleton<TestViewModel>();
        }
    }
    
    // Punto de entrada
    public class Program
    {
        [STAThread]
        public static void Main()
        {
            var app = new App();
            app.Run();
        }
    }
}
