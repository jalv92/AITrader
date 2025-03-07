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
        public event PropertyChangedEventHandler PropertyChanged;
        
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
            
            // Crear el contenido
            var contentControl = new ContentControl
            {
                HorizontalAlignment = HorizontalAlignment.Stretch,
                VerticalAlignment = VerticalAlignment.Stretch
            };
            
            // Asignar el ViewModel como DataContext
            var viewModel = new TestViewModel();
            DataContext = viewModel;
            
            // Binding del ContentControl al ViewModel
            contentControl.Content = viewModel;
            
            // Añadir al contenido de la ventana
            Content = contentControl;
            
            // Asegurarnos que la ventana está visible
            Topmost = true;
            
            // Mensaje en la consola para confirmar
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
        
        [STAThread]
        public static void Main()
        {
            var app = new App();
            app.Run();
        }
    }
}
