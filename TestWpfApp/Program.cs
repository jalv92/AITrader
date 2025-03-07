// See https://aka.ms/new-console-template for more information
using System.Windows;

namespace TestWpfApp
{
    public class Program
    {
        [STAThread]
        public static void Main()
        {
            Application app = new Application();
            Window mainWindow = new Window
            {
                Title = "Ventana de Prueba - AITrader",
                Width = 800,
                Height = 600,
                WindowStartupLocation = WindowStartupLocation.CenterScreen,
                Background = System.Windows.Media.Brushes.White
            };

            // Agregar contenido a la ventana
            var grid = new System.Windows.Controls.Grid();
            mainWindow.Content = grid;

            // Agregar un textblock con mensaje
            var textBlock = new System.Windows.Controls.TextBlock
            {
                Text = "Esta es una ventana de prueba para AITrader",
                FontSize = 24,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                Foreground = System.Windows.Media.Brushes.DarkBlue
            };
            grid.Children.Add(textBlock);

            // Mostrar y ejecutar la aplicación
            mainWindow.Show();
            mainWindow.Activate();
            mainWindow.Focus();
            mainWindow.Topmost = true;

            // Desactivar Topmost después de un breve retraso
            var timer = new System.Windows.Threading.DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(3000);
            timer.Tick += (s, e) => {
                mainWindow.Topmost = false;
                timer.Stop();
            };
            timer.Start();

            app.Run();
        }
    }
}
