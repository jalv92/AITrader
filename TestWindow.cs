using System;
using System.Windows;

namespace AITrader
{
    public class TestWindow
    {
        public static void Run()
        {
            try
            {
                Application app = new Application();
                Window window = new Window
                {
                    Title = "Test Window",
                    Width = 800,
                    Height = 600,
                    WindowStartupLocation = WindowStartupLocation.CenterScreen,
                    Content = new System.Windows.Controls.TextBlock
                    {
                        Text = "La ventana de prueba está funcionando correctamente",
                        FontSize = 24,
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center
                    }
                };
                
                window.Show();
                app.Run();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al crear la ventana de prueba: {ex.Message}", 
                               "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}
