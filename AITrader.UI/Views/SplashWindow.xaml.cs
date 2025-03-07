using System;
using System.Windows;
using System.Windows.Threading;

namespace AITrader.UI.Views
{
    /// <summary>
    /// Lógica de interacción para SplashWindow.xaml
    /// </summary>
    public partial class SplashWindow : Window
    {
        private readonly DispatcherTimer _statusTimer;
        private int _statusIndex = 0;
        private readonly string[] _statusMessages = new string[]
        {
            "Inicializando el motor de Python...",
            "Cargando bibliotecas científicas...",
            "Preparando servicios de trading...",
            "Configurando interfaz de usuario..."
        };

        public SplashWindow()
        {
            InitializeComponent();
            
            // Configurar el timer para cambiar los mensajes de estado
            _statusTimer = new DispatcherTimer();
            _statusTimer.Interval = TimeSpan.FromSeconds(1.5);
            _statusTimer.Tick += StatusTimer_Tick;
            _statusTimer.Start();
        }

        private void StatusTimer_Tick(object sender, EventArgs e)
        {
            _statusIndex = (_statusIndex + 1) % _statusMessages.Length;
            StatusTextBlock.Text = _statusMessages[_statusIndex];
        }

        protected override void OnClosed(EventArgs e)
        {
            _statusTimer.Stop();
            base.OnClosed(e);
        }
    }
}
