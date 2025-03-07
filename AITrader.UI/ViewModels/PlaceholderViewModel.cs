using System;
using Microsoft.Extensions.Logging;

namespace AITrader.UI.ViewModels
{
    /// <summary>
    /// A simple placeholder ViewModel usado mientras se cargan otros ViewModels
    /// </summary>
    public class PlaceholderViewModel : ViewModelBase
    {
        private string _message;

        /// <summary>
        /// Mensaje a mostrar
        /// </summary>
        public string Message
        {
            get => _message;
            set
            {
                _message = value;
                OnPropertyChanged(nameof(Message));
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="message">Mensaje a mostrar</param>
        public PlaceholderViewModel(string message)
        {
            _message = message ?? "Loading...";
        }

        /// <summary>
        /// Constructor por defecto
        /// </summary>
        public PlaceholderViewModel() : this("Loading...")
        {
        }
    }
}
