using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace AITrader.UI.ViewModels
{
    /// <summary>
    /// Base class for all ViewModels
    /// </summary>
    public abstract class ViewModelBase : INotifyPropertyChanged, IDisposable
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        /// <summary>
        /// Set property value and raise PropertyChanged event if value has changed
        /// </summary>
        protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (Equals(field, value))
            {
                return false;
            }

            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        /// <summary>
        /// Raise PropertyChanged event
        /// </summary>
        protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Inicializa el viewmodel. Puede ser sobrescrito por clases derivadas para realizar cualquier
        /// inicialización específica que sea necesaria.
        /// </summary>
        public virtual void Initialize()
        {
            // Por defecto no hace nada, las clases derivadas pueden sobrescribir este método
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public virtual void Dispose()
        {
            // Nothing to dispose in the base class
        }
    }
}