using System;
using System.Windows;
using System.Windows.Controls;
using AITrader.UI.ViewModels.RealTimeTrading;

namespace AITrader.UI.Views.RealTimeTrading
{
    /// <summary>
    /// Interaction logic for RealTimeTradingView.xaml
    /// </summary>
    public partial class RealTimeTradingView : UserControl
    {
        private RealTimeTradingViewModel _viewModel;

        public RealTimeTradingView()
        {
            InitializeComponent();
            Loaded += ViewLoaded;
        }
        
        private async void ViewLoaded(object sender, RoutedEventArgs e)
        {
            // Get the ViewModel
            _viewModel = DataContext as RealTimeTradingViewModel;
            if (_viewModel == null)
            {
                // Try to get from resources
                _viewModel = TryFindResource("ViewModel") as RealTimeTradingViewModel;
                if (_viewModel == null)
                {
                    MessageBox.Show("ViewModel not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }
            }

            // Initialize the ViewModel
            await _viewModel.InitializeAsync();
            
            // Subscribe to market data updates
            _viewModel.PropertyChanged += ViewModelPropertyChanged;
        }

        private void ViewModelPropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            // Handle property changes as needed
        }
    }
}