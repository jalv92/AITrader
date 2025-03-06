using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using LiveCharts;
using LiveCharts.Wpf;
using AITrader.UI.ViewModels.RealTimeTrading;

namespace AITrader.UI.Views.RealTimeTrading
{
    /// <summary>
    /// Interaction logic for RealTimeTradingView.xaml
    /// </summary>
    public partial class RealTimeTradingView : UserControl
    {
        private RealTimeTradingViewModel _viewModel;

        public SeriesCollection PriceSeries { get; set; }
        public SeriesCollection VolumeSeries { get; set; }
        public List<string> TimeLabels { get; set; }

        public RealTimeTradingView()
        {
            InitializeComponent();
            Loaded += OnViewLoaded;
            
            // Initialize chart series
            PriceSeries = new SeriesCollection();
            VolumeSeries = new SeriesCollection();
            TimeLabels = new List<string>();
            
            // Set up price chart series
            PriceSeries.Add(new CandleSeries
            {
                Title = "Price",
                Values = new ChartValues<OhlcPoint>(),
                StrokeThickness = 1
            });
            
            // Set up volume chart series
            VolumeSeries.Add(new ColumnSeries
            {
                Title = "Volume",
                Values = new ChartValues<double>(),
                Fill = System.Windows.Media.Brushes.DodgerBlue
            });
            
            DataContext = this;
        }
        
        private async void OnViewLoaded(object sender, RoutedEventArgs e)
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
            _viewModel.PropertyChanged += OnViewModelPropertyChanged;
            
            // Set up chart data binding
            this.DataContext = _viewModel;
        }

        private void OnViewModelPropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(RealTimeTradingViewModel.MarketData))
            {
                UpdateCharts();
            }
        }

        private void UpdateCharts()
        {
            if (_viewModel == null || _viewModel.MarketData == null || _viewModel.MarketData.Count == 0)
                return;

            // Clear existing data
            var priceValues = (PriceSeries[0].Values as ChartValues<OhlcPoint>);
            var volumeValues = (VolumeSeries[0].Values as ChartValues<double>);
            
            if (priceValues == null || volumeValues == null)
                return;
                
            priceValues.Clear();
            volumeValues.Clear();
            TimeLabels.Clear();
            
            // Add new data
            foreach (var dataPoint in _viewModel.MarketData)
            {
                // Add price data
                priceValues.Add(new OhlcPoint(
                    dataPoint.Open,
                    dataPoint.High,
                    dataPoint.Low,
                    dataPoint.Close
                ));
                
                // Add volume data
                volumeValues.Add(dataPoint.Volume);
                
                // Add time label
                try {
                    var time = DateTime.Parse(dataPoint.Timestamp);
                    TimeLabels.Add(time.ToString("HH:mm"));
                }
                catch {
                    TimeLabels.Add(dataPoint.Timestamp);
                }
            }
        }
    }
}