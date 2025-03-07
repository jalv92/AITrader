using System;
using System.Collections.Generic;

namespace AITrader.Core.Models
{
    /// <summary>
    /// Real-time market state data model
    /// </summary>
    public class RealTimeMarketState
    {
        public bool IsDataConnected { get; set; }
        public bool IsOrderConnected { get; set; }
        public DateTime LastUpdateTime { get; set; }
        public string MarketRegime { get; set; } = "Unknown";
        public double LastPrice { get; set; }
        public string LastTradeSignal { get; set; } = "NEUTRAL";
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; }
        public double Bid { get; set; }
        public double Ask { get; set; }
        public int Volume { get; set; }
        public int CurrentPosition { get; set; }
        public bool TradingEnabled { get; set; }
        public int DataPointsAvailable { get; set; }
        public bool ShouldTrade { get; set; }
        public string ErrorMessage { get; set; }
        public List<MarketDataPoint> RecentData { get; set; }
    }

    /// <summary>
    /// Market data point model
    /// </summary>
    public class MarketDataPoint
    {
        public string Timestamp { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public double Volume { get; set; }
        public Dictionary<string, double> Indicators { get; set; } = new Dictionary<string, double>();
    }

    /// <summary>
    /// Event args for real-time status updates
    /// </summary>
    public class RealTimeStatusEventArgs : EventArgs
    {
        public bool IsDataConnected { get; set; }
        public bool IsOrderConnected { get; set; }
    }

    /// <summary>
    /// Parameters for trading
    /// </summary>
    public class TradingParameters
    {
        public string Instrument { get; set; } = string.Empty;
        public int Quantity { get; set; } = 1;
        public double StopLoss { get; set; } = 0.0;
        public double TakeProfit { get; set; } = 0.0;
    }
}
