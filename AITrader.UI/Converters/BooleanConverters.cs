using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace AITrader.UI.Converters
{
    /// <summary>
    /// Converts a boolean value to its inverse (true to false, false to true)
    /// </summary>
    public class InverseBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                return !boolValue;
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                return !boolValue;
            }
            return false;
        }
    }

    /// <summary>
    /// Converts a boolean value to a Visibility enum value
    /// </summary>
    public class BooleanToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                return boolValue ? Visibility.Visible : Visibility.Collapsed;
            }
            return Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is Visibility visibilityValue)
            {
                return visibilityValue == Visibility.Visible;
            }
            return false;
        }
    }

    /// <summary>
    /// Converts a boolean value to a color based on parameter
    /// </summary>
    public class BooleanToColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool boolValue && parameter is string colorValues)
            {
                string[] colors = colorValues.Split(':');
                if (colors.Length >= 2)
                {
                    // Try to create a SolidColorBrush from the color name
                    string colorName = boolValue ? colors[0] : colors[1];
                    try
                    {
                        if (colorName.StartsWith("#"))
                        {
                            // It's a color code
                            return new SolidColorBrush((Color)ColorConverter.ConvertFromString(colorName));
                        }
                        else
                        {
                            // It's a named color
                            var color = (Color)typeof(Colors).GetProperty(colorName)?.GetValue(null);
                            return new SolidColorBrush(color);
                        }
                    }
                    catch
                    {
                        return new SolidColorBrush(Colors.Black);
                    }
                }
            }
            return new SolidColorBrush(Colors.Black);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Checks if a numeric value is greater than zero
    /// </summary>
    public class GreaterThanZeroConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is decimal decimalValue)
            {
                return decimalValue > 0;
            }
            if (value is double doubleValue)
            {
                return doubleValue > 0;
            }
            if (value is int intValue)
            {
                return intValue > 0;
            }
            if (value is float floatValue)
            {
                return floatValue > 0;
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Checks if a numeric value is less than zero
    /// </summary>
    public class LessThanZeroConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is decimal decimalValue)
            {
                return decimalValue < 0;
            }
            if (value is double doubleValue)
            {
                return doubleValue < 0;
            }
            if (value is int intValue)
            {
                return intValue < 0;
            }
            if (value is float floatValue)
            {
                return floatValue < 0;
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
