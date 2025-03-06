#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class ExportDataToCSV1 : Indicator
    {
        [NinjaScriptProperty]
        [Display(Name = "Export to CSV", Description = "Set to true to export data to CSV", Order = 1, GroupName = "Parameters")]
        public bool ExportToCSV { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Export Path", Description = "Path to export CSV file", Order = 2, GroupName = "Parameters")]
        public string ExportPath { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Max Bars to Export", Description = "Maximum number of bars to export (from newest to oldest)", Order = 3, GroupName = "Parameters")]
        public int MaxBarsToExport { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Exporta los datos del gráfico y valores de indicadores a un archivo CSV";
                Name = "ExportDataToCSV";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                DrawOnPricePanel = false;
                IsSuspendedWhileInactive = true;

                // Valores por defecto
                ExportPath = @"C:\Users\javlo\Documents\NT8 RL\data";
                MaxBarsToExport = 100; // Por defecto exportar 100 barras
            }
        }

        protected override void OnBarUpdate()
        {
            if (ExportToCSV)
            {
                ExportDataToCSV();
                ExportToCSV = false;
            }
        }

        private void ExportDataToCSV()
        {
            try
            {
                if (ChartControl == null)
                {
                    Print("ChartControl no está disponible, no se puede exportar");
                    return;
                }

                // Obtener y sanitizar el nombre del instrumento
                string instrumentName = Instrument.FullName;
                string sanitizedInstrument = SanitizeFileName(instrumentName);

                // Obtener fecha y hora actuales
                DateTime now = DateTime.Now;
                string datePart = now.ToString("MM-yyyy"); // mes-año en formato MM-yyyy sin slash
                string timePart = now.ToString("HHmm"); // hora en formato HHMM sin dos puntos

                // Construir el nombre del archivo con formato compatible con Windows
                string fileName = $"{sanitizedInstrument}_Data_{datePart}_{timePart}.csv";

                // Usar la ruta especificada por el usuario
                string exportFolder = ExportPath;
                if (!Directory.Exists(exportFolder))
                {
                    try
                    {
                        Directory.CreateDirectory(exportFolder);
                    }
                    catch (Exception ex)
                    {
                        Print($"No se pudo crear la carpeta: {ex.Message}");
                        // Usar el directorio temporal del sistema como alternativa
                        exportFolder = Path.GetTempPath();
                        Print($"Usando directorio temporal: {exportFolder}");
                    }
                }

                // Ruta completa del archivo
                string fullPath = Path.Combine(exportFolder, fileName);
                
                // Verificar que la ruta sea válida
                try
                {
                    FileInfo fileInfo = new FileInfo(fullPath);
                    Print($"Ruta del archivo validada: {fullPath}");
                }
                catch (Exception ex)
                {
                    Print($"Error en la ruta del archivo: {ex.Message}");
                    // Intentar con una ruta alternativa más simple
                    fullPath = Path.Combine(Path.GetTempPath(), "export_data.csv");
                    Print($"Usando ruta alternativa: {fullPath}");
                }

                // Obtener la lista de otros indicadores en el gráfico
                var otherIndicators = ChartControl.Indicators.Where(ind => ind != this).ToList();

                // Crear mapeo de indicadores a sus valores
                var indicatorValues = new List<Tuple<string, Func<int, string>>>();

                // Construir el encabezado del CSV
                List<string> header = new List<string> { "Timestamp", "Open", "High", "Low", "Close" };
                
                // Procesar indicadores para el encabezado y crear funciones de obtención de valores
                foreach (var ind in otherIndicators)
                {
                    try
                    {
                        var plots = ind.Plots.ToList();
                        for (int plotIndex = 0; plotIndex < plots.Count; plotIndex++)
                        {
                            var plot = plots[plotIndex];
                            string plotName = plot.Name;
                            string headerName = $"{ind.Name}_{plotName}";
                            header.Add(headerName);
                            
                            int finalPlotIndex = plotIndex; // Capturar para la lambda
                            indicatorValues.Add(Tuple.Create(headerName, (Func<int, string>)(barsAgo => {
                                try
                                {
                                    if (barsAgo < 0 || barsAgo >= BarsArray[0].Count)
                                        return string.Empty;
                                        
                                    double value = ((ISeries<double>)ind.Values[finalPlotIndex])[barsAgo];
                                    return value.ToString(System.Globalization.CultureInfo.InvariantCulture);
                                }
                                catch
                                {
                                    return string.Empty;
                                }
                            })));
                        }
                    }
                    catch (Exception ex)
                    {
                        Print($"Error al procesar indicador {ind.Name}: {ex.Message}");
                        string headerName = $"{ind.Name}_Value";
                        header.Add(headerName);
                        indicatorValues.Add(Tuple.Create(headerName, (Func<int, string>)(_ => string.Empty)));
                    }
                }

                // Obtener el número total de barras
                int totalBarsInChart = 0;
                
                try
                {
                    // Intentar obtener el conteo de barras desde ChartBars
                    if (ChartControl != null && ChartControl.ChartPanels[0].ChartObjects != null)
                    {
                        foreach (var obj in ChartControl.ChartPanels[0].ChartObjects)
                        {
                            if (obj is ChartBars chartBars)
                            {
                                totalBarsInChart = chartBars.Count;
                                break;
                            }
                        }
                    }
                    
                    // Si no se obtiene de ChartBars, usar Bars.Count
                    if (totalBarsInChart <= 0)
                    {
                        totalBarsInChart = Bars.Count;
                    }
                    
                    // Si aún no hay valor válido, usar BarsArray
                    if (totalBarsInChart <= 0)
                    {
                        totalBarsInChart = BarsArray[0].Count;
                    }
                    
                    // Última opción: usar un valor por defecto
                    if (totalBarsInChart <= 0)
                    {
                        totalBarsInChart = 1000;
                        Print("No se pudo determinar el número exacto de barras, usando valor por defecto: 1000");
                    }
                }
                catch (Exception ex)
                {
                    Print($"Error al obtener el total de barras: {ex.Message}");
                    totalBarsInChart = 1000; // Valor por defecto si ocurre un error
                }
                
                // Limitar al máximo especificado por el usuario
                int barsToExport = Math.Min(totalBarsInChart, MaxBarsToExport);
                
                Print($"Barras totales en el gráfico: {totalBarsInChart}, Barras a exportar: {barsToExport}");

                // Lista para almacenar las filas
                List<string> allRows = new List<string>();

                // Escribir en el archivo CSV
                using (StreamWriter writer = new StreamWriter(fullPath))
                {
                    writer.WriteLine(string.Join(",", header));

                    // Procesar barras desde la más reciente hacia atrás
                    for (int i = 0; i < barsToExport; i++)
                    {
                        int barsAgo = Math.Min(i, CurrentBar);
                        
                        if (barsAgo >= BarsArray[0].Count)
                            break;

                        List<string> row = new List<string>();
                        
                        try
                        {
                            row.Add(Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss"));
                            row.Add(Open[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(High[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(Low[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(Close[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            foreach (var valueFunc in indicatorValues)
                            {
                                string value = valueFunc.Item2(barsAgo);
                                row.Add(value);
                            }
                            
                            allRows.Add(string.Join(",", row));
                        }
                        catch (Exception ex)
                        {
                            Print($"Error al procesar barra con barsAgo={barsAgo}: {ex.Message}");
                            break;
                        }
                    }

                    // Escribir filas en orden cronológico (invertido)
                    for (int i = allRows.Count - 1; i >= 0; i--)
                    {
                        writer.WriteLine(allRows[i]);
                    }
                }

                Print($"Datos exportados: {allRows.Count} barras a {fullPath}");
            }
            catch (Exception ex)
            {
                Print($"Error al exportar datos: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Print($"Inner Exception: {ex.InnerException.Message}");
                }
                Print($"Stack Trace: {ex.StackTrace}");
            }
        }

        private string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
            {
                name = name.Replace(c, '_');
            }
            
            name = name.Replace(' ', '_')
                       .Replace(':', '_').Replace('/', '_').Replace('\\', '_')
                       .Replace('[', '_').Replace(']', '_').Replace('{', '_').Replace('}', '_')
                       .Replace('(', '_').Replace(')', '_').Replace('*', '_').Replace('?', '_');
            
            return name;
        }
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private ExportDataToCSV1[] cacheExportDataToCSV1;
		public ExportDataToCSV1 ExportDataToCSV1(bool exportToCSV, string exportPath, int maxBarsToExport)
		{
			return ExportDataToCSV1(Input, exportToCSV, exportPath, maxBarsToExport);
		}

		public ExportDataToCSV1 ExportDataToCSV1(ISeries<double> input, bool exportToCSV, string exportPath, int maxBarsToExport)
		{
			if (cacheExportDataToCSV1 != null)
				for (int idx = 0; idx < cacheExportDataToCSV1.Length; idx++)
					if (cacheExportDataToCSV1[idx] != null && cacheExportDataToCSV1[idx].ExportToCSV == exportToCSV && cacheExportDataToCSV1[idx].ExportPath == exportPath && cacheExportDataToCSV1[idx].MaxBarsToExport == maxBarsToExport && cacheExportDataToCSV1[idx].EqualsInput(input))
						return cacheExportDataToCSV1[idx];
			return CacheIndicator<ExportDataToCSV1>(new ExportDataToCSV1(){ ExportToCSV = exportToCSV, ExportPath = exportPath, MaxBarsToExport = maxBarsToExport }, input, ref cacheExportDataToCSV1);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.ExportDataToCSV1 ExportDataToCSV1(bool exportToCSV, string exportPath, int maxBarsToExport)
		{
			return indicator.ExportDataToCSV1(Input, exportToCSV, exportPath, maxBarsToExport);
		}

		public Indicators.ExportDataToCSV1 ExportDataToCSV1(ISeries<double> input , bool exportToCSV, string exportPath, int maxBarsToExport)
		{
			return indicator.ExportDataToCSV1(input, exportToCSV, exportPath, maxBarsToExport);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.ExportDataToCSV1 ExportDataToCSV1(bool exportToCSV, string exportPath, int maxBarsToExport)
		{
			return indicator.ExportDataToCSV1(Input, exportToCSV, exportPath, maxBarsToExport);
		}

		public Indicators.ExportDataToCSV1 ExportDataToCSV1(ISeries<double> input , bool exportToCSV, string exportPath, int maxBarsToExport)
		{
			return indicator.ExportDataToCSV1(input, exportToCSV, exportPath, maxBarsToExport);
		}
	}
}

#endregion
