using System.Threading.Tasks;

namespace AITrader.Core.Services.Python
{
    /// <summary>
    /// Interface for Python engine service
    /// </summary>
    public interface IPythonEngineService
    {
        /// <summary>
        /// Initialize the Python engine
        /// </summary>
        Task<bool> InitializeAsync();
        
        /// <summary>
        /// Check if Python engine is initialized
        /// </summary>
        bool IsInitialized { get; }
    }
}