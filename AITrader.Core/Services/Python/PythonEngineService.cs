using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Python.Included;
using Python.Runtime;

namespace AITrader.Core.Services.Python
{
    /// <summary>
    /// Service for managing the Python engine
    /// </summary>
    public class PythonEngineService : IPythonEngineService
    {
        private readonly ILogger<PythonEngineService> _logger;
        private static bool _isInitialized = false;
        private static readonly object _initLock = new object();

        public bool IsInitialized => _isInitialized;

        /// <summary>
        /// Constructor
        /// </summary>
        public PythonEngineService(ILogger<PythonEngineService> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Initialize the Python engine
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            // Si ya está inicializado, simplemente devuelve true
            if (_isInitialized)
            {
                _logger.LogDebug("Python engine already initialized, skipping initialization");
                return true;
            }

            // Protect initialization but allow await outside the lock
            bool needsSetup = false;
            string installPath = string.Empty;
            
            // Adquiere un bloqueo para asegurar que solo una instancia inicialice Python a nivel estático
            // Esto evita que múltiples instancias intenten inicializar Python en paralelo
            lock (_initLock)
            {
                // Doble verificación para evitar inicialización múltiple
                if (_isInitialized)
                {
                    _logger.LogDebug("Python engine already initialized by another thread, skipping initialization");
                    return true;
                }

                try
                {
                    _logger.LogInformation("Initializing Python engine...");

                    // Get the current assembly directory
                    string assemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                    
                    // Define Python paths to check in order of preference
                    string[] pythonPathsToCheck = new string[] 
                    {
                        // Python 3.11
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python311"),
                        // Python 3.11 - Alternate location
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Python311"),
                        // Python 3.10
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python310"),
                        // Embedded Python from Python.Included
                        Path.Combine(assemblyDir, "python-runtime")
                    };
                    
                    bool pythonFound = false;
                    
                    foreach (string pythonPath in pythonPathsToCheck)
                    {
                        string pythonDllName = pythonPath.Contains("310") ? "python310.dll" : "python311.dll";
                        string pythonDllPath = Path.Combine(pythonPath, pythonDllName);
                        
                        if (Directory.Exists(pythonPath) && File.Exists(pythonDllPath))
                        {
                            _logger.LogInformation($"Found Python installation at: {pythonPath}");
                            
                            // Verificar si el runtime ya está inicializado antes de configurar PythonDLL
                            if (!PythonEngine.IsInitialized)
                            {
                                Runtime.PythonDLL = pythonDllPath;
                            }
                            else
                            {
                                _logger.LogWarning("Python runtime already initialized, cannot set PythonDLL");
                            }
                            
                            // Add Python site-packages to path
                            string pythonSitePackages = Path.Combine(pythonPath, "Lib", "site-packages");
                            Environment.SetEnvironmentVariable("PYTHONPATH", 
                                $"{pythonSitePackages}{Path.PathSeparator}{Environment.GetEnvironmentVariable("PYTHONPATH")}");
                            
                            // For user site-packages (where pip installs packages for the current user)
                            string userSitePackages = Path.Combine(
                                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                                "AppData", "Roaming", "Python", 
                                pythonPath.Contains("310") ? "Python310" : "Python311", 
                                "site-packages");
                                
                            if (Directory.Exists(userSitePackages))
                            {
                                Environment.SetEnvironmentVariable("PYTHONPATH", 
                                    $"{userSitePackages}{Path.PathSeparator}{Environment.GetEnvironmentVariable("PYTHONPATH")}");
                                _logger.LogInformation($"Added user site-packages: {userSitePackages}");
                            }
                            
                            pythonFound = true;
                            break;
                        }
                    }
                    
                    // Add Python scripts directory to PYTHONPATH
                    string pythonScriptsDir = Path.Combine(assemblyDir, "Python");
                    Environment.SetEnvironmentVariable("PYTHONPATH", 
                        $"{pythonScriptsDir}{Path.PathSeparator}{Environment.GetEnvironmentVariable("PYTHONPATH")}");
                    
                    _logger.LogInformation($"Python scripts directory: {pythonScriptsDir}");
                    _logger.LogInformation($"PYTHONPATH: {Environment.GetEnvironmentVariable("PYTHONPATH")}");

                    if (!pythonFound)
                    {
                        // Flag that we need to install Python outside the lock
                        needsSetup = true;
                        installPath = Path.Combine(assemblyDir, "python-runtime");
                        Installer.InstallPath = installPath;
                        Installer.LogMessage += message => _logger.LogInformation(message);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error initializing Python engine");
                    return false;
                }
            }
            
            // If needed, run Python setup outside the lock
            if (needsSetup)
            {
                try
                {
                    _logger.LogInformation("No Python installation found, installing embedded Python runtime...");
                    _logger.LogInformation("Installing Python runtime...");
                    await Installer.SetupPython();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error installing Python runtime");
                    return false;
                }
            }
            
            // Continue with initialization
            try
            {
                // Initialize PythonEngine if not already initialized
                if (!PythonEngine.IsInitialized)
                {
                    PythonEngine.Initialize();
                    _logger.LogInformation("Python engine initialized");
                }
                else
                {
                    _logger.LogInformation("Python engine already initialized by another component");
                }
                
                // Set PYTHONHOME to ensure modules can be found
                using (Py.GIL())
                {
                    dynamic sys = Py.Import("sys");
                    _logger.LogInformation($"Python version: {sys.version}");
                    _logger.LogInformation($"Python executable: {sys.executable}");
                    
                    // Convertir sys.path a lista de strings para evitar problemas con tipos dinámicos
                    var pathList = new System.Collections.Generic.List<string>();
                    foreach (var path in sys.path)
                    {
                        pathList.Add(path.ToString());
                    }
                    _logger.LogInformation($"Python path: {string.Join(", ", pathList)}");
                
                    // Import required modules to verify they are available
                    try
                    {
                        // Try importing key modules
                        dynamic np = Py.Import("numpy");
                        string npVersion = np.__version__.ToString();
                        _logger.LogInformation($"Numpy version: {npVersion}");
                        
                        try
                        {
                            dynamic pd = Py.Import("pandas");
                            string pdVersion = pd.__version__.ToString();
                            _logger.LogInformation($"Pandas version: {pdVersion}");
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning($"Pandas not available: {ex.Message}");
                        }
                        
                        try
                        {
                            dynamic sb3 = Py.Import("stable_baselines3");
                            _logger.LogInformation("Stable-Baselines3 available");
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning($"Stable-Baselines3 not available: {ex.Message}");
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error importing Python modules");
                        return false;
                    }
                }
                
                lock (_initLock)
                {
                    _isInitialized = true;
                }
                _logger.LogInformation("Python engine initialized successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing Python engine");
                return false;
            }
        }
    }
}