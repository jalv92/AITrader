using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using AITrader.Core.Services.Python;
using AITrader.Core.Services.RealTimeTrading;

namespace AITrader.Core.Services
{
    /// <summary>
    /// Service configuration for AITrader
    /// </summary>
    public static class ServiceConfiguration
    {
        /// <summary>
        /// Register all services in the dependency injection container
        /// </summary>
        public static IServiceCollection RegisterServices(this IServiceCollection services)
        {
            // Register Python engine service
            services.AddSingleton<IPythonEngineService, PythonEngineService>();
            
            // Register real-time trading service
            services.AddSingleton<RealTimeTradingService>();
            
            // Register other services
            // ...
            
            return services;
        }
    }
}