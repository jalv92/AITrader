import subprocess
import sys
import os

def main():
    print("Setting up Python environment for AITrader...")
    
    # Get current Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # List of required packages
    required_packages = [
        "numpy",
        "pandas",
        "gymnasium",
        "stable-baselines3",
        "matplotlib"
    ]
    
    # Install required packages
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Verify installation
    print("\nVerifying installations:")
    for package in required_packages:
        try:
            if package == "stable-baselines3":
                module_name = "stable_baselines3"
            else:
                module_name = package
                
            module = __import__(module_name)
            if hasattr(module, "__version__"):
                print(f"{package}: {module.__version__}")
            else:
                print(f"{package}: installed (no version info)")
        except ImportError as e:
            print(f"{package}: NOT INSTALLED - {str(e)}")
    
    print("\nSetup completed.")

if __name__ == "__main__":
    main()
