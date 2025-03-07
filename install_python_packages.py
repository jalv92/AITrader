import sys
import subprocess
import os

def main():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Path to the Python executable being used
    python_exe = sys.executable
    
    # Required packages
    required_packages = [
        "numpy",
        "pandas",
        "gymnasium",
        "stable-baselines3",
        "matplotlib",
        "torch"  # Add torch explicitly as it's needed by stable-baselines3
    ]
    
    # Install packages
    for package in required_packages:
        print(f"\nInstalling {package}...")
        try:
            # Use subprocess to run pip install
            subprocess.check_call([python_exe, "-m", "pip", "install", "--user", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    # Verify installations
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
    
    print("\nSetup completed. You can now run the AITrader application.")

if __name__ == "__main__":
    main()
