# Python Environment Setup for AITrader

AITrader requires specific Python packages to function properly. This guide will help you set up your Python environment.

## Prerequisites

- Python 3.8+ installed on your system
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

## Setting Up the Python Environment

### Option 1: Using the Embedded Python Runtime (Recommended)

AITrader comes with Python.Included, which can automatically set up an embedded Python runtime. In most cases, you don't need to manually install Python or packages when using this option.

However, if you encounter issues with the embedded runtime, follow Option 2 below.

### Option 2: Setting Up a Manual Python Environment

1. **Create a Virtual Environment** (recommended):
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Required Packages**:
   ```
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with `talib-binary`, you may need to install TA-Lib separately:
   - Windows: Download and install from [TA-Lib Windows Binary](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Linux: `apt-get install ta-lib`
   - macOS: `brew install ta-lib`

3. **Verify Installation**:
   ```python
   python -c "import numpy; import pandas; import gymnasium; import stable_baselines3; print('All packages imported successfully!')"
   ```

## Troubleshooting

### Common Issues

1. **ImportError: DLL load failed**:
   - This usually means there's a missing dependency for one of the packages
   - For NumPy/SciPy issues on Windows, try installing the Visual C++ Redistributable

2. **No module named 'stable_baselines3'**:
   - Make sure you've installed all requirements:
     ```
     pip install stable-baselines3[extra]
     ```

3. **TA-Lib installation issues**:
   - Use pre-compiled binaries for your platform
   - For Windows: `pip install --no-cache-dir TA-Lib-binary`

4. **Gymnasium vs Gym confusion**:
   - AITrader uses both Gym and Gymnasium APIs
   - Make sure both packages are installed:
     ```
     pip install gym gymnasium
     ```

### Environment Variables

If AITrader cannot find your Python installation, you might need to set environment variables:

- Add Python to your PATH
- Set PYTHONHOME to your Python installation directory
- Ensure PYTHONPATH includes your AITrader directory

## Advanced: Creating a Custom Python Environment for AITrader

If you need specific packages or have special requirements:

1. Create a conda environment:
   ```
   conda create -n aitrader python=3.9
   conda activate aitrader
   ```

2. Install the required packages:
   ```
   conda install -c conda-forge numpy pandas matplotlib
   pip install stable-baselines3[extra] gymnasium
   ```

3. Configure AITrader to use this environment by modifying the PythonEngineService.cs file