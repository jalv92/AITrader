# NinjaTrader 8 Setup Guide for AITrader

This guide explains how to set up the required components in NinjaTrader 8 to work with the AITrader application.

## Installing the RLExecutor Strategy

1. **Locate the NinjaTrader User Files directory**:
   - Open NinjaTrader 8
   - Go to the Control Center
   - Click on `Help` > `About NinjaTrader`
   - Note the `User Files:` directory path

2. **Copy the RLExecutor strategy**:
   - Navigate to the `integrations` folder in the AITrader project
   - Locate the `RLExecutor.cs` file
   - Copy this file to the NinjaTrader `Documents\NinjaTrader 8\bin\Custom\Strategies` directory

3. **Import the strategy into NinjaTrader**:
   - In NinjaTrader, go to the Control Center
   - Click on `Tools` > `Import` > `NinjaScript Add-On...`
   - Navigate to where you copied the `RLExecutor.cs` file
   - Select the file and click `Open`
   - Select `Strategy` when prompted for the type
   - Click `OK` to import

4. **Compile the strategy**:
   - In NinjaTrader, go to the Control Center
   - Click on `New` > `NinjaScript Editor`
   - Press F5 or click the Build button (hammer icon) to compile all NinjaScript files

## Applying the RLExecutor Strategy to a Chart

1. **Open a chart**:
   - In NinjaTrader, go to the Control Center
   - Click on `New` > `Chart`
   - Select your desired instrument and timeframe
   - Click `OK`

2. **Add the RLExecutor strategy**:
   - Right-click on the chart
   - Select `Strategies...`
   - In the list of available strategies, find and select `RLExecutor`
   - Configure the settings as needed:
     - Set `Server IP` to `127.0.0.1`
     - Set `Data Port` to `5000`
     - Set `Order Port` to `5001`
     - Set `Base Position Size` to your desired position size
   - Click `OK` to apply the strategy

## Testing the Connection

To test if NinjaTrader is properly configured, you can use the provided test script:

1. **Start NinjaTrader** with the RLExecutor strategy applied to a chart
2. **Open a command prompt** and navigate to the AITrader project directory
3. **Run the test script**:
   ```
   python AITrader.Core/Python/Scripts/test_connection.py
   ```
4. The script should connect to NinjaTrader and you should see messages in both NinjaTrader's output window and the command prompt indicating successful connection.

## Troubleshooting

If you encounter connection issues:

1. **Check ports**: Make sure the ports (5000 and 5001) are not being used by other applications
2. **Check firewall**: Ensure your firewall is not blocking the connections
3. **Check NinjaTrader logs**: In NinjaTrader, go to the Control Center and check the Logs tab for any errors
4. **Verify strategy is running**: Make sure the RLExecutor strategy is actually running on your chart
5. **Check strategy parameters**: Ensure the IP address and port settings match between NinjaTrader and AITrader

## Next Steps

After successfully setting up NinjaTrader and confirming the connection works, you can proceed to run the AITrader application which will connect to NinjaTrader for real-time trading.