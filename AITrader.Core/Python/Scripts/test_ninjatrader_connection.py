"""
Test NinjaTrader Connection Script

This script tests the connection between the Python client and NinjaTrader RLExecutor,
providing a simple CLI interface to send commands and receive responses.
"""

import os
import sys
import time
import argparse
import threading
import logging

# Adjust Python path to find AITrader modules
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(root_dir)

# Import the socket client
from AITrader.Core.Python.RealTime.socket_client import NinjaTraderSocketClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def on_data_received(data):
    """Callback for handling received market data"""
    logger.info(f"Received market data: {data}")


def on_order_update(data):
    """Callback for handling order updates"""
    logger.info(f"Received order update: {data}")


def on_connection_status(data_connected, order_connected):
    """Callback for handling connection status changes"""
    logger.info(f"Connection status changed: data={data_connected}, order={order_connected}")


def run_cli(client):
    """Run a simple CLI for testing commands"""
    help_text = """
    Available commands:
    
    help                             - Show this help text
    status                           - Show connection status
    signal <s> <ema> <size> <sl> <tp> - Send trading signal
                                     (s: -1=short, 0=flat, 1=long,
                                      ema: 0=none, 1=short, 2=long,
                                      size: position sizing multiplier,
                                      sl: stop loss in ticks,
                                      tp: take profit in ticks)
    exit                             - Exit the program
    """
    
    print(help_text)
    
    while True:
        try:
            # Get user input
            cmd = input("NinjaTrader> ").strip()
            
            # Process commands
            if cmd == "help":
                print(help_text)
            
            elif cmd == "status":
                data_connected, order_connected = client.is_connected()
                print(f"Connection status: data={data_connected}, order={order_connected}")
            
            elif cmd.startswith("signal"):
                # Parse signal parameters
                parts = cmd.split()
                if len(parts) != 6:
                    print("Error: Invalid signal command format")
                    print("Usage: signal <s> <ema> <size> <sl> <tp>")
                    continue
                
                try:
                    signal = int(parts[1])
                    ema_choice = int(parts[2])
                    position_size = float(parts[3])
                    stop_loss = float(parts[4])
                    take_profit = float(parts[5])
                    
                    # Validate parameters
                    if signal not in [-1, 0, 1]:
                        print("Error: Signal must be -1, 0, or 1")
                        continue
                    
                    if ema_choice not in [0, 1, 2]:
                        print("Error: EMA choice must be 0, 1, or 2")
                        continue
                    
                    if position_size <= 0 or position_size > 5:
                        print("Error: Position size must be between 0 and 5")
                        continue
                    
                    if stop_loss < 0 or take_profit < 0:
                        print("Error: Stop loss and take profit must be non-negative")
                        continue
                    
                    # Send signal
                    success = client.send_trading_signal(
                        signal=signal,
                        ema_choice=ema_choice,
                        position_size=position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if success:
                        print(f"Signal sent: {signal},{ema_choice},{position_size},{stop_loss},{take_profit}")
                    else:
                        print("Error sending signal")
                    
                except ValueError:
                    print("Error: Invalid numeric parameters")
            
            elif cmd == "exit":
                break
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Exiting CLI...")


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test NinjaTrader connection")
    parser.add_argument("--data-host", type=str, default="127.0.0.1", help="Host for data socket")
    parser.add_argument("--data-port", type=int, default=5000, help="Port for data socket")
    parser.add_argument("--order-host", type=str, default="127.0.0.1", help="Host for order socket")
    parser.add_argument("--order-port", type=int, default=5001, help="Port for order socket")
    args = parser.parse_args()
    
    # Create client
    client = NinjaTraderSocketClient(
        data_host=args.data_host,
        data_port=args.data_port,
        order_host=args.order_host,
        order_port=args.order_port
    )
    
    # Register callbacks
    client.register_data_callback(on_data_received)
    client.register_order_callback(on_order_update)
    client.register_connection_callback(on_connection_status)
    
    try:
        # Start client
        logger.info("Starting socket client...")
        success = client.start()
        
        if not success:
            logger.error("Failed to start socket client")
            return
        
        logger.info("Socket client started")
        
        # Run CLI
        run_cli(client)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Stop client
        logger.info("Stopping socket client...")
        client.stop()
        logger.info("Socket client stopped")


if __name__ == "__main__":
    main()