"""
Test NinjaTrader Connection Script

Simple script to test the connection to NinjaTrader 8 RLExecutor strategy.
"""

import os
import sys
import socket
import time
import logging
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocketTester:
    def __init__(self, host="127.0.0.1", data_port=5000, order_port=5001):
        self.host = host
        self.data_port = data_port
        self.order_port = order_port
        self.data_socket = None
        self.order_socket = None
        self.running = False
        self.data_thread = None
        self.order_thread = None
    
    def start(self):
        """Start the socket tester"""
        self.running = True
        
        # Connect to data socket
        logger.info(f"Connecting to data socket at {self.host}:{self.data_port}...")
        try:
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.connect((self.host, self.data_port))
            logger.info("Connected to data socket")
            
            # Start data thread
            self.data_thread = Thread(target=self._data_loop, daemon=True)
            self.data_thread.start()
        except Exception as e:
            logger.error(f"Failed to connect to data socket: {e}")
        
        # Connect to order socket
        logger.info(f"Connecting to order socket at {self.host}:{self.order_port}...")
        try:
            self.order_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.order_socket.connect((self.host, self.order_port))
            logger.info("Connected to order socket")
            
            # Start order thread
            self.order_thread = Thread(target=self._order_loop, daemon=True)
            self.order_thread.start()
        except Exception as e:
            logger.error(f"Failed to connect to order socket: {e}")
    
    def stop(self):
        """Stop the socket tester"""
        self.running = False
        
        # Close sockets
        if self.data_socket:
            self.data_socket.close()
        if self.order_socket:
            self.order_socket.close()
        
        # Wait for threads to terminate
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(2)
        if self.order_thread and self.order_thread.is_alive():
            self.order_thread.join(2)
        
        logger.info("Socket tester stopped")
    
    def send_trading_signal(self, signal, ema_choice, position_size, stop_loss, take_profit):
        """Send a trading signal to NinjaTrader"""
        if not self.order_socket:
            logger.error("Order socket not connected, cannot send signal")
            return False
        
        try:
            # Format the message as expected by RLExecutor
            message = f"{signal},{ema_choice},{position_size},{stop_loss},{take_profit}\n"
            
            # Send the message
            self.order_socket.sendall(message.encode('ascii'))
            
            logger.info(f"Sent trading signal: {message.strip()}")
            return True
        except Exception as e:
            logger.error(f"Error sending trading signal: {e}")
            return False
    
    def _data_loop(self):
        """Main loop for the data socket thread"""
        buffer = ""
        
        while self.running and self.data_socket:
            try:
                # Receive data
                data = self.data_socket.recv(4096)
                
                if not data:
                    logger.warning("Data socket connection closed by server")
                    break
                
                # Decode data
                buffer += data.decode('ascii')
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    
                    # Process the line
                    self._process_data_message(line)
                    
            except Exception as e:
                logger.error(f"Error in data loop: {e}")
                break
        
        logger.info("Data loop terminated")
    
    def _order_loop(self):
        """Main loop for the order socket thread"""
        buffer = ""
        
        while self.running and self.order_socket:
            try:
                # Receive data
                data = self.order_socket.recv(4096)
                
                if not data:
                    logger.warning("Order socket connection closed by server")
                    break
                
                # Decode data
                buffer += data.decode('ascii')
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    
                    # Process the line
                    self._process_order_message(line)
                    
            except Exception as e:
                logger.error(f"Error in order loop: {e}")
                break
        
        logger.info("Order loop terminated")
    
    def _process_data_message(self, message):
        """Process a message from the data socket"""
        logger.info(f"Data: {message}")
        
        # Handle server ready message
        if message == "SERVER_READY":
            logger.info("Data server is ready, sending connection info")
            self.data_socket.sendall("CONNECT:PythonRLAgent\n".encode('ascii'))
        
        # Handle heartbeat
        if message == "PING":
            logger.info("Received PING, sending PONG")
            self.data_socket.sendall("PONG\n".encode('ascii'))
    
    def _process_order_message(self, message):
        """Process a message from the order socket"""
        logger.info(f"Order: {message}")
        
        # Handle server ready message
        if message == "ORDER_SERVER_READY":
            logger.info("Order server is ready")
        
        # Handle heartbeat
        if message == "PING":
            logger.info("Received PING, sending PONG")
            self.order_socket.sendall("PONG\n".encode('ascii'))

def main():
    try:
        # Create socket tester
        tester = SocketTester()
        
        # Start socket tester
        tester.start()
        
        # Wait for user to press Enter
        logger.info("Press Enter to send a test trading signal (Long, 1 contract)...")
        input()
        
        # Send a test trading signal
        # signal: 1 (Long), ema_choice: 0 (None), position_size: 1.0, stop_loss: 10, take_profit: 20
        tester.send_trading_signal(1, 0, 1.0, 10, 20)
        
        # Wait for user to press Enter to exit
        logger.info("Press Enter to exit...")
        input()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop socket tester
        if 'tester' in locals():
            tester.stop()

if __name__ == "__main__":
    main()