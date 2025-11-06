"""
Data Simulator - Simulates real-time data feed from historical data
"""
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Callable
import logging
import threading

logger = logging.getLogger(__name__)


class DataSimulator:
    """
    Simulates real-time market data by replaying historical data.
    Prevents lookahead bias by only exposing past data up to current simulation time.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        speed_multiplier: float = 1.0,
        max_lookback: str = "5d",
        tick_mode: bool = True,
        tick_interval: float = 2.0
    ):
        """
        Initialize DataSimulator
        
        Args:
            historical_data: DataFrame with historical OHLCV data
            speed_multiplier: Simulation speed (1.0 = real-time, 2.0 = 2x, etc.) - used in time-based mode
            max_lookback: Maximum historical data LLM can access (e.g., "5d", "2h", "30m")
            tick_mode: If True, advances one data point per tick_interval (better for daily data)
            tick_interval: Seconds between ticks in tick mode (e.g., 2.0 = new data every 2 seconds)
        """
        self.historical_data = historical_data.copy()
        self.historical_data.sort_values('Datetime', inplace=True)
        self.historical_data.reset_index(drop=True, inplace=True)
        
        self.speed_multiplier = speed_multiplier
        self.max_lookback = self._parse_lookback(max_lookback)
        self.tick_mode = tick_mode
        self.tick_interval = tick_interval
        
        # Simulation state
        self.current_index = 0
        self.simulation_start_time = None
        self.real_start_time = None
        self.is_running = False
        self.is_paused = False
        self.last_tick_time = None
        
        # Callbacks
        self.on_new_data_callbacks = []
        
        mode_desc = f"tick mode ({tick_interval}s per point)" if tick_mode else f"time-based mode (speed {speed_multiplier}x)"
        logger.info(
            f"DataSimulator initialized with {len(self.historical_data)} data points, "
            f"{mode_desc}, max lookback: {max_lookback}"
        )
    
    def _parse_lookback(self, lookback_str: str) -> timedelta:
        """Parse lookback string like '5d', '2h', '30m' into timedelta"""
        unit = lookback_str[-1]
        value = int(lookback_str[:-1])
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        else:
            raise ValueError(f"Invalid lookback format: {lookback_str}")
    
    def start(self):
        """Start the simulation"""
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.is_paused = False
        self.simulation_start_time = self.historical_data['Datetime'].iloc[0]
        self.real_start_time = datetime.now()
        self.last_tick_time = datetime.now()
        self.current_index = 0
        
        mode_desc = f"tick mode ({self.tick_interval}s intervals)" if self.tick_mode else f"{self.speed_multiplier}x speed"
        logger.info(f"Simulation started at {self.simulation_start_time} in {mode_desc}")
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
    
    def _run_simulation(self):
        """Main simulation loop (runs in background thread)"""
        logger.info(f"DATA SIMULATOR: Starting simulation loop with {len(self.historical_data)} data points")
        
        if self.tick_mode:
            # Tick-based mode: advance one data point per tick interval
            logger.info(f"DATA SIMULATOR: Using TICK MODE - new data every {self.tick_interval}s")
            while self.is_running and self.current_index < len(self.historical_data):
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Check if it's time for next tick
                time_since_last_tick = (datetime.now() - self.last_tick_time).total_seconds()
                
                if time_since_last_tick >= self.tick_interval:
                    # Advance to next data point
                    new_data = self.historical_data.iloc[self.current_index].to_dict()
                    data_time = self.historical_data['Datetime'].iloc[self.current_index]
                    logger.info(f"DATA SIMULATOR: ⏰ TICK {self.current_index + 1}/{len(self.historical_data)} - Price: ₹{new_data.get('Close', 0):.2f} (simulated time: {data_time})")
                    self._notify_new_data(new_data)
                    self.current_index += 1
                    self.last_tick_time = datetime.now()
                else:
                    # Wait for next tick
                    time.sleep(0.1)
        else:
            # Time-based mode: respect actual timestamps and speed multiplier
            logger.info(f"DATA SIMULATOR: Using TIME-BASED MODE at {self.speed_multiplier}x speed")
            while self.is_running and self.current_index < len(self.historical_data):
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Calculate current simulation time based on elapsed real time
                real_elapsed = (datetime.now() - self.real_start_time).total_seconds()
                sim_elapsed_seconds = real_elapsed * self.speed_multiplier
                current_sim_time = self.simulation_start_time + timedelta(seconds=sim_elapsed_seconds)
                
                # Find the latest data point that should be available
                next_data_time = self.historical_data['Datetime'].iloc[self.current_index]
                
                if current_sim_time >= next_data_time:
                    # New data point is available
                    new_data = self.historical_data.iloc[self.current_index].to_dict()
                    logger.info(f"DATA SIMULATOR: Advancing to point {self.current_index + 1}/{len(self.historical_data)} - Price: ₹{new_data.get('Close', 0):.2f} at {next_data_time}")
                    self._notify_new_data(new_data)
                    self.current_index += 1
                else:
                    # Wait until next data point
                    sleep_time = 0.1 / self.speed_multiplier  # Check frequently
                    time.sleep(sleep_time)
        
        if self.current_index >= len(self.historical_data):
            logger.info("DATA SIMULATOR: ✅ Simulation completed - all data consumed")
            self.is_running = False
        else:
            logger.info(f"DATA SIMULATOR: ⏸️ Simulation stopped at point {self.current_index}/{len(self.historical_data)}")
    
    def pause(self):
        """Pause the simulation"""
        self.is_paused = True
        logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation"""
        if not self.is_running:
            logger.warning("Cannot resume - simulation not started")
            return
        
        self.is_paused = False
        # Adjust real start time to account for pause duration
        real_elapsed = (datetime.now() - self.real_start_time).total_seconds()
        sim_elapsed_seconds = real_elapsed * self.speed_multiplier
        pause_compensation = datetime.now() - self.real_start_time - timedelta(seconds=sim_elapsed_seconds / self.speed_multiplier)
        self.real_start_time += pause_compensation
        
        logger.info("Simulation resumed")
    
    def stop(self):
        """Stop the simulation"""
        self.is_running = False
        logger.info("Simulation stopped")
    
    def set_speed(self, speed_multiplier: float):
        """
        Change simulation speed
        
        Args:
            speed_multiplier: New speed (1.0 = real-time, 2.0 = 2x, etc.)
        """
        if not self.is_running:
            self.speed_multiplier = speed_multiplier
            logger.info(f"Speed set to {speed_multiplier}x")
            return
        
        # Adjust real start time to maintain simulation continuity
        real_elapsed = (datetime.now() - self.real_start_time).total_seconds()
        sim_elapsed_seconds = real_elapsed * self.speed_multiplier
        
        self.speed_multiplier = speed_multiplier
        self.real_start_time = datetime.now() - timedelta(seconds=sim_elapsed_seconds / speed_multiplier)
        
        logger.info(f"Speed changed to {speed_multiplier}x")
    
    def get_current_time(self) -> Optional[datetime]:
        """Get current simulation time"""
        if not self.is_running or self.current_index == 0:
            return None
        
        real_elapsed = (datetime.now() - self.real_start_time).total_seconds()
        sim_elapsed_seconds = real_elapsed * self.speed_multiplier
        return self.simulation_start_time + timedelta(seconds=sim_elapsed_seconds)
    
    def get_current_data(self) -> Optional[dict]:
        """Get the most recent data point"""
        if self.current_index == 0:
            return None
        return self.historical_data.iloc[self.current_index - 1].to_dict()
    
    def get_available_data(self, lookback: Optional[str] = None) -> pd.DataFrame:
        """
        Get all data available up to current simulation time
        
        Args:
            lookback: Optional lookback period (e.g., "1h", "30m"). 
                     If None, uses max_lookback from initialization
        
        Returns:
            DataFrame with available historical data
        """
        if self.current_index == 0:
            return pd.DataFrame()
        
        # Get all data up to current point
        available_data = self.historical_data.iloc[:self.current_index].copy()
        
        # Apply lookback limit
        lookback_period = self._parse_lookback(lookback) if lookback else self.max_lookback
        current_time = available_data['Datetime'].iloc[-1]
        cutoff_time = current_time - lookback_period
        
        filtered_data = available_data[available_data['Datetime'] >= cutoff_time]
        
        return filtered_data
    
    def get_progress(self) -> dict:
        """
        Get simulation progress information
        
        Returns:
            Dictionary with progress details
        """
        total_points = len(self.historical_data)
        progress_pct = (self.current_index / total_points * 100) if total_points > 0 else 0
        
        return {
            "current_index": self.current_index,
            "total_points": total_points,
            "progress_percentage": progress_pct,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_time": self.get_current_time(),
            "speed_multiplier": self.speed_multiplier
        }
    
    def register_callback(self, callback: Callable):
        """
        Register a callback to be called when new data arrives
        
        Args:
            callback: Function to call with new data dict
        """
        self.on_new_data_callbacks.append(callback)
    
    def _notify_new_data(self, data: dict):
        """Notify all registered callbacks of new data"""
        for callback in self.on_new_data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")
    
    def reset(self):
        """Reset simulation to beginning"""
        self.stop()
        self.current_index = 0
        self.simulation_start_time = None
        self.real_start_time = None
        logger.info("Simulation reset")
