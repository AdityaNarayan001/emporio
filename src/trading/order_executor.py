"""
Order Executor - Executes trading orders and manages order flow
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Executes trading orders and validates them"""
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize OrderExecutor
        
        Args:
            portfolio: Portfolio instance to execute orders against
        """
        self.portfolio = portfolio
        self.pending_orders = []
        
    def execute_market_buy(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        timestamp: datetime,
        reason: str = ""
    ) -> Dict:
        """
        Execute a market buy order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (must be whole number)
            current_price: Current market price
            timestamp: Order timestamp
            reason: Reason for the trade
        
        Returns:
            Transaction result
        """
        try:
            # Validate quantity is whole number
            if not isinstance(quantity, int):
                quantity = int(quantity)
                logger.warning(f"Quantity {quantity} converted to whole number: {int(quantity)}")
            
            # Check if order can be executed
            can_execute, message = self.portfolio.can_buy(symbol, quantity, current_price)
            
            if not can_execute:
                logger.warning(f"Buy order rejected: {message}")
                return {
                    "success": False,
                    "reason": message,
                    "timestamp": timestamp
                }
            
            # Execute the buy
            transaction = self.portfolio.execute_buy(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                timestamp=timestamp,
                reason=reason
            )
            
            return {
                "success": True,
                "transaction": transaction,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error executing buy order: {str(e)}")
            return {
                "success": False,
                "reason": str(e),
                "timestamp": timestamp
            }
    
    def execute_market_sell(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        timestamp: datetime,
        reason: str = ""
    ) -> Dict:
        """
        Execute a market sell order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (must be whole number)
            current_price: Current market price
            timestamp: Order timestamp
            reason: Reason for the trade
        
        Returns:
            Transaction result
        """
        try:
            # Validate quantity is whole number
            if not isinstance(quantity, int):
                quantity = int(quantity)
                logger.warning(f"Quantity {quantity} converted to whole number: {int(quantity)}")
            
            # Check if order can be executed
            can_execute, message = self.portfolio.can_sell(symbol, quantity)
            
            if not can_execute:
                logger.warning(f"Sell order rejected: {message}")
                return {
                    "success": False,
                    "reason": message,
                    "timestamp": timestamp
                }
            
            # Execute the sell
            transaction = self.portfolio.execute_sell(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                timestamp=timestamp,
                reason=reason
            )
            
            return {
                "success": True,
                "transaction": transaction,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error executing sell order: {str(e)}")
            return {
                "success": False,
                "reason": str(e),
                "timestamp": timestamp
            }
    
    def get_max_affordable_quantity(self, symbol: str, price: float) -> int:
        """
        Calculate maximum affordable quantity at given price
        
        Args:
            symbol: Stock symbol
            price: Price per share
        
        Returns:
            Maximum whole shares that can be bought
        """
        available_cash = self.portfolio.get_cash()
        commission_rate = self.portfolio.commission_rate
        
        # Calculate max quantity considering commission
        # total_cost = quantity * price * (1 + commission_rate)
        # available_cash >= total_cost
        # quantity <= available_cash / (price * (1 + commission_rate))
        
        max_quantity = int(available_cash / (price * (1 + commission_rate)))
        return max_quantity
    
    def validate_order(self, order_type: str, symbol: str, quantity: int, price: float) -> tuple[bool, str]:
        """
        Validate an order before execution
        
        Args:
            order_type: "BUY" or "SELL"
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
        
        Returns:
            (is_valid, reason)
        """
        # Check quantity is whole number
        if not isinstance(quantity, int) or quantity <= 0:
            return False, "Quantity must be a positive whole number"
        
        if order_type.upper() == "BUY":
            return self.portfolio.can_buy(symbol, quantity, price)
        elif order_type.upper() == "SELL":
            return self.portfolio.can_sell(symbol, quantity)
        else:
            return False, f"Invalid order type: {order_type}"
