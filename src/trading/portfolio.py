"""
Portfolio Manager - Tracks portfolio state, positions, and cash
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages trading portfolio with cash and stock positions"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.0):
        """
        Initialize Portfolio
        
        Args:
            initial_capital: Starting cash amount
            commission_rate: Trading commission as percentage (e.g., 0.1 for 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate / 100.0  # Convert to decimal
        
        # Positions: {symbol: quantity}
        self.positions: Dict[str, int] = {}
        
        # Transaction history
        self.transactions: List[Dict] = []
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.peak_portfolio_value = initial_capital
        
        logger.info(f"Portfolio initialized with ₹{initial_capital:.2f} capital")
    
    def get_position(self, symbol: str) -> int:
        """Get current position quantity for a symbol"""
        return self.positions.get(symbol, 0)
    
    def get_cash(self) -> float:
        """Get current cash balance"""
        return self.cash
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_prices: Dictionary of {symbol: current_price}
        
        Returns:
            Total portfolio value (cash + positions)
        """
        position_value = sum(
            qty * current_prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def get_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Get total value of all positions"""
        return sum(
            qty * current_prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
    
    def can_buy(self, symbol: str, quantity: int, price: float) -> tuple[bool, str]:
        """
        Check if a buy order can be executed
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
        
        Returns:
            (can_execute, reason)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if not isinstance(quantity, int):
            return False, "Quantity must be a whole number (no fractional shares)"
        
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if total_cost > self.cash:
            return False, f"Insufficient funds. Need ₹{total_cost:.2f}, have ₹{self.cash:.2f}"
        
        return True, "OK"
    
    def can_sell(self, symbol: str, quantity: int) -> tuple[bool, str]:
        """
        Check if a sell order can be executed
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
        
        Returns:
            (can_execute, reason)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if not isinstance(quantity, int):
            return False, "Quantity must be a whole number (no fractional shares)"
        
        current_position = self.get_position(symbol)
        if quantity > current_position:
            return False, f"Insufficient shares. Trying to sell {quantity}, have {current_position}"
        
        return True, "OK"
    
    def execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        reason: str = ""
    ) -> Dict:
        """
        Execute a buy order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            timestamp: Transaction timestamp
            reason: Reason for the trade
        
        Returns:
            Transaction record
        """
        can_execute, message = self.can_buy(symbol, quantity, price)
        if not can_execute:
            raise ValueError(f"Cannot execute buy: {message}")
        
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        # Update portfolio
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        self.total_commission_paid += commission
        
        # Record transaction
        transaction = {
            "timestamp": timestamp,
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "cost": cost,
            "commission": commission,
            "total_cost": total_cost,
            "cash_after": self.cash,
            "reason": reason
        }
        self.transactions.append(transaction)
        
        logger.info(
            f"BUY: {quantity} shares of {symbol} @ ₹{price:.2f} "
            f"(Commission: ₹{commission:.2f}, Total: ₹{total_cost:.2f})"
        )
        
        return transaction
    
    def execute_sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        reason: str = ""
    ) -> Dict:
        """
        Execute a sell order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            timestamp: Transaction timestamp
            reason: Reason for the trade
        
        Returns:
            Transaction record
        """
        can_execute, message = self.can_sell(symbol, quantity)
        if not can_execute:
            raise ValueError(f"Cannot execute sell: {message}")
        
        revenue = quantity * price
        commission = revenue * self.commission_rate
        net_revenue = revenue - commission
        
        # Update portfolio
        self.cash += net_revenue
        self.positions[symbol] -= quantity
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        self.total_commission_paid += commission
        
        # Record transaction
        transaction = {
            "timestamp": timestamp,
            "type": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "revenue": revenue,
            "commission": commission,
            "net_revenue": net_revenue,
            "cash_after": self.cash,
            "reason": reason
        }
        self.transactions.append(transaction)
        
        logger.info(
            f"SELL: {quantity} shares of {symbol} @ ₹{price:.2f} "
            f"(Commission: ₹{commission:.2f}, Net: ₹{net_revenue:.2f})"
        )
        
        return transaction
    
    def get_transactions(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get transaction history
        
        Args:
            limit: Maximum number of recent transactions to return
        
        Returns:
            List of transaction records
        """
        if limit:
            return self.transactions[-limit:]
        return self.transactions.copy()
    
    def get_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get portfolio summary
        
        Args:
            current_prices: Dictionary of {symbol: current_price}
        
        Returns:
            Dictionary with portfolio summary
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        positions_value = self.get_positions_value(current_prices)
        total_return = portfolio_value - self.initial_capital
        return_pct = (total_return / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        # Update peak value for drawdown calculation
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        drawdown = ((self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100) if self.peak_portfolio_value > 0 else 0
        
        return {
            "initial_capital": self.initial_capital,
            "current_cash": self.cash,
            "positions_value": positions_value,
            "total_value": portfolio_value,
            "total_return": total_return,
            "return_percentage": return_pct,
            "total_commission_paid": self.total_commission_paid,
            "num_transactions": len(self.transactions),
            "current_positions": self.positions.copy(),
            "peak_value": self.peak_portfolio_value,
            "drawdown_percentage": drawdown
        }
