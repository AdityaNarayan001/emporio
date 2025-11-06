"""
Memory Bank - JSON-based evolving memory for LLM trading decisions
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryBank:
    """
    Manages LLM's memory of past decisions, reasoning, and learnings.
    Stores data in an evolving JSON structure.
    """
    
    def __init__(self, file_path: str = "data/memory_bank.json", max_entries: int = 1000):
        """
        Initialize MemoryBank
        
        Args:
            file_path: Path to JSON file for storage
            max_entries: Maximum entries before archiving old ones
        """
        self.file_path = file_path
        self.max_entries = max_entries
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Load or initialize memory
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict:
        """Load memory from JSON file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    memory = json.load(f)
                logger.info(f"Loaded memory bank with {len(memory.get('decisions', []))} decisions")
                return memory
            except Exception as e:
                logger.error(f"Error loading memory bank: {str(e)}")
                return self._create_empty_memory()
        else:
            logger.info("Creating new memory bank")
            return self._create_empty_memory()
    
    def _create_empty_memory(self) -> Dict:
        """Create empty memory structure"""
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            },
            "decisions": [],
            "learnings": [],
            "patterns": [],
            "mistakes": [],
            "successful_strategies": []
        }
    
    def save(self):
        """Save memory to JSON file"""
        try:
            self.memory["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Archive old decisions if exceeding max_entries
            if len(self.memory["decisions"]) > self.max_entries:
                self._archive_old_entries()
            
            with open(self.file_path, 'w') as f:
                json.dump(self.memory, f, indent=2, default=str)
            
            logger.debug("Memory bank saved")
        except Exception as e:
            logger.error(f"Error saving memory bank: {str(e)}")
    
    def _archive_old_entries(self):
        """Archive old decision entries"""
        archive_file = self.file_path.replace('.json', f'_archive_{datetime.now().strftime("%Y%m%d")}.json')
        old_decisions = self.memory["decisions"][:-self.max_entries]
        
        archive_data = {
            "archived_at": datetime.now().isoformat(),
            "decisions": old_decisions
        }
        
        with open(archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2, default=str)
        
        self.memory["decisions"] = self.memory["decisions"][-self.max_entries:]
        logger.info(f"Archived {len(old_decisions)} old decisions to {archive_file}")
    
    def add_decision(
        self,
        timestamp: datetime,
        action: str,
        reasoning: str,
        market_data: Dict,
        portfolio_state: Dict,
        confidence: float,
        news_context: Optional[List[str]] = None,
        result: Optional[Dict] = None
    ):
        """
        Record a trading decision
        
        Args:
            timestamp: Decision timestamp
            action: Action taken (e.g., "BUY 10 shares", "SELL 5 shares", "HOLD")
            reasoning: LLM's reasoning for the decision
            market_data: Market data at decision time
            portfolio_state: Portfolio state at decision time
            confidence: Confidence level (0-1)
            news_context: Relevant news articles considered
            result: Execution result (if action was taken)
        """
        decision = {
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "market_data": {
                "price": market_data.get("Close", market_data.get("price")),
                "volume": market_data.get("Volume", market_data.get("volume")),
                "high": market_data.get("High"),
                "low": market_data.get("Low")
            },
            "portfolio_state": {
                "cash": portfolio_state.get("current_cash"),
                "positions": portfolio_state.get("current_positions"),
                "total_value": portfolio_state.get("total_value")
            },
            "news_context": news_context or [],
            "result": result
        }
        
        self.memory["decisions"].append(decision)
        logger.info(f"Decision recorded: {action} (confidence: {confidence:.2f})")
    
    def add_learning(self, learning: str, context: Optional[Dict] = None):
        """
        Add a learning/insight
        
        Args:
            learning: Learning text
            context: Optional context dictionary
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "learning": learning,
            "context": context or {}
        }
        self.memory["learnings"].append(entry)
        logger.info(f"Learning added: {learning}")
    
    def add_pattern(self, pattern: str, description: str, effectiveness: float):
        """
        Record an identified pattern
        
        Args:
            pattern: Pattern name
            description: Pattern description
            effectiveness: Effectiveness score (0-1)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "description": description,
            "effectiveness": effectiveness
        }
        self.memory["patterns"].append(entry)
        logger.info(f"Pattern recorded: {pattern}")
    
    def add_mistake(self, mistake: str, lesson: str, context: Optional[Dict] = None):
        """
        Record a mistake and lesson learned
        
        Args:
            mistake: What went wrong
            lesson: What was learned
            context: Optional context
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mistake": mistake,
            "lesson": lesson,
            "context": context or {}
        }
        self.memory["mistakes"].append(entry)
        logger.info(f"Mistake recorded: {mistake}")
    
    def add_successful_strategy(self, strategy: str, description: str, profit: float):
        """
        Record a successful strategy
        
        Args:
            strategy: Strategy name
            description: Strategy description
            profit: Profit generated
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "description": description,
            "profit": profit
        }
        self.memory["successful_strategies"].append(entry)
        logger.info(f"Successful strategy recorded: {strategy}")
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent trading decisions"""
        return self.memory["decisions"][-limit:]
    
    def get_all_learnings(self) -> List[Dict]:
        """Get all learnings"""
        return self.memory["learnings"]
    
    def get_patterns(self) -> List[Dict]:
        """Get identified patterns"""
        return self.memory["patterns"]
    
    def get_mistakes(self) -> List[Dict]:
        """Get recorded mistakes"""
        return self.memory["mistakes"]
    
    def get_successful_strategies(self) -> List[Dict]:
        """Get successful strategies"""
        return self.memory["successful_strategies"]
    
    def get_context_summary(self, include_recent: int = 5) -> str:
        """
        Generate a context summary for LLM prompts
        
        Args:
            include_recent: Number of recent decisions to include
        
        Returns:
            Formatted context string
        """
        summary_parts = []
        
        # Recent decisions
        recent = self.get_recent_decisions(include_recent)
        if recent:
            summary_parts.append("=== Recent Decisions ===")
            for d in recent:
                action = d.get("action", "UNKNOWN")
                reasoning = d.get("reasoning", "")
                confidence = d.get("confidence", 0)
                summary_parts.append(f"- {action} (confidence: {confidence:.2f})")
                summary_parts.append(f"  Reasoning: {reasoning[:100]}...")
        
        # Key learnings
        learnings = self.get_all_learnings()
        if learnings:
            summary_parts.append("\n=== Key Learnings ===")
            for l in learnings[-5:]:  # Last 5 learnings
                summary_parts.append(f"- {l.get('learning', '')}")
        
        # Patterns
        patterns = self.get_patterns()
        if patterns:
            summary_parts.append("\n=== Identified Patterns ===")
            for p in patterns[-3:]:  # Last 3 patterns
                summary_parts.append(f"- {p.get('pattern', '')}: {p.get('description', '')}")
        
        # Mistakes to avoid
        mistakes = self.get_mistakes()
        if mistakes:
            summary_parts.append("\n=== Mistakes to Avoid ===")
            for m in mistakes[-3:]:  # Last 3 mistakes
                summary_parts.append(f"- {m.get('mistake', '')}")
                summary_parts.append(f"  Lesson: {m.get('lesson', '')}")
        
        return "\n".join(summary_parts) if summary_parts else "No memory context available yet."
    
    def get_statistics(self) -> Dict:
        """Get memory bank statistics"""
        return {
            "total_decisions": len(self.memory["decisions"]),
            "total_learnings": len(self.memory["learnings"]),
            "total_patterns": len(self.memory["patterns"]),
            "total_mistakes": len(self.memory["mistakes"]),
            "total_successful_strategies": len(self.memory["successful_strategies"]),
            "created_at": self.memory["metadata"]["created_at"],
            "last_updated": self.memory["metadata"]["last_updated"]
        }
