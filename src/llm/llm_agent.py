"""
LLM Trading Agent - Gemini-powered autonomous trading agent
"""
import google.generativeai as genai
import os
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import re

from .prompts import SYSTEM_PROMPT, format_decision_prompt, format_reflection_prompt
from .memory_bank import MemoryBank
from ..tools.news_search import NewsSearchTool

logger = logging.getLogger(__name__)


class LLMTradingAgent:
    """
    AI Trading Agent powered by Google Gemini.
    Makes autonomous trading decisions based on market data, memory, and news.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        memory_bank: Optional[MemoryBank] = None,
        news_tool: Optional[NewsSearchTool] = None,
        min_confidence: float = 0.6,
        debug: bool = False
    ):
        """
        Initialize LLM Trading Agent
        
        Args:
            api_key: Gemini API key (required, from config.yaml)
            model_name: Gemini model name
            temperature: Model temperature (0.0 = deterministic, 1.0 = creative)
            memory_bank: MemoryBank instance for storing decisions
            news_tool: NewsSearchTool instance for news search
            min_confidence: Minimum confidence threshold for trades
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Gemini API key is required. Please set it in config.yaml under llm.api_key")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        self.temperature = temperature
        self.min_confidence = min_confidence
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Initialize chat session with system prompt
        self.chat = self.model.start_chat(history=[])
        
        # Components
        self.memory_bank = memory_bank
        self.news_tool = news_tool
        
        # Decision tracking
        self.last_decision_time = None
        self.decision_count = 0
        
        # Debug / visibility state (correctly scoped inside __init__)
        self.debug = debug
        self.last_prompt = None
        self.last_raw_response = None
        self.last_decision = None
        self.last_error = None

        logger.info(f"LLM Trading Agent initialized with {self.model_name} (debug={self.debug})")
    
    def make_decision(
        self,
        market_data: Dict,
        portfolio_summary: Dict,
        current_time: datetime,
        max_affordable: int,
        lookback_data: Optional[Dict] = None
    ) -> Dict:
        """
        Make a trading decision based on current market state
        
        Args:
            market_data: Current market data (price, volume, etc.)
            portfolio_summary: Current portfolio state
            current_time: Current simulation time
            max_affordable: Maximum shares affordable
            lookback_data: Historical market data for analysis
        
        Returns:
            Decision dictionary with action, quantity, reasoning, confidence
        """
        try:
            # Get memory context
            logger.debug("LLM TOOL: Retrieving memory context...")
            memory_context = self.memory_bank.get_context_summary() if self.memory_bank else "No memory available"
            logger.debug(f"LLM TOOL: Memory context retrieved ({len(memory_context)} characters)")
            
            # Get news context (if enabled)
            news_context = self._get_news_context(current_time)
            
                        # Calculate price trend
            price_trend = self._format_price_trend(lookback_data) if lookback_data is not None and not lookback_data.empty else "No historical data"
            
            # Calculate price change
            price_change = 0.0
            if lookback_data is not None and not lookback_data.empty and len(lookback_data) > 1:
                prev_close = lookback_data.iloc[-2]['Close']
                current_price = market_data.get('Close', market_data.get('price', 0))
                price_change = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            # Format the decision prompt
            prompt = format_decision_prompt(
                current_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                symbol=market_data.get('symbol', 'IRCTC'),
                current_price=market_data.get('Close', market_data.get('price', 0)),
                price_change=price_change,
                volume=market_data.get('Volume', 0),
                high=market_data.get('High', 0),
                low=market_data.get('Low', 0),
                price_trend=price_trend,
                trend_periods=len(lookback_data) if lookback_data is not None else 0,
                cash=portfolio_summary.get('current_cash', 0),
                positions=portfolio_summary.get('current_positions', {}),
                portfolio_value=portfolio_summary.get('total_value', 0),
                total_return=portfolio_summary.get('return_percentage', 0),
                max_affordable=max_affordable,
                memory_context=memory_context,
                news_context=news_context
            )
            
            # Get LLM decision
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            logger.info("LLM TOOL: Sending prompt to Gemini API...")
            # Save prompt for visibility
            self.last_prompt = full_prompt
            response = self.model.generate_content(full_prompt)
            # Safely extract text from multi-part responses
            extracted_text = self._extract_response_text(response)
            logger.info(f"LLM TOOL: Received response ({len(extracted_text)} characters)")
            self.last_raw_response = extracted_text
            
            # Parse response
            decision = self._parse_decision(extracted_text)
            logger.debug(f"LLM TOOL: Parsed decision - {decision['action']} {decision.get('quantity', 0)} shares")
            
            # Validate decision
            decision = self._validate_decision(decision, max_affordable, portfolio_summary)
            
            # Store in memory
            if self.memory_bank and decision['action'] != 'HOLD':
                self.memory_bank.add_decision(
                    timestamp=current_time,
                    action=f"{decision['action']} {decision.get('quantity', 0)} shares",
                    reasoning=decision['reasoning'],
                    market_data=market_data,
                    portfolio_state=portfolio_summary,
                    confidence=decision['confidence'],
                    news_context=[news_context] if news_context else None
                )
                self.memory_bank.save()
            
            self.last_decision_time = current_time
            self.decision_count += 1
            
            logger.info(
                f"Decision #{self.decision_count}: {decision['action']} "
                f"(confidence: {decision['confidence']:.2f})"
            )
            
            self.last_decision = decision
            return decision
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error making decision: {str(e)}")
            return {
                "action": "HOLD",
                "quantity": 0,
                "reasoning": f"Error occurred: {str(e)}",
                "confidence": 0.0,
                "risk_level": "HIGH"
            }
    
    def _get_news_context(self, current_time: datetime) -> str:
        """Get news context for decision making"""
        if not self.news_tool or not self.news_tool.is_enabled():
            logger.debug("LLM TOOL: News search disabled")
            return "News search not enabled"
        
        logger.info("LLM TOOL: Searching for news...")
        # Search for relevant news
        keywords = ["IRCTC", "Indian Railways", "railway stocks"]
        news_context = self.news_tool.search_for_trading(
            keywords=keywords,
            current_date=current_time,
            lookback_days=7,
            max_results_per_keyword=2
        )
        
        logger.info(f"LLM TOOL: News search completed, {len(news_context)} characters returned")
        return news_context
    
    def _format_price_trend(self, lookback_data) -> str:
        """Format recent price trend for prompt"""
        if lookback_data is None or len(lookback_data) == 0:
            return "No data"
        
        recent = lookback_data.tail(20)  # Last 20 data points
        trend_lines = []
        
        for _, row in recent.iterrows():
            time_str = row['Datetime'].strftime("%m/%d %H:%M")
            trend_lines.append(f"{time_str}: ₹{row['Close']:.2f} (Vol: {row['Volume']:,})")
        
        return "\n".join(trend_lines)
    
    def _parse_decision(self, response_text: str) -> Dict:
        """Parse LLM response into decision dictionary"""
        if self.debug:
            logger.debug("DEBUG: Raw LLM response start >>>")
            logger.debug(response_text)  # full, UI handles scroll
            logger.debug("DEBUG: Raw LLM response end <<<")
        # Try to extract JSON from response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        
        if json_match:
            try:
                decision = json.loads(json_match.group(1))
                return self._normalize_decision(decision)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response")
        
        # Fallback: parse from text
        decision = {
            "action": "HOLD",
            "quantity": 0,
            # Keep full cleaned reasoning (no truncation; UI scroll handles length)
            "reasoning": self._clean_reasoning_text(response_text),
            "confidence": 0.5,
            "risk_level": "MEDIUM"
        }
        
        # Extract action
        if "BUY" in response_text.upper():
            decision["action"] = "BUY"
        elif "SELL" in response_text.upper():
            decision["action"] = "SELL"
        
        # Extract quantity (look for numbers)
        qty_match = re.search(r'quantity["\s:]+(\d+)', response_text, re.IGNORECASE)
        if qty_match:
            decision["quantity"] = int(qty_match.group(1))
        
        # Extract confidence
        conf_match = re.search(r'confidence["\s:]+([0-9.]+)', response_text, re.IGNORECASE)
        if conf_match:
            decision["confidence"] = float(conf_match.group(1))
        
        return decision

    def _clean_reasoning_text(self, raw: str) -> str:
        """Remove duplicated metric tokens from raw LLM response before storing as reasoning."""
        if not raw:
            return ""
        patterns = [r'^Confidence:?\s*[0-9.]+%?$', r'^Risk Level:?\s*\w+$', r'^Quantity:?\s*\d+$']
        lines = raw.splitlines()
        import re
        kept = []
        for ln in lines:
            if any(re.match(p, ln.strip(), re.IGNORECASE) for p in patterns):
                continue
            kept.append(ln)
        cleaned = '\n'.join(kept)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned

    def _extract_response_text(self, response) -> str:
        """Handle Gemini multi-part responses robustly returning concatenated text."""
        try:
            if hasattr(response, 'text') and isinstance(response.text, str):
                # Some simple responses still work
                return response.text
        except Exception:
            pass
        parts_text = []
        try:
            # Preferred: response.parts
            if hasattr(response, 'parts') and response.parts:
                for p in response.parts:
                    if hasattr(p, 'text') and p.text:
                        parts_text.append(p.text)
                    elif isinstance(getattr(p, 'content', None), str):
                        parts_text.append(p.content)
            # Fallback: response.candidates
            elif hasattr(response, 'candidates') and response.candidates:
                for cand in response.candidates:
                    content = getattr(cand, 'content', None)
                    if content and hasattr(content, 'parts'):
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
            combined = "\n".join(parts_text).strip()
            return combined or "(Empty response)"
        except Exception as e:
            logger.error(f"Failed to extract multi-part response text: {e}")
            return f"(Extraction error: {e})"

    def get_visibility(self) -> Dict:
        """Return internal visibility state for debugging/UI"""
        return {
            "last_prompt": self.last_prompt,
            "last_raw_response": (self.last_raw_response[:2000] + '…') if self.last_raw_response and len(self.last_raw_response) > 2000 else self.last_raw_response,
            "last_decision": self.last_decision,
            "last_error": self.last_error,
            "decision_count": self.decision_count,
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None,
            "min_confidence": self.min_confidence,
            "model_name": self.model_name
        }
    
    def _normalize_decision(self, decision: Dict) -> Dict:
        """Normalize and validate decision format"""
        normalized = {
            "action": decision.get("action", "HOLD").upper(),
            "quantity": int(decision.get("quantity", 0)),
            "reasoning": decision.get("reasoning", "No reasoning provided"),
            "confidence": float(decision.get("confidence", 0.5)),
            "risk_level": decision.get("risk_level", "MEDIUM"),
            "expected_outcome": decision.get("expected_outcome", ""),
            "stop_loss": decision.get("stop_loss"),
            "take_profit": decision.get("take_profit")
        }
        
        # Ensure action is valid
        if normalized["action"] not in ["BUY", "SELL", "HOLD"]:
            normalized["action"] = "HOLD"
        
        # Ensure confidence is in range
        normalized["confidence"] = max(0.0, min(1.0, normalized["confidence"]))
        
        return normalized
    
    def _validate_decision(self, decision: Dict, max_affordable: int, portfolio_summary: Dict) -> Dict:
        """Validate and adjust decision if needed"""
        # Check confidence threshold
        if decision["confidence"] < self.min_confidence and decision["action"] != "HOLD":
            logger.info(
                f"Confidence {decision['confidence']:.2f} below threshold {self.min_confidence}, "
                "changing to HOLD"
            )
            decision["action"] = "HOLD"
            decision["quantity"] = 0
            decision["reasoning"] += " (Confidence below threshold)"
        
        # Validate quantity
        if decision["action"] == "BUY":
            if decision["quantity"] > max_affordable:
                logger.warning(
                    f"Requested quantity {decision['quantity']} exceeds affordable {max_affordable}, "
                    "adjusting..."
                )
                decision["quantity"] = max_affordable
                decision["reasoning"] += f" (Adjusted to max affordable: {max_affordable})"
        
        elif decision["action"] == "SELL":
            current_position = portfolio_summary.get('current_positions', {}).get('IRCTC', 0)
            if decision["quantity"] > current_position:
                logger.warning(
                    f"Requested sell quantity {decision['quantity']} exceeds position {current_position}, "
                    "adjusting..."
                )
                decision["quantity"] = current_position
                decision["reasoning"] += f" (Adjusted to current position: {current_position})"
        
        # Ensure quantity is integer
        decision["quantity"] = int(decision["quantity"])
        
        return decision
    
    def reflect_on_trade(
        self,
        trade_result: Dict,
        portfolio_after: Dict
    ):
        """
        Reflect on a completed trade and extract learnings
        
        Args:
            trade_result: Result of the executed trade
            portfolio_after: Portfolio state after trade
        """
        if not self.memory_bank:
            return
        
        try:
            # Format reflection prompt
            action = trade_result.get("action", "UNKNOWN")
            transaction = trade_result.get("transaction", {})
            
            prompt = format_reflection_prompt(
                action=action,
                quantity=transaction.get("quantity", 0),
                price=transaction.get("price", 0),
                total=transaction.get("total_cost", transaction.get("net_revenue", 0)),
                reasoning=trade_result.get("reasoning", ""),
                success=trade_result.get("success", False),
                outcome_details=trade_result.get("reason", "Executed successfully"),
                cash_after=portfolio_after.get("current_cash", 0),
                positions_after=portfolio_after.get("current_positions", {}),
                portfolio_value_after=portfolio_after.get("total_value", 0)
            )
            
            # Get reflection from LLM
            response = self.model.generate_content(prompt)
            reflection = response.text
            
            # Extract and store learnings
            self._extract_learnings(reflection, trade_result)
            
            logger.info("Trade reflection completed")
            
        except Exception as e:
            logger.error(f"Error in trade reflection: {str(e)}")
    
    def _extract_learnings(self, reflection: str, trade_result: Dict):
        """Extract structured learnings from reflection"""
        # Extract key learning
        learning_match = re.search(r'Key Learning:\s*(.+?)(?:\n|$)', reflection, re.IGNORECASE)
        if learning_match:
            self.memory_bank.add_learning(learning_match.group(1).strip())
        
        # Extract pattern
        pattern_match = re.search(r'Pattern Identified.*?:\s*(.+?)(?:\n|$)', reflection, re.IGNORECASE)
        if pattern_match:
            self.memory_bank.add_pattern(
                pattern="Trading Pattern",
                description=pattern_match.group(1).strip(),
                effectiveness=0.7  # Default effectiveness
            )
        
        # Extract mistake
        mistake_match = re.search(r'Mistake.*?:\s*(.+?)(?:\n|$)', reflection, re.IGNORECASE)
        if mistake_match and not trade_result.get("success", True):
            lesson_match = re.search(r'Lesson.*?:\s*(.+?)(?:\n|$)', reflection, re.IGNORECASE)
            lesson = lesson_match.group(1).strip() if lesson_match else "Learn from this mistake"
            self.memory_bank.add_mistake(
                mistake=mistake_match.group(1).strip(),
                lesson=lesson
            )
        
        self.memory_bank.save()
