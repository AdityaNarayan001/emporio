"""
Trading Prompts - System prompts and templates for LLM trading agent
"""


SYSTEM_PROMPT = """You are an expert stock trading AI agent specializing in Indian stock market trading, specifically IRCTC stock.

Your goal is to maximize profit through strategic trading decisions. You have access to:
- Real-time market data (price, volume, historical trends)
- Your portfolio status (cash, positions, total value)
- News search capability (when enabled)
- Your own memory of past decisions and learnings

TRADING RULES:
1. You can only trade WHOLE SHARES (no fractional shares)
2. You start with ₹10,000 (configurable)
3. Trading commission may apply (check your portfolio info)
4. You can BUY, SELL, or HOLD based on your analysis
5. You decide WHEN to trade based on your confidence and market conditions
6. You cannot see future data - only current and historical data

DECISION MAKING:
- Analyze market trends, patterns, and indicators
- Consider news and sentiment (if news search is enabled)
- Use technical analysis: support/resistance, moving averages, volume analysis
- Learn from your past decisions and outcomes
- Only trade when you have HIGH CONFIDENCE (>60%)
- Consider risk management - don't put all capital in one trade

OUTPUT FORMAT:
When you decide to make a trade, respond with a structured decision:

```json
{
  "action": "BUY" | "SELL" | "HOLD",
  "quantity": <number of shares (integer only)>,
  "reasoning": "<your detailed reasoning>",
  "confidence": <0.0 to 1.0>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "expected_outcome": "<what you expect to happen>",
  "stop_loss": <price level to exit if wrong>,
  "take_profit": <target price level>
}
```

REMEMBER:
- Think like a professional trader
- Base decisions on data and analysis, not emotions
- Learn from mistakes and successes
- Be patient - not every moment requires action
- Preserve capital - avoid unnecessary risks
"""


DECISION_PROMPT_TEMPLATE = """
Current Time: {current_time}

=== MARKET DATA ===
Symbol: {symbol}
Current Price: ₹{current_price}
Change: {price_change}% 
Volume: {volume}
High: ₹{high}
Low: ₹{low}

Recent Price Trend (last {trend_periods} periods):
{price_trend}

=== PORTFOLIO STATUS ===
Cash Available: ₹{cash:.2f}
Current Positions: {positions}
Portfolio Value: ₹{portfolio_value:.2f}
Total Return: {total_return:.2f}% 
Max Affordable Shares: {max_affordable}

=== YOUR MEMORY ===
{memory_context}

=== NEWS CONTEXT ===
{news_context}

=== ANALYSIS REQUEST ===
Based on the above data:
1. Analyze the current market situation
2. Review your past decisions and learnings
3. Consider the news context (if available)
4. Decide whether to BUY, SELL, or HOLD
5. If trading, specify the exact quantity (whole shares only)
6. Provide your confidence level (0.0 to 1.0)

Remember: Only trade if you have high confidence (>0.6) and solid reasoning.
Make your decision now.
"""


REFLECTION_PROMPT_TEMPLATE = """
You just executed a trade. Let's reflect on it:

=== TRADE DETAILS ===
Action: {action}
Quantity: {quantity} shares
Price: ₹{price}
Total Cost/Revenue: ₹{total}
Reasoning: {reasoning}

=== OUTCOME ===
Success: {success}
{outcome_details}

=== PORTFOLIO AFTER ===
Cash: ₹{cash_after}
Positions: {positions_after}
Portfolio Value: ₹{portfolio_value_after}

Please analyze:
1. Was this decision correct based on what happened?
2. What patterns or insights can you extract?
3. What would you do differently next time?
4. Any learnings to remember?

Provide your reflection in a structured format:
- Key Learning: <one sentence>
- Pattern Identified (if any): <pattern description>
- Mistake (if any): <what went wrong and why>
- Strategy Note: <strategy insight>
"""


NEWS_SEARCH_PROMPT = """
Based on the current market situation for {symbol}, what news topics should I search for?
Provide 2-3 specific search queries that would help inform your trading decision.
Consider: company news, sector trends, regulatory changes, economic indicators.

Format: Return a simple list of search queries, one per line.
"""


def format_decision_prompt(
    current_time,
    symbol: str,
    current_price: float,
    price_change: float,
    volume: int,
    high: float,
    low: float,
    price_trend: str,
    trend_periods: int,
    cash: float,
    positions: dict,
    portfolio_value: float,
    total_return: float,
    max_affordable: int,
    memory_context: str,
    news_context: str = "News search not enabled"
) -> str:
    """Format the decision prompt with current data"""
    positions_str = ", ".join([f"{sym}: {qty} shares" for sym, qty in positions.items()]) or "None"
    
    return DECISION_PROMPT_TEMPLATE.format(
        current_time=current_time,
        symbol=symbol,
        current_price=current_price,
        price_change=price_change,
        volume=volume,
        high=high,
        low=low,
        price_trend=price_trend,
        trend_periods=trend_periods,
        cash=cash,
        positions=positions_str,
        portfolio_value=portfolio_value,
        total_return=total_return,
        max_affordable=max_affordable,
        memory_context=memory_context,
        news_context=news_context
    )


def format_reflection_prompt(
    action: str,
    quantity: int,
    price: float,
    total: float,
    reasoning: str,
    success: bool,
    outcome_details: str,
    cash_after: float,
    positions_after: dict,
    portfolio_value_after: float
) -> str:
    """Format the reflection prompt after a trade"""
    positions_str = ", ".join([f"{sym}: {qty} shares" for sym, qty in positions_after.items()]) or "None"
    
    return REFLECTION_PROMPT_TEMPLATE.format(
        action=action,
        quantity=quantity,
        price=price,
        total=total,
        reasoning=reasoning,
        success=success,
        outcome_details=outcome_details,
        cash_after=cash_after,
        positions_after=positions_str,
        portfolio_value_after=portfolio_value_after
    )
