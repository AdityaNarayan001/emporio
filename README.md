# 🤖 Emporio - LLM Stock Trading Simulator

> An autonomous AI trading system where a Large Language Model (Gemini 2.0) learns to trade IRCTC stock in **async real-time simulation**.

## ⚡ Quick Start

```bash
# 1. Install dependencies (if not done)
pip install -r requirements.txt

# 2. Add your Gemini API key to config/config.yaml
#    Find: llm.api_key and replace with your key
#    Get key: https://makersuite.google.com/app/apikey

# 3. Run the simulator
python main.py
```

That's it! The UI will open in your browser. Click "🚀 Initialize System" → "▶️ Start Simulation" to begin.

---

## 🎯 Features

- **🧠 Autonomous LLM Trading**: Gemini 2.0 Flash makes trading decisions independently
- **⚡ Async Real-Time Mode**: Stock prices flow continuously while LLM analyzes in background
- **📊 Professional Charts**: Candlestick, volume, portfolio tracking with buy/sell markers
- **🎯 Whole Shares Only**: Realistic trading with integer quantities (no fractional shares)
- **💾 Evolving Memory**: JSON-based memory bank storing decisions, learnings, and patterns
- **📰 News Integration**: Time-aware news search to inform trading decisions (toggle-able)
- **📈 Interactive UI**: Beautiful Streamlit dashboard with live charts and metrics
- **🎮 Full Control**: Pause, resume, speed control (0.5x to 20x)
- **⚙️ Highly Configurable**: YAML configuration for all parameters
- **🔍 Full Transparency**: See LLM's reasoning for every decision

## 🏗️ Architecture

```
emporio/
├── config/
│   ├── config.yaml          # Main configuration
│   └── .env.template        # API keys template
├── src/
│   ├── data/
│   │   ├── data_loader.py   # Historical data fetching
│   │   └── data_simulator.py # Real-time simulation
│   ├── llm/
│   │   ├── llm_agent.py     # Gemini trading agent
│   │   ├── memory_bank.py   # JSON memory storage
│   │   └── prompts.py       # Trading prompts
│   ├── tools/
│   │   └── news_search.py   # NewsAPI integration
│   ├── trading/
│   │   ├── portfolio.py     # Portfolio management
│   │   ├── order_executor.py # Trade execution
│   │   └── performance.py   # Performance tracking
│   └── ui/
│       └── app.py           # Streamlit UI
├── data/                    # Data storage
├── logs/                    # Trading logs
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- NewsAPI key (optional, [Get one here](https://newsapi.org/register))

### 2. Installation

```bash
# Navigate to the project directory
cd emporio

# Create and activate virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

**⚠️ Important: Your `config/config.yaml` file with API keys is gitignored and won't be committed to GitHub.**

If you're setting up for the first time:
```bash
# Copy the template (if config.yaml doesn't exist)
cp config/config.yaml.template config/config.yaml
```

Edit `config/config.yaml` and add your API keys:

```yaml
# API Keys
api_keys:
  gemini_api_key: "your_gemini_api_key_here"  # Required - Get from https://makersuite.google.com/app/apikey
  news_api_key: "your_newsapi_key_here"       # Optional - Get from https://newsapi.org/register
```

**Get API Keys:**

- **Gemini API Key (Required)**: Visit https://makersuite.google.com/app/apikey
- **NewsAPI Key (Optional)**: Visit https://newsapi.org/register

You can also customize other settings in `config/config.yaml`:
- Initial capital (default: ₹10,000)
- Stock symbol (default: IRCTC.NS)
- Simulation speed (default: 1.0x)
- LLM parameters (temperature, confidence threshold)
- Trading rules (commission percentage)
- News search settings

### 4. Run the Simulator

```bash
# Run the main application
python main.py
```

The Streamlit UI will open in your browser automatically.

## 🎮 How to Use

1. **Initialize System**: Click "🚀 Initialize System" in the sidebar
   - Loads 1 year of IRCTC historical data
   - Initializes all components (portfolio, LLM, memory bank)

2. **Start Simulation**: Click "▶️ Start Simulation"
   - Historical data begins replaying as real-time feed
   - LLM autonomously analyzes market and makes trading decisions

3. **Control Simulation**:
   - **Pause/Resume**: Control simulation flow
   - **Speed Control**: Adjust from 0.5x to 20x speed
   - **News Toggle**: Enable/disable news search

4. **Monitor Performance**:
   - Live portfolio value and returns
   - Price charts with trade markers
   - LLM reasoning and confidence
   - Trade history
   - Memory bank insights

## 📋 Key Concepts

### Time-Travel Simulation
- Historical data is replayed chronologically
- LLM can only see past data up to current simulation time
- No lookahead bias - ensures realistic testing

### Autonomous Trading
- LLM decides **WHEN** to trade (not forced on a schedule)
- Trades only when confidence exceeds threshold (default: 60%)
- Whole shares only (no fractional trading)

### Memory Bank
- Stores all decisions with reasoning
- Learns from successes and mistakes
- Identifies patterns over time
- Memory informs future decisions

### News Integration
- Time-aware search (only past news, not future)
- Searches relevant keywords (IRCTC, Indian Railways, etc.)
- Can be toggled on/off during simulation

## ⚙️ Configuration Reference

### Key Configuration Parameters

```yaml
# Trading Configuration
trading:
  initial_capital: 10000              # Starting capital (₹)
  commission_percentage: 0.0          # Trading fees (0 = no fees)
  
# Simulation Configuration  
simulation:
  historical_period_days: 365         # Data period (1 year)
  data_interval: "5m"                 # 1m, 5m, 15m, 30m, 1h, 1d
  simulation_speed: 1.0               # Real-time (1.0x)
  max_lookback_period: "5d"           # Max history LLM can see
  
# LLM Configuration
llm:
  model_name: "gemini-2.0-flash-exp"  # Gemini model
  temperature: 0.7                    # Creativity (0.0-1.0)
  min_confidence_threshold: 0.6       # Min confidence to trade
  
# Tools Configuration
tools:
  news_search:
    enabled: true                     # Enable news search
    max_articles: 5                   # Articles per query
```

## 📊 Understanding the UI

### Main Dashboard

1. **Metrics Bar**: Portfolio value, cash, positions, price, progress
2. **Price Chart**: Candlestick chart with trade markers
3. **LLM Activity**: Current decision, reasoning, confidence
4. **Trades Table**: Recent executed trades
5. **Memory Bank**: Decision count, learnings, patterns

### Trade Markers
- 🟢 Green Triangle Up: BUY trade
- 🔴 Red Triangle Down: SELL trade

## 🧪 Example Trading Scenario

```
Time: 10:30 AM
Price: ₹850
LLM Analysis: 
  - Price broke resistance at ₹845
  - Volume increasing (bullish signal)
  - News: Railway budget approval positive
  - Memory: Similar pattern led to 3% gain previously
  
Decision: BUY 10 shares
Confidence: 0.78 (High)
Reasoning: "Strong bullish momentum with volume confirmation..."

Result: ✅ Executed - Total cost: ₹8,500
```

## 🔧 Troubleshooting

### Issue: "GEMINI_API_KEY not configured"
- Open `config/config.yaml`
- Find the `api_keys` section
- Replace `your_gemini_api_key_here` with your actual Gemini API key
- Get key from: https://makersuite.google.com/app/apikey

### Issue: "No data found for IRCTC.NS"
- Check internet connection
- Verify stock symbol is correct in config
- Try adjusting date range in config (yfinance has limitations on intraday data)

### Issue: "News search disabled"
- News is optional - simulation works without it
- To enable: Add your NewsAPI key to `config/config.yaml` under `api_keys.news_api_key`
- Or toggle off in UI sidebar

### Issue: Simulation runs too fast/slow
- Adjust speed slider in sidebar (0.5x to 20x)
- Lower speed for detailed observation
- Higher speed for quick backtesting

## 📈 Performance Metrics

The simulator tracks:
- **Total Return**: Absolute and percentage gains/losses
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable periods
- **Trade Count**: Number of executed trades

## 🎓 Learning Features

### Memory Bank Categories

1. **Decisions**: All trading actions with reasoning
2. **Learnings**: Key insights extracted from experiences
3. **Patterns**: Identified market patterns and their effectiveness
4. **Mistakes**: Recorded errors with lessons learned
5. **Strategies**: Successful approaches and their profits

## 🔬 Advanced Usage

### Custom Lookback Periods

```yaml
simulation:
  max_lookback_period: "5d"  # 5 days
  # Options: "30m", "1h", "4h", "1d", "5d", etc.
```

### Adjusting LLM Behavior

```yaml
llm:
  temperature: 0.3        # Conservative (0.0-0.5)
  temperature: 0.7        # Balanced (default)
  temperature: 1.0        # Aggressive (0.8-1.0)
  
  min_confidence_threshold: 0.8  # Strict (fewer trades)
  min_confidence_threshold: 0.5  # Relaxed (more trades)
```

### Commission Testing

```yaml
trading:
  commission_percentage: 0.0   # No fees
  commission_percentage: 0.1   # 0.1% per trade
  commission_percentage: 0.5   # 0.5% per trade
```

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Experiment with different configurations
- Add new indicators or analysis tools
- Improve LLM prompts
- Add more data sources

## ⚠️ Disclaimer

**This is a SIMULATION for educational and research purposes only.**

- Not financial advice
- Not for live trading
- Past performance ≠ future results
- Use at your own risk

## 📝 License

MIT License - Free to use and modify

## 🆘 Support

For issues or questions:
1. Check this README
2. Review `config/config.yaml` comments
3. Check logs in `logs/trading.log`
4. Examine memory bank in `data/memory_bank.json`

## 🎉 Acknowledgments

- **yfinance**: Historical stock data
- **Google Gemini**: LLM capabilities
- **Streamlit**: Beautiful UI framework
- **NewsAPI**: News data provider

---

**Happy Trading! 🚀📈**

Remember: The goal is to learn how an LLM can adapt and improve its trading strategies over time through experience and reflection.
