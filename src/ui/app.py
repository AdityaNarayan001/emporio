"""
Streamlit UI - Main application interface for LLM Stock Trading Simulator
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.portfolio import Portfolio
from src.trading.order_executor import OrderExecutor
from src.trading.performance import PerformanceTracker
from src.data.data_loader import DataLoader
from src.data.data_simulator import DataSimulator
from src.llm.llm_agent import LLMTradingAgent
from src.llm.memory_bank import MemoryBank
from src.tools.news_search import NewsSearchTool

import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file"""
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def initialize_session_state(config):
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.simulation_started = False
        st.session_state.config = config
        
        # Components (will be initialized when simulation starts)
        st.session_state.data_loader = None
        st.session_state.simulator = None
        st.session_state.portfolio = None
        st.session_state.executor = None
        st.session_state.performance = None
        st.session_state.memory_bank = None
        st.session_state.news_tool = None
        st.session_state.llm_agent = None
        
        # State
        st.session_state.last_decision = None
        st.session_state.decisions_log = []
        st.session_state.trades_log = []


def initialize_components(config):
    """Initialize all trading components"""
    try:
        with st.spinner("Initializing trading system..."):
            # Load historical data
            st.session_state.data_loader = DataLoader(
                symbol=config['trading']['stock_symbol'],
                cache_dir="data"
            )
            
            # Fetch data
            historical_data = st.session_state.data_loader.fetch_historical_data(
                start_date=config['simulation']['start_date'],
                end_date=config['simulation']['end_date'],
                period_days=config['simulation']['historical_period_days'],
                interval=config['simulation']['data_interval']
            )
            
            # Initialize simulator
            st.session_state.simulator = DataSimulator(
                historical_data=historical_data,
                speed_multiplier=config['simulation']['simulation_speed'],
                max_lookback=config['simulation']['max_lookback_period']
            )
            
            # Initialize portfolio
            st.session_state.portfolio = Portfolio(
                initial_capital=config['trading']['initial_capital'],
                commission_rate=config['trading']['commission_percentage']
            )
            
            # Initialize executor
            st.session_state.executor = OrderExecutor(st.session_state.portfolio)
            
            # Initialize performance tracker
            st.session_state.performance = PerformanceTracker()
            
            # Initialize memory bank
            st.session_state.memory_bank = MemoryBank(
                file_path=config['memory']['file_path'],
                max_entries=config['memory']['max_entries']
            )
            
            # Initialize news tool
            st.session_state.news_tool = NewsSearchTool(
                api_key=config['tools']['news_search'].get('api_key'),
                enabled=config['tools']['news_search']['enabled']
            )
            
            # Initialize LLM agent
            st.session_state.llm_agent = LLMTradingAgent(
                api_key=config['llm']['api_key'],
                model_name=config['llm']['model_name'],
                temperature=config['llm']['temperature'],
                memory_bank=st.session_state.memory_bank,
                news_tool=st.session_state.news_tool,
                min_confidence=config['llm']['min_confidence_threshold']
            )
            
            st.session_state.initialized = True
            st.success("✅ All components initialized successfully!")
            
    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        logger.error(f"Initialization error: {str(e)}", exc_info=True)


def render_header():
    """Render application header"""
    st.title("🤖 Emporio - LLM Stock Trading Simulator")
    st.markdown("---")


def render_control_panel():
    """Render simulation control panel"""
    st.sidebar.header("🎮 Control Panel")
    
    config = st.session_state.config
    
    # Initialization
    if not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize System", use_container_width=True):
            initialize_components(config)
        return
    
    # Start/Stop simulation
    if not st.session_state.simulation_started:
        if st.sidebar.button("▶️ Start Simulation", use_container_width=True):
            st.session_state.simulator.start()
            st.session_state.simulation_started = True
            st.rerun()
    else:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.session_state.simulator.is_paused:
                if st.button("▶️ Resume", use_container_width=True):
                    st.session_state.simulator.resume()
                    st.rerun()
            else:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.simulator.pause()
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.simulator.stop()
                st.session_state.simulation_started = False
                st.rerun()
    
    # Speed control
    if st.session_state.initialized:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Speed Control")
        
        speed = st.sidebar.select_slider(
            "Simulation Speed",
            options=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            value=config['simulation']['simulation_speed'],
            format_func=lambda x: f"{x}x"
        )
        
        if speed != st.session_state.simulator.speed_multiplier:
            st.session_state.simulator.set_speed(speed)
    
    # News toggle
    if st.session_state.initialized and st.session_state.news_tool:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📰 News Search")
        news_enabled = st.sidebar.checkbox(
            "Enable News Search",
            value=st.session_state.news_tool.is_enabled()
        )
        st.session_state.news_tool.toggle(news_enabled)


def render_metrics():
    """Render key metrics"""
    if not st.session_state.initialized or not st.session_state.simulator:
        return
    
    progress = st.session_state.simulator.get_progress()
    current_data = st.session_state.simulator.get_current_data()
    
    if current_data is None:
        st.info("⏳ Waiting for simulation to start...")
        return
    
    # Get portfolio summary
    current_prices = {config['trading']['stock_symbol']: current_data['Close']}
    portfolio_summary = st.session_state.portfolio.get_summary(current_prices)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"₹{portfolio_summary['total_value']:.2f}",
            f"{portfolio_summary['return_percentage']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"₹{portfolio_summary['current_cash']:.2f}"
        )
    
    with col3:
        positions = portfolio_summary['current_positions']
        pos_str = f"{positions.get(config['trading']['stock_symbol'], 0)} shares"
        st.metric("Position", pos_str)
    
    with col4:
        st.metric(
            "Current Price",
            f"₹{current_data['Close']:.2f}"
        )
    
    with col5:
        st.metric(
            "Progress",
            f"{progress['progress_percentage']:.1f}%"
        )


def render_price_chart():
    """Render price and portfolio value chart"""
    if not st.session_state.initialized or not st.session_state.simulator:
        return
    
    available_data = st.session_state.simulator.get_available_data()
    
    if len(available_data) == 0:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Stock Price', 'Volume'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=available_data['Datetime'],
            open=available_data['Open'],
            high=available_data['High'],
            low=available_data['Low'],
            close=available_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=available_data['Datetime'],
            y=available_data['Volume'],
            name='Volume',
            marker_color='rgba(0,100,250,0.3)'
        ),
        row=2, col=1
    )
    
    # Add trade markers
    for trade in st.session_state.trades_log:
        trade_time = pd.to_datetime(trade['timestamp'])
        if trade_time in available_data['Datetime'].values:
            color = 'green' if trade['type'] == 'BUY' else 'red'
            symbol_marker = 'triangle-up' if trade['type'] == 'BUY' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[trade_time],
                    y=[trade['price']],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol=symbol_marker),
                    name=f"{trade['type']} {trade['quantity']}",
                    showlegend=False
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        height=config['ui']['chart_height'],
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_llm_activity():
    """Render LLM decision activity"""
    st.subheader("🧠 LLM Activity")
    
    if st.session_state.last_decision:
        decision = st.session_state.last_decision
        
        # Decision box
        decision_color = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }.get(decision['action'], '⚪')
        
        st.markdown(f"### {decision_color} Last Decision: **{decision['action']}**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Reasoning:**")
            st.info(decision['reasoning'])
        
        with col2:
            st.metric("Confidence", f"{decision['confidence']:.0%}")
            st.metric("Risk Level", decision.get('risk_level', 'N/A'))
            if decision.get('quantity', 0) > 0:
                st.metric("Quantity", f"{decision['quantity']} shares")


def render_trades_table():
    """Render recent trades table"""
    st.subheader("📊 Recent Trades")
    
    if not st.session_state.trades_log:
        st.info("No trades executed yet")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.trades_log[-10:])  # Last 10 trades
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Format columns
    display_df = df[['timestamp', 'type', 'quantity', 'price', 'total_cost', 'cash_after']].copy()
    display_df.columns = ['Time', 'Type', 'Qty', 'Price (₹)', 'Total (₹)', 'Cash After (₹)']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_memory_bank():
    """Render memory bank insights"""
    if not st.session_state.memory_bank:
        return
    
    st.subheader("🧠 Memory Bank")
    
    stats = st.session_state.memory_bank.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Decisions", stats['total_decisions'])
    with col2:
        st.metric("Learnings", stats['total_learnings'])
    with col3:
        st.metric("Patterns", stats['total_patterns'])
    with col4:
        st.metric("Mistakes", stats['total_mistakes'])
    
    # Show recent learnings
    with st.expander("📚 Recent Learnings"):
        learnings = st.session_state.memory_bank.get_all_learnings()
        if learnings:
            for learning in learnings[-5:]:
                st.markdown(f"- {learning['learning']}")
        else:
            st.info("No learnings recorded yet")


def process_simulation_step():
    """Process one simulation step (LLM decision making)"""
    if not st.session_state.initialized or not st.session_state.simulation_started:
        return
    
    if st.session_state.simulator.is_paused or not st.session_state.simulator.is_running:
        return
    
    # Get current data
    current_data = st.session_state.simulator.get_current_data()
    if current_data is None:
        return
    
    current_time = st.session_state.simulator.get_current_time()
    if current_time is None:
        return
    
    # Get available historical data
    lookback_data = st.session_state.simulator.get_available_data()
    
    # Get portfolio summary
    current_prices = {config['trading']['stock_symbol']: current_data['Close']}
    portfolio_summary = st.session_state.portfolio.get_summary(current_prices)
    
    # Calculate max affordable
    max_affordable = st.session_state.executor.get_max_affordable_quantity(
        config['trading']['stock_symbol'],
        current_data['Close']
    )
    
    # LLM makes decision
    decision = st.session_state.llm_agent.make_decision(
        market_data=current_data,
        portfolio_summary=portfolio_summary,
        current_time=current_time,
        max_affordable=max_affordable,
        lookback_data=lookback_data
    )
    
    st.session_state.last_decision = decision
    st.session_state.decisions_log.append({
        'timestamp': current_time,
        'decision': decision
    })
    
    # Execute trade if not HOLD
    if decision['action'] != 'HOLD' and decision['quantity'] > 0:
        if decision['action'] == 'BUY':
            result = st.session_state.executor.execute_market_buy(
                symbol=config['trading']['stock_symbol'],
                quantity=decision['quantity'],
                current_price=current_data['Close'],
                timestamp=current_time,
                reason=decision['reasoning']
            )
        else:  # SELL
            result = st.session_state.executor.execute_market_sell(
                symbol=config['trading']['stock_symbol'],
                quantity=decision['quantity'],
                current_price=current_data['Close'],
                timestamp=current_time,
                reason=decision['reasoning']
            )
        
        if result['success']:
            st.session_state.trades_log.append(result['transaction'])
            
            # LLM reflection
            st.session_state.llm_agent.reflect_on_trade(
                trade_result=result,
                portfolio_after=st.session_state.portfolio.get_summary(current_prices)
            )
    
    # Record performance snapshot
    st.session_state.performance.record_snapshot(
        timestamp=current_time,
        portfolio_value=portfolio_summary['total_value'],
        cash=portfolio_summary['current_cash'],
        positions_value=portfolio_summary['positions_value'],
        positions=portfolio_summary['current_positions'],
        current_prices=current_prices
    )


def main():
    """Main application"""
    # Configure page
    st.set_page_config(
        page_title="Emporio - LLM Trading Simulator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load configuration
    global config
    config = load_config()
    
    # Initialize session state
    initialize_session_state(config)
    
    # Render UI
    render_header()
    render_control_panel()
    
    if st.session_state.initialized:
        # Process simulation step (autonomous LLM trading)
        process_simulation_step()
        
        # Render main content
        render_metrics()
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_price_chart()
        
        with col2:
            render_llm_activity()
        
        st.markdown("---")
        render_trades_table()
        
        st.markdown("---")
        render_memory_bank()
        
        # Auto-refresh
        if st.session_state.simulation_started and not st.session_state.simulator.is_paused:
            st.rerun()


if __name__ == "__main__":
    main()
