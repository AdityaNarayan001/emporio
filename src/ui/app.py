"""
Streamlit UI - Main application interface for LLM Stock Trading Simulator
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import threading
import time as time_module

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
        st.session_state.llm_thinking = False  # Loading indicator for LLM
        st.session_state.last_llm_decision_time = None  # Track when LLM last made a decision
        st.session_state.llm_decision_thread = None  # Background thread for LLM decisions


def initialize_components(config):
    """Initialize all trading components"""
    try:
        logger.info("=== INITIALIZATION STARTED ===")
        with st.spinner("Initializing trading system..."):
            # Load historical data
            logger.info("Loading DataLoader...")
            st.session_state.data_loader = DataLoader(
                symbol=config['trading']['stock_symbol'],
                cache_dir="data"
            )
            
            # Fetch data
            logger.info(f"Fetching historical data from {config['simulation']['start_date']} to {config['simulation']['end_date']}")
            historical_data = st.session_state.data_loader.fetch_historical_data(
                start_date=config['simulation']['start_date'],
                end_date=config['simulation']['end_date'],
                period_days=config['simulation']['historical_period_days'],
                interval=config['simulation']['data_interval']
            )
            logger.info(f"Loaded {len(historical_data)} data points")
            
            # Initialize simulator
            logger.info(f"Initializing DataSimulator with speed {config['simulation']['simulation_speed']}x")
            tick_mode = config['simulation'].get('tick_mode', True)
            tick_interval = config['simulation'].get('tick_interval', 2.0)
            logger.info(f"Simulator mode: {'TICK (new data every ' + str(tick_interval) + 's)' if tick_mode else 'TIME-BASED'}")
            
            st.session_state.simulator = DataSimulator(
                historical_data=historical_data,
                speed_multiplier=config['simulation']['simulation_speed'],
                max_lookback=config['simulation']['max_lookback_period'],
                tick_mode=tick_mode,
                tick_interval=tick_interval
            )
            
            # Initialize portfolio
            logger.info(f"Initializing Portfolio with ₹{config['trading']['initial_capital']}")
            st.session_state.portfolio = Portfolio(
                initial_capital=config['trading']['initial_capital'],
                commission_rate=config['trading']['commission_percentage']
            )
            
            # Initialize executor
            logger.info("Initializing OrderExecutor")
            st.session_state.executor = OrderExecutor(st.session_state.portfolio)
            
            # Initialize performance tracker
            logger.info("Initializing PerformanceTracker")
            st.session_state.performance = PerformanceTracker()
            
            # Initialize memory bank
            logger.info(f"Initializing MemoryBank at {config['memory']['file_path']}")
            st.session_state.memory_bank = MemoryBank(
                file_path=config['memory']['file_path'],
                max_entries=config['memory']['max_entries']
            )
            
            # Initialize news tool
            logger.info(f"Initializing NewsSearchTool (enabled: {config['tools']['news_search']['enabled']})")
            st.session_state.news_tool = NewsSearchTool(
                api_key=config['tools']['news_search'].get('api_key'),
                enabled=config['tools']['news_search']['enabled']
            )
            
            # Initialize LLM agent
            logger.info(f"Initializing LLM Agent with model {config['llm']['model_name']}")
            st.session_state.llm_agent = LLMTradingAgent(
                api_key=config['llm']['api_key'],
                model_name=config['llm']['model_name'],
                temperature=config['llm']['temperature'],
                memory_bank=st.session_state.memory_bank,
                news_tool=st.session_state.news_tool,
                min_confidence=config['llm']['min_confidence_threshold']
            )
            
            st.session_state.initialized = True
            logger.info("=== INITIALIZATION COMPLETED SUCCESSFULLY ===")
            st.success("✅ All components initialized successfully!")
            
    except Exception as e:
        logger.error(f"❌ Initialization error: {str(e)}", exc_info=True)
        st.error(f"❌ Error initializing components: {str(e)}")


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
        if st.sidebar.button("🚀 Initialize System", key="init_btn"):
            logger.info("USER ACTION: Initialize System button clicked")
            initialize_components(config)
        return
    
    # Start/Stop simulation
    if not st.session_state.simulation_started:
        if st.sidebar.button("▶️ Start Simulation", key="start_btn"):
            logger.info("USER ACTION: Start Simulation button clicked")
            st.session_state.simulator.start()
            st.session_state.simulation_started = True
            st.session_state.last_llm_decision_time = datetime.now()
            logger.info(f"Simulation started at speed {st.session_state.simulator.speed_multiplier}x")
            
            # Start LLM decision thread in continuous mode
            if config.get('simulation', {}).get('continuous_mode', True):
                start_llm_decision_thread()
            
            st.rerun()
    else:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.session_state.simulator.is_paused:
                if st.button("▶️ Resume", key="resume_btn"):
                    logger.info("USER ACTION: Resume button clicked")
                    st.session_state.simulator.resume()
                    st.rerun()
            else:
                if st.button("⏸️ Pause", key="pause_btn"):
                    logger.info("USER ACTION: Pause button clicked")
                    st.session_state.simulator.pause()
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", key="stop_btn"):
                logger.info("USER ACTION: Stop button clicked")
                st.session_state.simulator.stop()
                st.session_state.simulation_started = False
                logger.info("Simulation stopped")
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
            logger.info(f"USER ACTION: Speed changed from {st.session_state.simulator.speed_multiplier}x to {speed}x")
            st.session_state.simulator.set_speed(speed)
    
    # News toggle
    if st.session_state.initialized and st.session_state.news_tool:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📰 News Search")
        news_enabled = st.sidebar.checkbox(
            "Enable News Search",
            value=st.session_state.news_tool.is_enabled()
        )
        if news_enabled != st.session_state.news_tool.is_enabled():
            logger.info(f"USER ACTION: News search toggled to {news_enabled}")
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
    
    logger.debug(f"UI RENDER: Metrics at price ₹{current_data['Close']:.2f}, progress {progress['progress_percentage']:.1f}%")
    
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
    """Render price and portfolio value chart with buy/sell markers"""
    if not st.session_state.initialized or not st.session_state.simulator:
        st.warning("⚠️ System not initialized. Click 'Initialize System' in sidebar.")
        logger.warning("UI RENDER: Cannot render chart - system not initialized")
        return
    
    available_data = st.session_state.simulator.get_available_data()
    
    if available_data is None or len(available_data) == 0:
        st.info("⏳ Waiting for market data... Click 'Start Simulation' in sidebar.")
        logger.warning("UI RENDER: Cannot render chart - no data available")
        return
    
    logger.info(f"UI RENDER: Chart with {len(available_data)} total data points available")
    
    # Ensure Datetime column exists and is properly formatted
    if 'Datetime' not in available_data.columns:
        logger.error("UI RENDER: 'Datetime' column missing from data!")
        st.error("❌ Data format error: Missing timestamp information")
        return
    
    # Show last 60 data points for smooth scrolling
    window_size = 60
    if len(available_data) > window_size:
        display_data = available_data.tail(window_size).copy()
        logger.info(f"UI RENDER: Displaying last {window_size} of {len(available_data)} points")
    else:
        display_data = available_data.copy()
        logger.info(f"UI RENDER: Displaying all {len(display_data)} points")
    
    # Verify required columns
    required_cols = ['Datetime', 'Close', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_cols if col not in display_data.columns]
    if missing_cols:
        logger.error(f"UI RENDER: Missing required columns: {missing_cols}")
        st.error(f"❌ Data format error: Missing columns {missing_cols}")
        return
    
    logger.info(f"UI RENDER: Price range ₹{display_data['Close'].min():.2f} - ₹{display_data['Close'].max():.2f}, {len(display_data)} points displayed")
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('📈 IRCTC Stock Price', '💰 Portfolio Value'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # 1. Price line chart (main chart)
    fig.add_trace(
        go.Scatter(
            x=display_data['Datetime'],
            y=display_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#2196f3', width=2),
            hovertemplate='Price: ₹%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add BUY markers
    buy_trades = [t for t in st.session_state.trades_log if t['type'] == 'BUY']
    if buy_trades:
        logger.debug(f"UI RENDER: {len(buy_trades)} BUY trades in history")
        display_times = set(display_data['Datetime'].dt.strftime('%Y-%m-%d'))
        buy_times = []
        buy_prices = []
        buy_quantities = []
        
        for t in buy_trades:
            trade_time = pd.to_datetime(t['timestamp']).strftime('%Y-%m-%d')
            if trade_time in display_times:
                buy_times.append(pd.to_datetime(t['timestamp']))
                buy_prices.append(t['price'])
                buy_quantities.append(t['quantity'])
        
        if buy_times:
            logger.debug(f"UI RENDER: Showing {len(buy_times)} BUY markers on chart")
            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='triangle-up', line=dict(width=2, color='white')),
                    name='BUY',
                    text=[f'BUY {q}' for q in buy_quantities],
                    hovertemplate='<b>BUY</b><br>Price: ₹%{y:.2f}<br>%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add SELL markers
    sell_trades = [t for t in st.session_state.trades_log if t['type'] == 'SELL']
    if sell_trades:
        logger.debug(f"UI RENDER: {len(sell_trades)} SELL trades in history")
        display_times = set(display_data['Datetime'].dt.strftime('%Y-%m-%d'))
        sell_times = []
        sell_prices = []
        sell_quantities = []
        
        for t in sell_trades:
            trade_time = pd.to_datetime(t['timestamp']).strftime('%Y-%m-%d')
            if trade_time in display_times:
                sell_times.append(pd.to_datetime(t['timestamp']))
                sell_prices.append(t['price'])
                sell_quantities.append(t['quantity'])
        
        if sell_times:
            logger.debug(f"UI RENDER: Showing {len(sell_times)} SELL markers on chart")
            fig.add_trace(
                go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='triangle-down', line=dict(width=2, color='white')),
                    name='SELL',
                    text=[f'SELL {q}' for q in sell_quantities],
                    hovertemplate='<b>SELL</b><br>Price: ₹%{y:.2f}<br>%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # 2. Portfolio value chart
    if st.session_state.performance and len(st.session_state.performance.performance_history) > 0:
        perf_df = st.session_state.performance.get_performance_df()
        logger.debug(f"UI RENDER: Portfolio history with {len(perf_df)} snapshots")
        
        # Filter to match display window
        if len(display_data) > 0:
            start_time = display_data['Datetime'].iloc[0]
            end_time = display_data['Datetime'].iloc[-1]
            perf_df = perf_df[
                (perf_df['timestamp'] >= start_time) & 
                (perf_df['timestamp'] <= end_time)
            ]
        
        if len(perf_df) > 0:
            current_portfolio_value = perf_df['portfolio_value'].iloc[-1]
            logger.debug(f"UI RENDER: Current portfolio value ₹{current_portfolio_value:.2f}")
            fig.add_trace(
                go.Scatter(
                    x=perf_df['timestamp'],
                    y=perf_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#4caf50', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(76, 175, 80, 0.1)',
                    hovertemplate='Portfolio: ₹%{y:,.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Initial capital line
            initial_capital = config['trading']['initial_capital']
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial",
                row=2, col=1
            )
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Value (₹)", row=2, col=1)
    
    st.plotly_chart(fig, key="irctc_chart")


def render_llm_activity():
    """Render LLM decision activity"""
    st.subheader("🧠 LLM Activity")
    
    # Show loading indicator when LLM is thinking
    if st.session_state.get('llm_thinking', False):
        st.info("� Thinking... Analyzing market data, news, and patterns...")
    
    # Show next decision countdown in continuous mode
    if st.session_state.get('simulation_started') and config.get('simulation', {}).get('continuous_mode', True):
        if st.session_state.last_llm_decision_time:
            interval = config['simulation'].get('llm_check_interval', 30)
            next_check = st.session_state.last_llm_decision_time + timedelta(seconds=interval)
            time_until = (next_check - datetime.now()).total_seconds()
            
            if time_until > 0:
                st.caption(f"⏱️ Next market check in: {int(time_until)}s")
            else:
                st.caption("🔄 Analyzing market now...")
        else:
            st.caption("🔄 Preparing first market analysis...")
    
    if st.session_state.last_decision:
        decision = st.session_state.last_decision
        
        # Decision box with color coding
        decision_color = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }.get(decision['action'], '⚪')
        
        # Action badge
        action_style = {
            'BUY': 'background-color: #28a745; color: white; padding: 5px 15px; border-radius: 5px; font-weight: bold;',
            'SELL': 'background-color: #dc3545; color: white; padding: 5px 15px; border-radius: 5px; font-weight: bold;',
            'HOLD': 'background-color: #ffc107; color: black; padding: 5px 15px; border-radius: 5px; font-weight: bold;'
        }.get(decision['action'], '')
        
        st.markdown(f"### {decision_color} Last Decision")
        st.markdown(f"<span style='{action_style}'>{decision['action']}</span>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**💭 Reasoning:**")
            st.info(decision['reasoning'])
            
            # Show expected outcome if available
            if decision.get('expected_outcome'):
                st.markdown(f"**🎯 Expected Outcome:** {decision['expected_outcome']}")
        
        with col2:
            st.metric("Confidence", f"{decision['confidence']:.0%}")
            st.metric("Risk Level", decision.get('risk_level', 'N/A'))
            if decision.get('quantity', 0) > 0:
                st.metric("Quantity", f"{decision['quantity']} shares")
            
            # Show stop loss and take profit if available
            if decision.get('stop_loss'):
                st.markdown(f"**🛑 Stop Loss:** ₹{decision['stop_loss']:.2f}")
            if decision.get('take_profit'):
                st.markdown(f"**✅ Take Profit:** ₹{decision['take_profit']:.2f}")
    else:
        st.info("💤 Waiting for first trading decision...")


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
    
    st.dataframe(display_df, hide_index=True, key="trades_table")


def render_memory_bank():
    """Render memory bank insights with better organization"""
    if not st.session_state.memory_bank:
        return
    
    st.subheader("🧠 AI Memory & Learning System")
    
    stats = st.session_state.memory_bank.get_statistics()
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💡 Decisions", stats['total_decisions'])
    with col2:
        st.metric("📚 Learnings", stats['total_learnings'])
    with col3:
        st.metric("🔍 Patterns", stats['total_patterns'])
    with col4:
        st.metric("⚠️ Mistakes", stats['total_mistakes'])
    
    # Tabbed interface for different memory categories
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Recent Decisions", "💡 Key Learnings", "🔍 Patterns", "⚠️ Mistakes"])
    
    with tab1:
        st.markdown("### Recent Trading Decisions")
        recent = st.session_state.memory_bank.get_recent_decisions(10)
        if recent:
            for i, d in enumerate(reversed(recent), 1):
                action = d.get("action", "UNKNOWN")
                confidence = d.get("confidence", 0)
                reasoning = d.get("reasoning", "")
                
                # Color code by action
                if "BUY" in action.upper():
                    icon = "🟢"
                    color = "#d4edda"
                elif "SELL" in action.upper():
                    icon = "🔴"
                    color = "#f8d7da"
                else:
                    icon = "🟡"
                    color = "#fff3cd"
                
                with st.expander(f"{icon} Decision #{len(recent)-i+1}: {action} (Confidence: {confidence:.0%})"):
                    st.markdown(f"**Reasoning:** {reasoning[:200]}...")
                    if d.get('market_data'):
                        price = d['market_data'].get('price', 0)
                        st.markdown(f"**Price at decision:** ₹{price:.2f}")
        else:
            st.info("No decisions recorded yet")
    
    with tab2:
        st.markdown("### Key Learnings from Experience")
        learnings = st.session_state.memory_bank.get_all_learnings()
        if learnings:
            for i, l in enumerate(reversed(learnings[-10:]), 1):
                st.markdown(f"**{i}.** {l.get('learning', '')}")
                st.caption(f"_Recorded: {l.get('timestamp', '')}_ ")
                st.divider()
        else:
            st.info("No learnings recorded yet. The AI will learn from its trading experiences.")
    
    with tab3:
        st.markdown("### Identified Market Patterns")
        patterns = st.session_state.memory_bank.get_patterns()
        if patterns:
            for p in reversed(patterns[-5:]):
                effectiveness = p.get('effectiveness', 0)
                color = "🟢" if effectiveness > 0.7 else "🟡" if effectiveness > 0.4 else "🔴"
                st.markdown(f"{color} **{p.get('pattern', '')}** (Effectiveness: {effectiveness:.0%})")
                st.write(p.get('description', ''))
                st.divider()
        else:
            st.info("No patterns identified yet. The AI will recognize patterns as it trades.")
    
    with tab4:
        st.markdown("### Mistakes & Lessons Learned")
        mistakes = st.session_state.memory_bank.get_mistakes()
        if mistakes:
            for m in reversed(mistakes[-5:]):
                st.error(f"**Mistake:** {m.get('mistake', '')}")
                st.success(f"**Lesson:** {m.get('lesson', '')}")
                st.divider()
        else:
            st.success("No major mistakes recorded yet. The AI is trading carefully!")


def llm_decision_loop():
    """Background thread that continuously checks market and makes decisions like a real trader"""
    logger.info("LLM decision loop started - will check market every 30s after each decision")
    
    # Wait for some initial data to accumulate (at least 5 data points)
    logger.info("⏳ Waiting for initial market data to accumulate...")
    initial_wait_time = 0
    while st.session_state.get('simulation_started', False):
        available_data = st.session_state.simulator.get_available_data()
        if len(available_data) >= 5:
            logger.info(f"✅ Initial data ready: {len(available_data)} points available")
            break
        time_module.sleep(1)
        initial_wait_time += 1
        if initial_wait_time >= 10:
            logger.warning(f"⚠️ Proceeding with only {len(available_data)} data points after 10s wait")
            break
    
    while st.session_state.get('simulation_started', False):
        try:
            # Check if simulation is paused
            if st.session_state.simulator.is_paused or not st.session_state.simulator.is_running:
                time_module.sleep(1)
                continue
            
            # Wait for check interval after last decision (30 seconds by default)
            check_interval = config['simulation'].get('llm_check_interval', 30)
            current_time = datetime.now()
            
            if st.session_state.last_llm_decision_time:
                time_since_last = (current_time - st.session_state.last_llm_decision_time).total_seconds()
                if time_since_last < check_interval:
                    time_module.sleep(1)
                    continue
                logger.info(f"⏰ {int(time_since_last)}s elapsed since last check - analyzing market now...")
            else:
                logger.info("🤖 First market check - analyzing current state...")
            
            logger.info("📊 LLM checking current market state...")
            
            # Get current market data
            current_data = st.session_state.simulator.get_current_data()
            if current_data is None:
                logger.warning("⚠️ No current data available, waiting...")
                time_module.sleep(1)
                continue
            
            sim_time = st.session_state.simulator.get_current_time()
            if sim_time is None:
                logger.warning("⚠️ No simulation time available, waiting...")
                time_module.sleep(1)
                continue
            
            # Set loading state
            st.session_state.llm_thinking = True
            
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
            
            logger.info(f"💰 Making LLM decision at price ₹{current_data['Close']:.2f}, Portfolio: ₹{portfolio_summary['total_value']:.2f}")
            
            # LLM makes decision
            decision_start_time = datetime.now()
            decision = st.session_state.llm_agent.make_decision(
                market_data=current_data,
                portfolio_summary=portfolio_summary,
                current_time=sim_time,
                max_affordable=max_affordable,
                lookback_data=lookback_data
            )
            decision_time = (datetime.now() - decision_start_time).total_seconds()
            
            logger.info(f"🎯 LLM Decision: {decision['action']} with confidence {decision['confidence']:.0%} (took {decision_time:.1f}s)")
            
            # Clear loading state and record time
            st.session_state.llm_thinking = False
            st.session_state.last_llm_decision_time = datetime.now()
            logger.info(f"⏰ Next market check will be in 30 seconds at {(st.session_state.last_llm_decision_time + timedelta(seconds=30)).strftime('%H:%M:%S')}")
            
            st.session_state.last_decision = decision
            st.session_state.decisions_log.append({
                'timestamp': sim_time,
                'decision': decision
            })
            
            # Execute trade if not HOLD
            if decision['action'] != 'HOLD' and decision['quantity'] > 0:
                logger.info(f"TRADE EXECUTION: {decision['action']} {decision['quantity']} shares at ₹{current_data['Close']:.2f}")
                if decision['action'] == 'BUY':
                    result = st.session_state.executor.execute_market_buy(
                        symbol=config['trading']['stock_symbol'],
                        quantity=decision['quantity'],
                        current_price=current_data['Close'],
                        timestamp=sim_time,
                        reason=decision['reasoning']
                    )
                else:  # SELL
                    result = st.session_state.executor.execute_market_sell(
                        symbol=config['trading']['stock_symbol'],
                        quantity=decision['quantity'],
                        current_price=current_data['Close'],
                        timestamp=sim_time,
                        reason=decision['reasoning']
                    )
                
                if result['success']:
                    logger.info(f"TRADE SUCCESS: {decision['action']} {decision['quantity']} shares, total cost ₹{result['transaction']['total_cost']:.2f}")
                    st.session_state.trades_log.append(result['transaction'])
                    
                    # LLM reflection
                    st.session_state.llm_agent.reflect_on_trade(
                        trade_result=result,
                        portfolio_after=st.session_state.portfolio.get_summary(current_prices)
                    )
                else:
                    logger.warning(f"TRADE FAILED: {result.get('error', 'Unknown error')}")
            
            # Record performance snapshot
            st.session_state.performance.record_snapshot(
                timestamp=sim_time,
                portfolio_value=portfolio_summary['total_value'],
                cash=portfolio_summary['current_cash'],
                positions_value=portfolio_summary['positions_value'],
                positions=portfolio_summary['current_positions'],
                current_prices=current_prices
            )
            
        except Exception as e:
            logger.error(f"Error in LLM decision loop: {str(e)}", exc_info=True)
            st.session_state.llm_thinking = False
            time_module.sleep(5)  # Wait before retrying


def start_llm_decision_thread():
    """Start the background LLM decision thread"""
    if st.session_state.llm_decision_thread is None or not st.session_state.llm_decision_thread.is_alive():
        st.session_state.llm_decision_thread = threading.Thread(
            target=llm_decision_loop,
            daemon=True
        )
        st.session_state.llm_decision_thread.start()
        logger.info("LLM decision thread started")


def main():
    """Main application"""
    # Configure page
    st.set_page_config(
        page_title="Emporio - LLM Trading Simulator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    logger.info("=== UI REFRESH ===")
    
    # Load configuration
    global config
    config = load_config()
    
    # Initialize session state
    initialize_session_state(config)
    
    # Render UI
    render_header()
    render_control_panel()
    
    if st.session_state.initialized:
        # Render main content (LLM runs in background thread now)
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
        
        # Auto-refresh only when simulation is running and not paused
        if st.session_state.simulation_started and not st.session_state.simulator.is_paused:
            logger.debug("UI AUTO-REFRESH: Waiting 2 seconds before next refresh")
            time_module.sleep(2)  # Refresh every 2 seconds for smooth updates
            st.rerun()


if __name__ == "__main__":
    main()
