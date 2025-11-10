"""
Main Orchestrator - Entry point for Emporio trading simulator
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import logging
from pathlib import Path

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)

# Add unified debug log file capturing all levels for deeper inspection
unified_log_path = log_dir / 'log.txt'
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(unified_log_path) for h in logging.getLogger().handlers):
    unified_handler = logging.FileHandler(unified_log_path)
    unified_handler.setLevel(logging.DEBUG)
    unified_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'))
    logging.getLogger().addHandler(unified_handler)
    logging.getLogger(__name__).info(f"Unified debug log initialized at {unified_log_path}")

logger = logging.getLogger(__name__)


def check_environment():
    """Check if environment is properly configured"""
    import yaml
    
    issues = []
    
    # Check for config file
    if not Path("config/config.yaml").exists():
        issues.append("❌ config/config.yaml not found")
        return issues
    
    logger.info("✅ config.yaml found")
    
    # Check for API keys in config
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        gemini_key = config.get('llm', {}).get('api_key')
        if not gemini_key or gemini_key in ['your_gemini_api_key_here', 'YOUR_GEMINI_API_KEY_HERE']:
            issues.append("❌ GEMINI_API_KEY not configured in config.yaml (llm.api_key)")
        else:
            logger.info("✅ GEMINI_API_KEY configured")
        
        news_key = config.get('tools', {}).get('news_search', {}).get('api_key')
        if news_key and news_key not in ['your_newsapi_key_here', 'YOUR_NEWSAPI_KEY_HERE']:
            logger.info("✅ NEWS_API_KEY configured")
        else:
            logger.warning("⚠️  NEWS_API_KEY not configured (news search will be disabled)")
    except Exception as e:
        issues.append(f"❌ Error reading config.yaml: {str(e)}")
    
    return issues


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("🤖 Emporio - LLM Stock Trading Simulator")
    logger.info("=" * 60)
    
    # Check environment
    logger.info("\n📋 Checking environment...")
    issues = check_environment()
    
    if issues:
        logger.error("\n⚠️  Environment check failed:")
        for issue in issues:
            logger.error(f"  {issue}")
        logger.error("\nPlease fix the issues above before running the simulator.")
        logger.error("Refer to README.md for setup instructions.")
        sys.exit(1)
    
    logger.info("\n✅ Environment check passed!")
    logger.info("\n🚀 Starting Streamlit UI...")
    logger.info("=" * 60)
    
    # Launch Streamlit app
    import subprocess
    
    app_path = Path("src/ui/app.py")
    subprocess.run([
        "streamlit", "run", str(app_path),
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
