"""
News Search Tool - Time-aware news search for trading decisions
"""
import os
import logging
from typing import List, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


class NewsSearchTool:
    """
    Searches for news articles relevant to trading decisions.
    Supports time-aware search to avoid looking into the future.
    """
    
    def __init__(self, api_key: Optional[str] = None, enabled: bool = True):
        """
        Initialize NewsSearchTool
        
        Args:
            api_key: NewsAPI key (from config.yaml)
            enabled: Whether news search is enabled
        """
        self.api_key = api_key
        self.enabled = enabled and self.api_key is not None and self.api_key not in ['YOUR_NEWSAPI_KEY_HERE', 'your_newsapi_key_here']
        
        if not self.enabled:
            if enabled and not self.api_key:
                logger.warning("News search disabled: no API key provided in config.yaml")
            else:
                logger.warning("News search is disabled in config")
        else:
            logger.info("News search tool initialized")
    
    def search(
        self,
        query: str,
        from_date: datetime,
        to_date: datetime,
        language: str = "en",
        sort_by: str = "relevancy",
        max_results: int = 5
    ) -> List[dict]:
        """
        Search for news articles within a time range
        
        Args:
            query: Search query
            from_date: Start date for search
            to_date: End date for search (must not be in future of simulation)
            language: Language code (en, hi, etc.)
            sort_by: Sort order (relevancy, popularity, publishedAt)
            max_results: Maximum number of articles to return
        
        Returns:
            List of news articles with title, description, url, publishedAt
        """
        if not self.enabled:
            logger.debug("News search disabled, returning empty results")
            return []
        
        try:
            # Use NewsAPI's everything endpoint for historical search
            url = "https://newsapi.org/v2/everything"
            
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "language": language,
                "sortBy": sort_by,
                "pageSize": max_results,
                "apiKey": self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get("articles", [])
            
            # Format results
            results = []
            for article in articles:
                results.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown")
                })
            
            logger.info(f"Found {len(results)} news articles for query: '{query}'")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching news: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in news search: {str(e)}")
            return []
    
    def search_for_trading(
        self,
        keywords: List[str],
        current_date: datetime,
        lookback_days: int = 7,
        max_results_per_keyword: int = 3
    ) -> str:
        """
        Search news relevant for trading decisions
        
        Args:
            keywords: List of search keywords
            current_date: Current simulation date (not future)
            lookback_days: How many days back to search
            max_results_per_keyword: Max results per keyword
        
        Returns:
            Formatted news context string
        """
        if not self.enabled:
            return "News search is disabled."
        
        from_date = current_date - timedelta(days=lookback_days)
        to_date = current_date
        
        all_articles = []
        
        for keyword in keywords:
            articles = self.search(
                query=keyword,
                from_date=from_date,
                to_date=to_date,
                max_results=max_results_per_keyword
            )
            all_articles.extend(articles)
        
        if not all_articles:
            return "No relevant news found in the specified time period."
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article["title"]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Format news context
        news_lines = [f"Found {len(unique_articles)} relevant news articles:\n"]
        
        for i, article in enumerate(unique_articles[:10], 1):  # Limit to 10 articles
            news_lines.append(f"{i}. {article['title']}")
            news_lines.append(f"   Source: {article['source']} | Published: {article['published_at']}")
            if article['description']:
                desc = article['description'][:150] + "..." if len(article['description']) > 150 else article['description']
                news_lines.append(f"   {desc}")
            news_lines.append("")
        
        return "\n".join(news_lines)
    
    def is_enabled(self) -> bool:
        """Check if news search is enabled"""
        return self.enabled
    
    def toggle(self, enabled: bool):
        """Toggle news search on/off"""
        if enabled and not self.api_key:
            logger.warning("Cannot enable news search without API key")
            return
        self.enabled = enabled
        logger.info(f"News search {'enabled' if enabled else 'disabled'}")
