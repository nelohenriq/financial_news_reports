import os
import json
import logging
import asyncio
import spacy
import structlog
from redis import Redis
from openai import OpenAI
from tavily import TavilyClient
from src.config import Settings
from src.data_fetcher import DataFetcher
from src.sentiment_analysis import SentimentAnalyzer
from src.text_processor import TextProcessor
from src.image_generator import ImageGenerator
from src.content_generator import ContentGenerator
from src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = structlog.get_logger()


class MarketAnalyzer:
    def __init__(self):
        self.config = Settings()
        self.nlp = spacy.load("en_core_web_sm")
        self.redis = Redis.from_url(self.config.REDIS_URL)
        self.logger = logger
        self.tavily_client = TavilyClient(api_key=self.config.TAVILY_API_KEY)
        self.openai_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.config.GROQ_API_KEY,
            max_retries=5,
            timeout=30,
        )
        self.data_fetcher = DataFetcher(self.config, self.redis)
        self.sentiment_analyzer = SentimentAnalyzer(self.nlp)
        self.text_processor = TextProcessor(self.nlp)
        self.image_generator = ImageGenerator(self.config)
        self.content_generator = ContentGenerator(
            self.config, self.openai_client, self.text_processor
        )
        self.report_generator = ReportGenerator(
            self.config, self.logger, self.image_generator
        )

    async def fetch_news(self, ticker):
        async def fetch():
            results = self.tavily_client.search(
                query=f"Latest {ticker} news and analysis",
                search_depth="advanced",
                max_results=10,
            )
            return [
                {
                    "title": r["title"],
                    "content": r["content"],
                    "url": r["url"],
                    "published_date": r.get("published_date"),
                }
                for r in results["results"]
            ]

        return await self.data_fetcher.get_cached_data(f"news_{ticker}", fetch)

    async def process_ticker(self, category, ticker):
        try:
            self.logger.info("Processing ticker", category=category, ticker=ticker)
            if category == "crypto":
                ticker_data = await self.data_fetcher.fetch_crypto_data(ticker)
            elif category == "forex":
                base_currency, target_currency = ticker.split("-")
                ticker_data = await self.data_fetcher.fetch_forex_data(
                    base_currency, target_currency
                )
            else:
                ticker_data = await self.data_fetcher.fetch_ticker_data(ticker)
            news = await self.fetch_news(ticker)
            sentiment = await self.sentiment_analyzer.analyze_sentiment(
                [article["content"] for article in news]
            )
            content = await self.content_generator.generate_content(
                ticker_data, sentiment, news
            )
            await self.report_generator.save_report(category, ticker, content)
            self.logger.info(
                "Ticker processed successfully", category=category, ticker=ticker
            )
            await asyncio.sleep(5)  # Rate limiting
        except Exception as e:
            self.logger.error(
                "Ticker processing failed",
                category=category,
                ticker=ticker,
                error=str(e),
            )

    async def run(self):
        try:
            self.logger.info("Starting market analysis")
            with open("portfolio.json", "r") as f:
                portfolio = json.load(f)

            tasks = [
                self.process_ticker(category, ticker)
                for category, tickers in portfolio.items()
                for ticker in tickers
            ]

            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info("Market analysis completed")
        except Exception as e:
            self.logger.error("Market analysis run failed", error=str(e))


if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    asyncio.run(analyzer.run())
