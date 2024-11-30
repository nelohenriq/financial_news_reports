import os
import json
import logging
import asyncio
import aiohttp
import random
import spacy
import time
import re
import html
import httpx
import yfinance as yf
import structlog
import numpy as np
import nest_asyncio
import string
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from openai import OpenAI
from PIL import Image
from forex_python.converter import CurrencyRates
from pycoingecko import CoinGeckoAPI
from textblob import TextBlob
from tavily import TavilyClient
from redis import Redis
from pydantic_settings import BaseSettings
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from fastapi import FastAPI, BackgroundTasks

logging.basicConfig(level=logging.DEBUG)


nest_asyncio.apply()


class Settings(BaseSettings):
    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    HF_API_TOKEN: str
    STABILITY_API_KEY: str
    DEBUG_LEVEL: str = "DEBUG"
    REDIS_URL: str
    PROMETHEUS_PORT: int = 8000
    BATCH_SIZE: int = 10
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"


class StructuredLogger:
    def __init__(self):
        self.logger = structlog.get_logger()

    def log_operation(self, operation: str, **kwargs):
        return self.logger.info(operation, **kwargs)


class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)
        self.cache_ttl = 3600

    async def get_or_set(self, key: str, fetch_func, ttl: Optional[int] = None):
        if cached := self.redis.get(key):
            return json.loads(cached)
        value = await fetch_func()
        self.redis.set(key, json.dumps(value), ex=ttl or self.cache_ttl)
        return value


class APIClient:
    def __init__(self, session: aiohttp.ClientSession, config: Settings):
        self.session = session
        self.config = config
        self._rate_limit_delay = 0.1

    async def fetch_with_retry(self, url: str, **kwargs):
        for attempt in range(3):
            try:
                async with self.session.get(url, **kwargs) as response:
                    return await response.json()
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)


class ImageService:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.api_url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
        self.headers = {
            "Authorization": f"Bearer {self.api_client.config.STABILITY_API_KEY}",
            "Accept": "application/json",  # Change to application/json if needed
        }

    async def generate_image(
        self,
        prompt: str,
        output_dir: str,
        output_format: str = "png",
        image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        mask_path: Optional[str] = None,
        strength: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Prepare form data
            data = {
                "prompt": prompt,
                "output_format": output_format,
            }
            if aspect_ratio:
                data["aspect_ratio"] = aspect_ratio
            if seed is not None:
                data["seed"] = seed
            if negative_prompt:
                data["negative_prompt"] = negative_prompt
            
            files = {}
            if image_path:
                files["image"] = open(image_path, 'rb')
                if strength is not None:
                    data["strength"] = strength
            
            if mask_path:
                files["mask"] = open(mask_path, 'rb')

            # Debug logging before sending request
            logging.debug("Sending FormData:")
            for field in data.items():
                name = field[0]
                value = field[1]
                logging.debug(f"Field name: {name}, Field value: {value}")


            # Send request
            logging.debug("Sending image generation request...")
            async with self.api_client.session.post(
                self.api_url,
                headers=self.headers,
                data=data,
            ) as response:
                if response.status == 200:
                    image_data = await response.read()
                    timestamp = int(time.time())
                    filename = f"image_{timestamp}.{output_format}"
                    output_path = os.path.join(output_dir, filename)

                    with open(output_path, "wb") as f:
                        f.write(image_data)

                    logging.info(f"Successfully generated image at {output_path}")
                    return Image.open(output_path), output_path
                
                # Handle errors
                error_content = await response.json()
                logging.error(f"Request failed with status {response.status}: {error_content}")
                return None, None

        except Exception as e:
            logging.error(f"Error during image generation: {str(e)}")
            return None, None


class DataPipeline:
    def __init__(self, services: Dict, config: Settings):
        self.tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
        self.config = config
        self.cache = services.get("cache")
        self.logger = services.get("logger")

    async def process_data(self, ticker: str) -> Dict:
        try:
            # Now using the actual ticker (e.g. BTC-USD) not the category (e.g. CRYPTO)
            ticker_data = await self._get_ticker_data(
                ticker
            )  # This will query for BTC-USD
            articles = await self._fetch_articles(
                ticker
            )  # This will search for BTC-USD news
            sentiment_data = await self._analyze_sentiment(articles)

            return {
                "ticker_data": ticker_data,
                "articles": articles,
                "sentiment_data": sentiment_data,
            }
        except Exception as e:
            self.logger.log_operation("data_pipeline_failed", error=str(e))
            raise

    async def _get_ticker_data(self, ticker: str) -> Dict:
        cache_key = f"ticker_data_{ticker}"

        def convert_numpy_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(v) for v in obj)
            return obj

        async def fetch():
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            history = ticker_obj.history(period="1mo")
            return convert_numpy_types(
                {
                    "info": info,
                    "price": history["Close"].iloc[-1],
                    "volume": history["Volume"].iloc[-1],
                    "change": (history["Close"].iloc[-1] - history["Close"].iloc[0])
                    / history["Close"].iloc[0],
                }
            )

        return await self.cache.get_or_set(cache_key, fetch)

    async def _fetch_articles(self, ticker: str) -> List[Dict]:
        cache_key = f"articles_{ticker}"

        async def fetch():
            search_results = self.tavily_client.search(
                query=f"Latest {ticker} news and analysis",
                search_depth="advanced",
                max_results=10,
            )
            return [
                {
                    "title": result["title"],
                    "content": result["content"],
                    "url": result["url"],
                    "published_date": result.get("published_date"),
                }
                for result in search_results["results"]
            ]

        return await self.cache.get_or_set(cache_key, fetch)

    async def _analyze_sentiment(self, articles: List[Dict]) -> Dict:
        combined_text = " ".join([article["content"] for article in articles])
        blob = TextBlob(combined_text)

        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "article_count": len(articles),
        }


class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    async def analyze(self, texts: List[str]) -> Dict[str, Any]:
        try:
            combined_text = " ".join(texts)
            blob = TextBlob(combined_text)
            textblob_sentiment = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            entity_sentiment = self._get_entity_sentiment(texts)
            temporal_sentiment = self._get_temporal_sentiment(texts)

            return {
                "overall": {
                    "polarity": textblob_sentiment,
                    "subjectivity": subjectivity,
                },
                "entities": entity_sentiment,
                "temporal": temporal_sentiment,
                "confidence_score": self._calculate_confidence(
                    textblob_sentiment, subjectivity
                ),
            }
        except Exception as e:
            raise

    def _get_textblob_sentiment(self, texts: List[str]) -> float:
        combined_text = " ".join(texts)
        blob = TextBlob(combined_text)
        return blob.sentiment.polarity

    def _get_entity_sentiment(self, texts: List[str]) -> Dict[str, float]:
        combined_text = " ".join(texts)
        doc = self.nlp(combined_text)
        entities = doc.ents
        entity_sentiments = {}
        for entity in entities:
            entity_text = entity.text
            blob = TextBlob(entity_text)
            sentiment = blob.sentiment.polarity
            entity_sentiments[entity_text] = sentiment
        return entity_sentiments

    def _get_temporal_sentiment(
        self, texts: List[str], num_periods: int = 3
    ) -> List[float]:
        total_texts = len(texts)
        if total_texts == 0:
            return []
        period_size = max(1, total_texts // num_periods)
        sentiments = []
        for i in range(num_periods):
            start = i * period_size
            end = (i + 1) * period_size if i != num_periods - 1 else total_texts
            period_texts = texts[start:end]
            combined_period_text = " ".join(period_texts)
            blob = TextBlob(combined_period_text)
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
        return sentiments

    def _calculate_confidence(self, polarity: float, subjectivity: float) -> float:
        # Confidence calculation: higher subjectivity means lower confidence
        # Confidence = 1 - subjectivity
        return 1 - subjectivity


class PortfolioProcessor:
    def __init__(self, services: Dict, logger: StructuredLogger):
        self.services = services
        self.logger = logger

    def read_portfolio(self) -> Dict[str, List[str]]:
        try:
            with open("portfolio.json", "r") as f:
                portfolio = json.load(f)
                # Log the actual tickers (not categories)
                self.logger.log_operation(
                    "portfolio_loaded",
                    tickers_count=sum(len(v) for v in portfolio.values()),
                )
                return portfolio  # Ensure this returns category-to-ticker mappings
        except Exception as e:
            self.logger.log_operation("portfolio_read_failed", error=str(e))
            return {"crypto": [], "forex": [], "stocks": []}


class CompleteBlogGenerator:
    def __init__(self):
        self.config = Settings()
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1", api_key=self.config.GROQ_API_KEY, max_retries=5, timeout=30
        )
        self.request_interval=10,
        self.last_request_time = 0
        self.nlp = spacy.load("en_core_web_sm")
        self.session = None
        self.setup_components()

    async def initialize_session(self):
        self.session = aiohttp.ClientSession()
        return self.session

    async def setup_async_components(self):
        self.session = await self.initialize_session()
        self.api_client = APIClient(self.session, self.config)
        
        self.services = {
            "logger": self.logger,
            "cache": self.cache,
            "api_client": self.api_client,
        }
        
        self.services.update({
            "data_pipeline": DataPipeline(self.services, self.config),
            "sentiment": EnhancedSentimentAnalyzer(),
            "portfolio": PortfolioProcessor(self.services, self.logger),
            "image": ImageService(self.api_client),
        })

    def setup_components(self):
        self.logger = StructuredLogger()
        self.cache = CacheManager(self.config.REDIS_URL)

    async def save_blog_post(self, category: str, ticker: str, content: Dict):
        try:
            output_dir = f"output/{category}/posts"
            os.makedirs(output_dir, exist_ok=True)

            filename = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            output_path = os.path.join(output_dir, filename)

            with open(output_path, "w") as f:
                # Title and timestamp
                f.write(f"# {content['title']}\n\n")
                f.write(f"**Timestamp:** {content['timestamp']}\n\n")

                # Display the generated image with a descriptive caption
                if content.get("image_path"):
                    f.write("## Market Analysis Visualization\n\n")
                    f.write(
                        f"![Market Analysis for {ticker}]({content['image_path']})\n\n"
                    )
                    f.write(
                        f"*Generated visualization based on market analysis for {ticker}*\n\n"
                    )

                # Rest of the content
                f.write(f"**Content:**\n\n{content['content']}\n\n")

                # Sentiment analysis
                f.write("**Sentiment Analysis:**\n\n")
                sentiment = content.get("sentiment", {})
                f.write(f"- Polarity: {sentiment.get('polarity', 'N/A')}\n")
                f.write(f"- Subjectivity: {sentiment.get('subjectivity', 'N/A')}\n")
                f.write(f"- Article Count: {sentiment.get('article_count', 'N/A')}\n\n")

                f.write(f"**Image Prompt:**\n\n{content.get('image_prompt', '')}\n\n")

            logging.info(f"Saved blog post for {ticker} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save blog post for {ticker}: {e}")

    def clean_html(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text()
        text = re.sub(r"\s+", " ", text).strip()
        text = html.unescape(text)
        return text

    def extract_nouns(self, text: str) -> List[str]:
        # Convert dict to string if needed
        if isinstance(text, dict):
            text = str(text)
        doc = self.nlp(text)
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        return list(set(nouns))

    def summarize_with_spacy(self, text: str, max_sentences: int = 3) -> str:
        # Convert dict to string if needed
        if isinstance(text, dict):
            text = str(text)
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        word_freq = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                word_freq[token.text] = word_freq.get(token.text, 0) + 1

        sentence_scores = {}
        for sent in sentences:
            score = sum(word_freq.get(word, 0) for word in sent.split())
            sentence_scores[sent] = score

        summary_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )[:max_sentences]

        return " ".join(sent[0] for sent in summary_sentences)

    def generate_refined_prompt_for_blog_post(self, post_content: str) -> str:
        summary = self.summarize_with_spacy(post_content)
        nouns = self.extract_nouns(summary)
        key_elements = ", ".join(nouns[:7])

        prompt = f"""
        Create a visually-rich prompt for AI image generation based on the summary of this blog post:

        {summary}

        Key visual themes: {key_elements}

        The prompt should:
        1. Be concise and focused on visual elements.
        2. Seamlessly incorporate key visual concepts without explicitly listing them.
        3. Maintain a professional tone inline with {summary}.
        4. Be optimized for AI image generation models, emphasizing clarity and detail.

        Please provide only the refined prompt without any additional commentary or explanation.
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.2-3b-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating prompts for AI image generation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.7,
            )

            if response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                "A modern illustration representing key themes from the blog post."

        except Exception as e:
            self.logger.log_operation("prompt_generation_failed", error=str(e))
            return "A modern illustration representing key themes from the blog post."

    async def generate_content(self, data: Dict, sentiment_data: Dict) -> Dict:
        
        """Generate blog post content using the OpenAI API"""
        try:
            # Extract sentiment values directly from sentiment_data
            sentiment_value = data["sentiment_data"]["polarity"]
            confidence_value = data["sentiment_data"]["subjectivity"]

            # Get ticker data directly
            ticker_name = data["ticker_data"]["info"]["shortName"]
            ticker_price = float(data["ticker_data"]["price"])
            ticker_change = float(data["ticker_data"]["change"])

            # Get article titles with list comprehension
            article_titles = " ".join(
                article["title"] for article in data["articles"][:3]
            )

            prompt = f"""
            Write a detailed blog post analyzing {ticker_name} based on:
            
            Market Data:
            - Current Price: ${ticker_price:.2f}
            - Price Change: {ticker_change:.2%}
            
            Sentiment Analysis:
            - Overall Sentiment: {sentiment_value:.2f}
            - Confidence Score: {confidence_value:.2f}
            
            Recent News:
            {article_titles}
            
            Include technical analysis, market sentiment discussion, and future outlook.
            Format in Markdown.
            """

            response = self.client.chat.completions.create(
                model="llama-3.2-3b-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst and writer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.0,
            )

            return {
                "content": response.choices[0].message.content,
                "title": f"Analysis: {ticker_name}",
                "timestamp": datetime.now().isoformat(),
                "sentiment": data["sentiment_data"],
                "image_prompt": self.generate_refined_prompt_for_blog_post(
                    response.choices[0].message.content
                ),
            }

        except Exception as e:
            print(f"Error details: {str(e)}")
            self.logger.log_operation("content_generation_failed", error=str(e))
            raise

    async def process_ticker(self, category: str, ticker: str):
        try:
            logging.info(f"Processing ticker: {ticker}")

            # Fetch data specific to the ticker
            logging.debug(f"Fetching data for ticker: {ticker}")
            data = await self.services["data_pipeline"].process_data(ticker)

            # Analyze sentiment for related news articles
            logging.debug(f"Analyzing sentiment for ticker: {ticker}")
            sentiment = await self.services["sentiment"].analyze(
                [article["content"] for article in data["articles"]]
            )

            # Generate blog content
            logging.debug(f"Generating content for ticker: {ticker}")
            content = await self.generate_content(data, sentiment)

            # Add a delay before each API request to avoid hitting rate limits
            await asyncio.sleep(5)  # Adjust the delay as needed

            # Generate an image prompt and create an image
            logging.debug(f"Generating image prompt for ticker: {ticker}")
            image_prompt = self.generate_refined_prompt_for_blog_post(content)

            logging.debug(f"Generating image for ticker: {ticker}")
            
            # Add a delay before each API request to avoid hitting rate limits
            await asyncio.sleep(3)  # Adjust the delay as needed

            # Uses custom values
            image_result = await self.services["image"].generate_image(
                prompt=image_prompt,
                output_dir=f"output/{category}/images",
            )

            # More explicit path assignment with logging
            if image_result[1]:
                content["image_path"] = image_result[1]
                logging.debug(f"Image generated successfully at: {image_result[1]}")
            else:
                logging.debug(f"No image generated for ticker: {ticker}")

            # Save the blog post
            logging.debug(f"Saving blog post for ticker: {ticker}")
            await self.save_blog_post(category, ticker, content)
        except Exception as e:
            logging.error(f"Ticker processing failed for {ticker}: {e}")

    async def run(self):
        portfolio = self.services["portfolio"].read_portfolio()
        async with aiohttp.ClientSession() as session:
            tasks = []
            for category, tickers in portfolio.items():
                for ticker in tickers:
                    tasks.append(self.process_ticker(category, ticker))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Task failed with exception: {result}")


if __name__ == "__main__":
    generator = CompleteBlogGenerator()
    async def main():
        await generator.setup_async_components()
        await generator.run()
        await generator.session.close()
    
    asyncio.run(main())
