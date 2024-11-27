import json
from redis import Redis
from forex_python.converter import CurrencyRates
from pycoingecko import CoinGeckoAPI
import yfinance as yf


class DataFetcher:
    def __init__(self, config, redis_client):
        self.config = config
        self.redis = redis_client
        self.currency_rates = CurrencyRates()
        self.coingecko = CoinGeckoAPI()

    async def get_cached_data(self, key, fetch_func):
        if cached := self.redis.get(key):
            return json.loads(cached)
        value = await fetch_func()
        self.redis.set(key, json.dumps(value), ex=self.config.CACHE_TTL)
        return value

    async def fetch_ticker_data(self, ticker):
        async def fetch():
            ticker_obj = yf.Ticker(ticker)
            history = ticker_obj.history(period="1mo")
            return {
                "info": ticker_obj.info,
                "price": float(history["Close"].iloc[-1]),
                "volume": float(history["Volume"].iloc[-1]),
                "change": float(
                    (history["Close"].iloc[-1] - history["Close"].iloc[0])
                    / history["Close"].iloc[0]
                ),
            }

        return await self.get_cached_data(f"ticker_{ticker}", fetch)

    async def fetch_crypto_data(self, ticker):
        # use yfinance for crypto data
        async def fetch():
            ticker_obj = yf.Ticker(ticker)
            history = ticker_obj.history(period="1mo")
            return {
                "info": ticker_obj.info,
                "price": float(history["Close"].iloc[-1]),
                "volume": float(history["Volume"].iloc[-1]),
                "change": float(
                    (history["Close"].iloc[-1] - history["Close"].iloc[0])
                    / history["Close"].iloc[0]
                ),
                "market_cap": float(ticker_obj.info["marketCap"]),
            }

        return await self.get_cached_data(f"crypto_{ticker}", fetch)

    async def fetch_forex_data(self, base_currency, target_currency):
        async def fetch():
            rate = self.currency_rates.get_rate(base_currency, target_currency)
            return {"rate": rate}

        return await self.get_cached_data(
            f"forex_{base_currency}_{target_currency}", fetch
        )
