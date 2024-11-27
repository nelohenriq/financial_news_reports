from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    HF_API_TOKEN: str
    REDIS_URL: str
    DEBUG_LEVEL: str = "INFO"
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"
