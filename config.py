import os
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field
from pathlib import Path
from functools import lru_cache

class LLMConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float = 0.1
    api_key_env_var: str

    @property
    def api_key(self) -> str:
        """Метод-хелпер для получения ключа из переменных окружения"""
        key = os.getenv(self.api_key_env_var)
        if not key:
            raise ValueError(f"Environment variable {self.api_key_env_var} is not set.")
        return key

class VectorStoreConfig(BaseModel):
    collection_name: str
    path: str
    top_k: int = 5

class EmbeddingConfig(BaseModel):
    model_name: str

class DomainConfig(BaseModel):
    domain_path: str

class Settings(BaseModel):
    llm: LLMConfig
    vector_store: VectorStoreConfig
    embedding: EmbeddingConfig
    domain: DomainConfig


def load_config(config_path: str = "config.toml") -> Settings:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.absolute()}")

    with open(path, "rb") as f:
        config_data = tomllib.load(f)

    return Settings(**config_data)

@lru_cache()
def get_settings() -> Settings:
    return load_config()

settings: Settings = get_settings()
