# config.py
import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class OllamaSettings(BaseSettings):
    """Ollamaの設定."""
    host: str = "http://localhost:11434"
    default_model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "OLLAMA_"
    }

class AppSettings(BaseSettings):
    """アプリケーション全体の設定."""
    log_level: str = "INFO"
    
    # Ollamaの設定
    ollama: OllamaSettings = OllamaSettings()
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "APP_"
    }

settings = AppSettings() 