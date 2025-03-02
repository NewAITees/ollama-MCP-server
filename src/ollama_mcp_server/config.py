# config.py
import os
import json
import logging
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logger = logging.getLogger("ollama-mcp-server")

class OllamaSettings(BaseSettings):
    """Ollamaの設定."""
    host: str = "http://localhost:11434"
    default_model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # パフォーマンス関連設定
    cache_size: int = 100
    max_connections: int = 10
    max_connections_per_host: int = 10
    request_timeout: int = 60  # 秒
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "OLLAMA_"
    }

class AppSettings(BaseSettings):
    """アプリケーション全体の設定."""
    log_level: str = "INFO"
    
    # Ollamaの設定
    ollama: OllamaSettings = OllamaSettings()
    
    # MCP config file path
    mcp_config_path: str = "./mcp_config.json"
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "APP_"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 環境変数からMCP設定ファイルパスを上書き
        env_path = os.environ.get("MCP_CONFIG_PATH")
        if env_path:
            self.mcp_config_path = env_path
        self._load_mcp_config()
    
    def _load_mcp_config(self):
        """MCP設定ファイルを読み込む"""
        if os.path.exists(self.mcp_config_path):
            try:
                with open(self.mcp_config_path, 'r') as f:
                    config = json.load(f)
                
                # mcpServersセクションが存在するか確認
                if "mcpServers" in config and "ollama-MCP-server" in config["mcpServers"]:
                    server_config = config["mcpServers"]["ollama-MCP-server"]
                    
                    # env設定があればパースする
                    if "env" in server_config:
                        for env_setting in server_config["env"]:
                            if isinstance(env_setting, dict):
                                # {"model": "deepseek:r14B"} のような形式
                                for key, value in env_setting.items():
                                    if key == "model" and value:
                                        logger.info(f"設定ファイルからモデルを読み込みました: {value}")
                                        self.ollama.default_model = value
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"MCP設定ファイル読み込みエラー: {e}")

settings = AppSettings() 