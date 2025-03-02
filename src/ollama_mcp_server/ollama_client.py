# ollama_client.py
import json
import logging
import asyncio
import aiohttp
import requests
import functools
from typing import Dict, Any, Optional, Union, Callable
from .config import settings

logger = logging.getLogger(__name__)

# グローバルセッションプール
_session_pool: Optional[aiohttp.ClientSession] = None

def lru_cache_with_args_serialization(maxsize: int = 128):
    """
    引数をシリアライズしてLRUキャッシュを適用するデコレータ
    """
    def decorator(func: Callable):
        cache = {}
        queue = []
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # プロンプトと追加パラメータを含むキーを作成
            key_parts = list(args)
            if kwargs:
                key_parts.append(json.dumps(kwargs, sort_keys=True))
            
            # キーをハッシュ化
            key = hash(str(key_parts))
            
            if key in cache:
                logger.debug(f"Cache hit for key: {key}")
                return cache[key]
            
            result = await func(*args, **kwargs)
            
            # キャッシュが上限に達した場合、最も古いエントリを削除
            if len(cache) >= maxsize:
                oldest_key = queue.pop(0)
                del cache[oldest_key]
            
            # 新しい結果をキャッシュに追加
            cache[key] = result
            queue.append(key)
            
            return result
        
        return wrapper
    
    return decorator

async def get_session() -> aiohttp.ClientSession:
    """
    非同期HTTPセッションを取得（必要に応じて作成）
    
    Returns:
        共有セッションインスタンス
    """
    global _session_pool
    
    if _session_pool is None or _session_pool.closed:
        # 接続プールの設定
        conn = aiohttp.TCPConnector(
            limit=settings.ollama.max_connections,
            limit_per_host=settings.ollama.max_connections_per_host
        )
        
        # セッションの作成
        timeout = aiohttp.ClientTimeout(total=settings.ollama.request_timeout)
        _session_pool = aiohttp.ClientSession(
            connector=conn,
            timeout=timeout,
            raise_for_status=True
        )
        
    return _session_pool

class OllamaClient:
    """OllamaのAPIクライアント."""
    
    def __init__(
        self, 
        model: str = settings.ollama.default_model,
        api_url: str = f"{settings.ollama.host}/api",
        temperature: float = settings.ollama.temperature
    ):
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        
    @lru_cache_with_args_serialization(maxsize=settings.ollama.cache_size)
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """テキストを非同期で生成."""
        url = f"{self.api_url}/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            **kwargs
        }
        
        session = await get_session()
        try:
            async with session.post(url, json=data) as response:
                result = await response.json()
                return result.get("response", "")
        except aiohttp.ClientError as e:
            logger.error(f"API通信エラー: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}")
            return ""
                
    @lru_cache_with_args_serialization(maxsize=settings.ollama.cache_size)
    async def generate_json(self, prompt: str, schema: Optional[Dict] = None, **kwargs) -> Dict:
        """JSON形式のデータを非同期で生成."""
        json_prompt = prompt
        if schema:
            json_prompt += f"\n\nOutput JSON SCHEMA: {json.dumps(schema)}"
            
        data = {
            "model": self.model,
            "prompt": json_prompt,
            "temperature": self.temperature,
            "stream": False,
            "format": "json",
            **kwargs
        }
        
        session = await get_session()
        try:
            async with session.post(f"{self.api_url}/generate", json=data) as response:
                result = await response.json()
                response_text = result.get("response", "{}")
                
                try:
                    if isinstance(response_text, str):
                        return json.loads(response_text)
                    return response_text
                except json.JSONDecodeError:
                    logger.error(f"JSON Parse Error: {response_text}")
                    return {}
        except aiohttp.ClientError as e:
            logger.error(f"API通信エラー: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}")
            return {}
                    
    def generate_text_sync(self, prompt: str, **kwargs) -> str:
        """テキストを同期的に生成."""
        url = f"{self.api_url}/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(url, json=data, timeout=settings.ollama.request_timeout)
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code}")
                return ""
                
            result = response.json()
            return result.get("response", "")
        except requests.RequestException as e:
            logger.error(f"API通信エラー: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}")
            return ""
        
    def generate_json_sync(self, prompt: str, schema: Optional[Dict] = None, **kwargs) -> Dict:
        """JSON形式のデータを同期的に生成."""
        json_prompt = prompt
        if schema:
            json_prompt += f"\n\nOutput JSON SCHEMA: {json.dumps(schema)}"
            
        data = {
            "model": self.model,
            "prompt": json_prompt,
            "temperature": self.temperature,
            "stream": False,
            "format": "json",
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/generate", 
                json=data, 
                timeout=settings.ollama.request_timeout
            )
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code}")
                return {}
                
            result = response.json()
            response_text = result.get("response", "{}")
            
            try:
                if isinstance(response_text, str):
                    return json.loads(response_text)
                return response_text
            except json.JSONDecodeError:
                logger.error(f"JSON Parse Error: {response_text}")
                return {}
        except requests.RequestException as e:
            logger.error(f"API通信エラー: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}")
            return {}

async def cleanup():
    """
    リソースのクリーンアップを行う
    """
    global _session_pool
    if _session_pool and not _session_pool.closed:
        await _session_pool.close()
        _session_pool = None
        logger.debug("HTTPセッションプールをクリーンアップしました") 