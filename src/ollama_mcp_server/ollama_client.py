# ollama_client.py
import json
import logging
import asyncio
import aiohttp
import requests
import functools
from typing import Dict, Any, Optional, Union, Callable, List
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
            raise_for_status=False  # エラーステータスコードでも例外を発生させない
        )
        
    return _session_pool

class OllamaClient:
    """OllamaのAPIクライアント."""
    
    def __init__(
        self, 
        model: Optional[str] = None,
        api_base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Ollamaクライアントのコンストラクタです。
        
        Args:
            model: 使用するモデル名。未指定の場合は設定からデフォルト値を使用。
            api_base_url: Ollama APIのベースURL。未指定の場合は設定からデフォルト値を使用。
            temperature: 生成時の温度パラメータ。未指定の場合は設定からデフォルト値を使用。
            max_tokens: 生成する最大トークン数。未指定の場合は設定からデフォルト値を使用。
        """
        self.model = model or settings.ollama.default_model
        self.api_base_url = api_base_url or settings.ollama.host
        self.api_url = f"{self.api_base_url}/api"
        self.temperature = temperature or settings.ollama.temperature
        self.max_tokens = max_tokens or settings.ollama.max_tokens
        logger.debug(f"OllamaClient初期化: model={self.model}, url={self.api_base_url}")
    
    async def get_available_models(self) -> List[str]:
        """
        利用可能なモデル一覧を取得します。
        
        Returns:
            利用可能なモデル名のリスト
        """
        try:
            session = await get_session()
            endpoint = f"{self.api_url}/tags"
            
            logger.debug(f"モデル一覧を取得: {endpoint}")
            async with session.get(endpoint) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"モデル一覧取得エラー: {response.status}, message='{error_text}'")
                    return []
                
                result = await response.json()
                models = [model["name"] for model in result.get("models", [])]
                logger.debug(f"利用可能なモデル: {models}")
                return models
        except Exception as e:
            logger.error(f"モデル一覧取得エラー: {str(e)}")
            return []
    
    async def check_model_exists(self, model_name: str) -> bool:
        """
        指定されたモデルが存在するかチェックします。
        
        Args:
            model_name: チェックするモデル名
            
        Returns:
            モデルが存在する場合はTrue、存在しない場合はFalse
        """
        available_models = await self.get_available_models()
        exists = model_name in available_models
        logger.debug(f"モデル '{model_name}' 存在チェック: {exists}")
        return exists
        
    @lru_cache_with_args_serialization(maxsize=settings.ollama.cache_size)
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        テキストを非同期で生成します。
        
        Args:
            prompt: 生成のためのプロンプト
            **kwargs: Ollama APIに渡す追加パラメータ
            
        Returns:
            生成されたテキスト。エラー時は空文字列。
        """
        url = f"{self.api_url}/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            **kwargs
        }
        
        session = await get_session()
        try:
            logger.debug(f"生成リクエスト送信: {url} (model={self.model})")
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API通信エラー: {response.status}, message='{error_text}', url='{url}'")
                    
                    # モデルが見つからない場合の詳細なエラーメッセージ
                    if response.status == 404:
                        models = await self.get_available_models()
                        model_list = ", ".join(models) if models else "利用可能なモデルがありません"
                        logger.error(f"モデル '{self.model}' が見つかりません。利用可能なモデル: {model_list}")
                    
                    return ""
                
                result = await response.json()
                text_response = result.get("response", "")
                logger.debug(f"生成成功: {len(text_response)} 文字のレスポンス")
                return text_response
        except aiohttp.ClientError as e:
            logger.error(f"API通信エラー: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}")
            return ""
                
    @lru_cache_with_args_serialization(maxsize=settings.ollama.cache_size)
    async def generate_json(self, prompt: str, schema: Optional[Dict] = None, **kwargs) -> Dict:
        """
        JSON形式のデータを非同期で生成します。
        
        Args:
            prompt: 生成のためのプロンプト
            schema: 生成するJSONのスキーマ定義
            **kwargs: Ollama APIに渡す追加パラメータ
            
        Returns:
            生成されたJSONデータ。エラー時は空の辞書。
        """
        json_prompt = prompt
        if schema:
            json_prompt += f"\n\nOutput JSON SCHEMA: {json.dumps(schema)}"
            
        data = {
            "model": self.model,
            "prompt": json_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "format": "json",
            **kwargs
        }
        
        session = await get_session()
        try:
            endpoint = f"{self.api_url}/generate"
            logger.debug(f"JSON生成リクエスト送信: {endpoint} (model={self.model})")
            async with session.post(endpoint, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API通信エラー: {response.status}, message='{error_text}', url='{endpoint}'")
                    
                    # モデルが見つからない場合の詳細なエラーメッセージ
                    if response.status == 404:
                        models = await self.get_available_models()
                        model_list = ", ".join(models) if models else "利用可能なモデルがありません"
                        logger.error(f"モデル '{self.model}' が見つかりません。利用可能なモデル: {model_list}")
                    
                    return {}
                
                result = await response.json()
                response_text = result.get("response", "{}")
                
                try:
                    if isinstance(response_text, str):
                        json_data = json.loads(response_text)
                        logger.debug(f"JSON生成成功: {len(json.dumps(json_data))} 文字のレスポンス")
                        return json_data
                    logger.debug("JSON生成成功: オブジェクト直接返却")
                    return response_text
                except json.JSONDecodeError:
                    logger.error(f"JSON解析エラー: {response_text}")
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
            "max_tokens": self.max_tokens,
            "stream": False,
            **kwargs
        }
        
        try:
            logger.debug(f"同期生成リクエスト送信: {url} (model={self.model})")
            response = requests.post(url, json=data, timeout=settings.ollama.request_timeout)
            if response.status_code != 200:
                logger.error(f"API通信エラー: {response.status_code}, message='{response.text}', url='{url}'")
                return ""
                
            result = response.json()
            text_response = result.get("response", "")
            logger.debug(f"同期生成成功: {len(text_response)} 文字のレスポンス")
            return text_response
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
            "max_tokens": self.max_tokens,
            "stream": False,
            "format": "json",
            **kwargs
        }
        
        try:
            endpoint = f"{self.api_url}/generate"
            logger.debug(f"JSON同期生成リクエスト送信: {endpoint} (model={self.model})")
            response = requests.post(
                endpoint, 
                json=data, 
                timeout=settings.ollama.request_timeout
            )
            if response.status_code != 200:
                logger.error(f"API通信エラー: {response.status_code}, message='{response.text}', url='{endpoint}'")
                return {}
                
            result = response.json()
            response_text = result.get("response", "{}")
            
            try:
                if isinstance(response_text, str):
                    json_data = json.loads(response_text)
                    logger.debug(f"JSON同期生成成功: {len(json.dumps(json_data))} 文字のレスポンス")
                    return json_data
                logger.debug("JSON同期生成成功: オブジェクト直接返却")
                return response_text
            except json.JSONDecodeError:
                logger.error(f"JSON解析エラー: {response_text}")
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