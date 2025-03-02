# ollama_client.py
import json
import logging
import aiohttp
import requests
from typing import Dict, Any, Optional, Union
from .config import settings

logger = logging.getLogger(__name__)

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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"API Error: {response.status}")
                    return ""
                    
                result = await response.json()
                return result.get("response", "")
                
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}/generate", json=data) as response:
                if response.status != 200:
                    logger.error(f"API Error: {response.status}")
                    return {}
                    
                result = await response.json()
                response_text = result.get("response", "{}")
                
                try:
                    if isinstance(response_text, str):
                        return json.loads(response_text)
                    return response_text
                except json.JSONDecodeError:
                    logger.error(f"JSON Parse Error: {response_text}")
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
        
        response = requests.post(url, json=data)
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code}")
            return ""
            
        result = response.json()
        return result.get("response", "")
        
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
        
        response = requests.post(f"{self.api_url}/generate", json=data)
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