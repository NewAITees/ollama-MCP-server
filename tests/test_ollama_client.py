import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, MagicMock
import json
import asyncio
from ollama_mcp_server.ollama_client import OllamaClient

class TestOllamaClientSync(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()
        
    @patch('ollama_mcp_server.ollama_client.requests.post')
    def test_generate_text_sync(self, mock_post):
        # テスト用のレスポンスを設定
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "生成されたテキスト"}
        mock_post.return_value = mock_response
        
        result = self.client.generate_text_sync("テスト用プロンプト")
        self.assertEqual(result, "生成されたテキスト")
        
    @patch('ollama_mcp_server.ollama_client.requests.post')
    def test_generate_json_sync(self, mock_post):
        # テスト用のレスポンスを設定
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": '{"key": "value"}'}
        mock_post.return_value = mock_response
        
        result = self.client.generate_json_sync("テスト用プロンプト")
        self.assertEqual(result, {"key": "value"})

class TestOllamaClientAsync(IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = OllamaClient()
        
    @patch('ollama_mcp_server.ollama_client.aiohttp.ClientSession.post')
    async def test_generate_text(self, mock_post):
        # テスト用のレスポンスを設定
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": "生成されたテキスト"})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        result = await self.client.generate_text("テスト用プロンプト")
        self.assertEqual(result, "生成されたテキスト")
        
    @patch('ollama_mcp_server.ollama_client.aiohttp.ClientSession.post')
    async def test_generate_json(self, mock_post):
        # テスト用のレスポンスを設定
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": '{"key": "value"}'})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        result = await self.client.generate_json("テスト用プロンプト")
        self.assertEqual(result, {"key": "value"})

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

if __name__ == '__main__':
    unittest.main() 