"""
Ollama MCP サーバー - Ollamaモデルを活用したMCP互換サーバー

このパッケージは、ローカルのOllama LLMインスタンスとMCP互換アプリケーションの間で
シームレスな統合を可能にし、高度なタスク分解、評価、ワークフロー管理を提供します。
"""

__version__ = "0.1.0"
__author__ = "Kai Kogure"

from . import server
from . import ollama_client
from . import repository
from . import prompts
from . import config
from . import models
import asyncio

def main():
    """メインエントリーポイント"""
    asyncio.run(server.main())

# パッケージレベルでのエクスポート
__all__ = [
    'main', 
    'server', 
    'ollama_client', 
    'repository', 
    'prompts', 
    'config',
    'models',
    '__version__'
]