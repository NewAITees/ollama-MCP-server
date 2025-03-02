"""
テスト用ユーティリティモジュール
"""
import json
from typing import Dict, List, Any, Optional, Union, Sequence
import asyncio
from mcp.server import Server
import mcp.types as types
from pydantic import AnyUrl

from ollama_mcp_server import server as server_module
from ollama_mcp_server.server import handle_call_tool

class MockMCPClient:
    """
    テスト用のモックMCPクライアント
    """
    
    def __init__(self, server: Server):
        """
        初期化
        
        Args:
            server: 対象のMCPサーバーインスタンス
        """
        self.server = server
        
    async def list_resources(self) -> List[types.Resource]:
        """
        リソース一覧を取得
        
        Returns:
            リソースのリスト
        """
        return await self.server.list_resources()
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        リソースを読み込む
        
        Args:
            uri: リソースURI
        
        Returns:
            リソースの内容
        """
        response = await self.server.read_resource(AnyUrl(uri))
        return json.loads(response)
    
    async def list_prompts(self) -> List[types.Prompt]:
        """
        プロンプト一覧を取得
        
        Returns:
            プロンプトのリスト
        """
        return await self.server.list_prompts()
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, str]] = None) -> types.GetPromptResult:
        """
        プロンプトを取得
        
        Args:
            name: プロンプト名
            arguments: 引数
            
        Returns:
            プロンプト取得結果
        """
        return await self.server.get_prompt(name, arguments)
    
    async def list_tools(self) -> List[types.Tool]:
        """
        ツール一覧を取得
        
        Returns:
            ツールのリスト
        """
        return await self.server.list_tools()
    
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ツールを呼び出す
        
        Args:
            name: ツール名
            arguments: 引数
            
        Returns:
            ツール呼び出し結果
        """
        # サーバーの実装では @server.call_tool() デコレータが使われているため
        # 直接 handle_call_tool 関数を呼び出す
        try:
            # テスト用に直接ハンドラー関数を呼び出す
            # 実際のサーバーコードでは、request_contextが設定されているが
            # テスト環境では設定されていないため、エラーが発生する可能性がある
            result = await handle_call_tool(name, arguments)
            if result and len(result) > 0:
                # TextContentから結果を抽出
                content = result[0]
                if hasattr(content, 'text'):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"raw_text": content.text}
            return {}
        except Exception as e:
            # エラーをキャッチして、エラー情報を返す
            error_message = str(e)
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            else:
                status_code = 500
                
            if hasattr(e, 'details'):
                details = e.details
            else:
                details = {}
                
            return {
                "error": {
                    "message": error_message,
                    "status_code": status_code,
                    "details": details
                }
            } 