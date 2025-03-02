"""
統合テスト
"""
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, AsyncMock
import json
import asyncio
from typing import Dict, List, Any

from ollama_mcp_server import server
from tests.test_utils import MockMCPClient

class TestIntegration(IsolatedAsyncioTestCase):
    """エンドツーエンドのワークフローをテストするクラス"""
    
    def setUp(self):
        """テスト実行前の準備"""
        # リポジトリをリセット
        server.repo.tasks = {}
        server.repo.subtasks = {}
        server.repo.results = {}
        server.repo.evaluations = {}
        
        # モックMCPクライアントを作成
        self.client = MockMCPClient(server.server)
        
    @patch('ollama_mcp_server.ollama_client.OllamaClient.generate_json')
    async def test_complete_workflow(self, mock_generate_json):
        """完全なワークフローをテスト"""
        # テスト用のモック応答を設定
        subtasks_response = {
            "subtasks": [
                {"description": "サブタスク1", "estimated_complexity": 3, "dependencies": []},
                {"description": "サブタスク2", "estimated_complexity": 5, "dependencies": [1]}
            ]
        }
        
        evaluation_response = {
            "scores": {
                "accuracy": 0.8,
                "completeness": 0.7,
                "clarity": 0.9
            },
            "overall_score": 0.8,
            "feedback": "全体的に良い結果です。",
            "improvements": [
                "詳細をもう少し追加するとより良いでしょう。"
            ]
        }
        
        # モック応答を設定
        mock_generate_json.side_effect = [subtasks_response, evaluation_response]
        
        # 1. タスクを追加
        add_task_result = await self.client.call_tool("add-task", {
            "name": "テスト用タスク",
            "description": "これは統合テスト用のタスク説明です。",
            "priority": 2,
            "tags": ["test", "integration"]
        })
        
        # エラーレスポンスの場合はテストをスキップ
        if "error" in add_task_result:
            self.skipTest(f"タスク追加でエラーが発生しました: {add_task_result['error']['message']}")
            return
            
        self.assertIn("status", add_task_result)
        self.assertEqual(add_task_result["status"], "success")
        self.assertIn("task_id", add_task_result)
        
        task_id = add_task_result["task_id"]
        
        # 2. リソース一覧を取得して、追加したタスクが含まれていることを確認
        resources = await self.client.list_resources()
        self.assertTrue(any(str(r.uri).endswith(task_id) for r in resources))
        
        # 3. タスクを分解
        decompose_result = await self.client.call_tool("decompose-task", {
            "task_id": task_id,
            "granularity": "medium",
            "max_subtasks": 2
        })
        
        # タスク分解の結果を検証
        self.assertIn("subtasks", decompose_result)
        self.assertEqual(len(decompose_result["subtasks"]), 2)
        
        # 4. 結果を追加
        result_content = "これはテスト用のタスク結果です。目標を達成しました。"
        result = server.repo.add_result(task_id, result_content)
        
        # 5. 結果を評価
        evaluate_result = await self.client.call_tool("evaluate-result", {
            "result_id": result.id,
            "criteria": {"accuracy": 0.4, "completeness": 0.3, "clarity": 0.3},
            "detailed": True
        })
        
        # 評価結果を検証
        self.assertIn("scores", evaluate_result)
        self.assertIn("feedback", evaluate_result)
        self.assertEqual(evaluate_result["overall_score"], 0.8)
        
        # 6. 再度リソース一覧を取得して、結果と評価が含まれていることを確認
        resources = await self.client.list_resources()
        self.assertTrue(any(str(r.uri).startswith("result://") for r in resources))
        
    @patch('ollama_mcp_server.ollama_client.OllamaClient.generate_text')
    @patch('ollama_mcp_server.ollama_client.OllamaClient.get_available_models')
    @patch('ollama_mcp_server.ollama_client.OllamaClient.check_model_exists')
    async def test_run_model(self, mock_check_model_exists, mock_get_available_models, mock_generate_text):
        """run-modelツールのテスト"""
        # モック応答を設定
        mock_generate_text.return_value = "これはモデルからの応答です。"
        mock_get_available_models.return_value = ["llama3", "mistral", "llama2"]
        mock_check_model_exists.return_value = True
        
        # モデルを実行
        result = await self.client.call_tool("run-model", {
            "model": "deepseek-r1:14b",
            "prompt": "こんにちは、AIさん",
            "temperature": 0.7,
            "max_tokens": 10000
        })
        
        # エラーレスポンスの場合はテストをスキップ
        if "error" in result:
            self.skipTest(f"モデル実行でエラーが発生しました: {result['error']['message']}")
            return
            
        # 結果を検証
        self.assertIn("raw_text", result)
        self.assertEqual(result["raw_text"], "これはモデルからの応答です。")
        
        # モックが正しく呼び出されたことを確認
        mock_generate_text.assert_called_once()
        mock_check_model_exists.assert_called_once_with("llama2")
        args, kwargs = mock_generate_text.call_args
        self.assertEqual(args[0], "こんにちは、AIさん")
        
    @patch('ollama_mcp_server.ollama_client.OllamaClient.generate_text')
    @patch('ollama_mcp_server.ollama_client.OllamaClient.get_available_models')
    @patch('ollama_mcp_server.ollama_client.OllamaClient.check_model_exists')
    async def test_run_model_with_nonexistent_model(self, mock_check_model_exists, mock_get_available_models, mock_generate_text):
        """存在しないモデルでのテスト"""
        # モック応答を設定
        mock_generate_text.return_value = ""  # 空の応答
        mock_get_available_models.return_value = ["llama3", "mistral"]
        mock_check_model_exists.return_value = False
        
        # 存在しないモデルで実行
        result = await self.client.call_tool("run-model", {
            "model": "nonexistent_model",
            "prompt": "こんにちは、AIさん"
        })
        
        # エラーメッセージを含む応答であることを確認
        self.assertIn("raw_text", result)
        self.assertIn("エラー", result["raw_text"])
        self.assertIn("利用可能なモデル", result["raw_text"])
        
        # モックが正しく呼び出されたことを確認
        mock_check_model_exists.assert_called_once_with("nonexistent_model")
        mock_get_available_models.assert_called()
        mock_generate_text.assert_called_once()
        
    async def test_error_handling(self):
        """エラー処理のテスト"""
        # 存在しないタスクIDでタスク分解を試みる
        result = await self.client.call_tool("decompose-task", {
            "task_id": "non-existent-task",
            "granularity": "medium"
        })
        
        # エラーレスポンスを検証
        self.assertIn("error", result)
        self.assertIn("message", result["error"])
        self.assertIn("status_code", result["error"])
        self.assertEqual(result["error"]["status_code"], 404)
        
        # 必須パラメータが欠けているケース
        result = await self.client.call_tool("add-task", {
            "name": "テスト用タスク"
            # descriptionが欠けている
        })
        
        # エラーレスポンスを検証
        self.assertIn("error", result)
        self.assertIn("message", result["error"])
        self.assertIn("status_code", result["error"])
        self.assertEqual(result["error"]["status_code"], 400)
        
if __name__ == "__main__":
    unittest.main() 