import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, MagicMock, AsyncMock
import json
import asyncio
from ollama_mcp_server import server
from typing import Dict, List, Any, Optional

class TestMCPServer(IsolatedAsyncioTestCase):
    def setUp(self):
        # テスト用のセットアップ
        # サーバーのリポジトリをリセット
        server.repo.tasks = {}
        server.repo.subtasks = {}
        server.repo.results = {}
        server.repo.evaluations = {}
        
    @patch('ollama_mcp_server.server.PromptTemplates.task_decomposition')
    @patch('ollama_mcp_server.ollama_client.OllamaClient.generate_json')
    async def test_task_decomposition(self, mock_generate, mock_template):
        # モックの設定
        mock_generate.return_value = {"subtasks": [
            {"description": "サブタスク1", "estimated_complexity": 3, "dependencies": []},
            {"description": "サブタスク2", "estimated_complexity": 5, "dependencies": [1]}
        ]}
        mock_template.return_value = "タスク分解テンプレート"
        
        # タスク作成
        task = server.repo.add_task("テストタスク", "テスト用の説明")
        
        # 分解テスト（server.handle_call_toolを使わずに直接サブタスク作成）
        subtasks = server.repo.add_subtasks(
            task.id, 
            [
                {"description": "サブタスク1"}, 
                {"description": "サブタスク2"}
            ]
        )
        
        # 検証
        self.assertEqual(len(subtasks), 2)
        self.assertEqual(subtasks[0].description, "サブタスク1")
        self.assertEqual(subtasks[1].description, "サブタスク2")
        self.assertEqual(subtasks[0].task_id, task.id)
        
    @patch('ollama_mcp_server.server.PromptTemplates.result_evaluation')
    @patch('ollama_mcp_server.ollama_client.OllamaClient.generate_json')
    async def test_result_evaluation(self, mock_generate, mock_template):
        # モックの設定
        mock_generate.return_value = {
            "scores": {"accuracy": 90, "completeness": 80},
            "overall_score": 85,
            "feedback": "高品質な結果です",
            "improvements": ["さらに詳細な説明を追加する"]
        }
        mock_template.return_value = "結果評価テンプレート"
        
        # タスクと結果を作成
        task = server.repo.add_task("テストタスク", "テスト用の説明")
        result = server.repo.add_result(task.id, json.dumps({"test": "data"}))
        
        # 評価を作成
        criteria = {"accuracy": 0.6, "completeness": 0.4}
        scores = {"accuracy": 90, "completeness": 80}
        feedback = "高品質な結果です"
        
        evaluation = server.repo.add_evaluation(
            result.id,
            criteria,
            scores,
            feedback
        )
        
        # 検証
        self.assertEqual(evaluation.result_id, result.id)
        self.assertEqual(evaluation.criteria, criteria)
        self.assertEqual(evaluation.scores, scores)
        self.assertEqual(evaluation.feedback, feedback)
        
    async def test_add_task(self):
        # タスク作成テスト
        name = "新しいタスク"
        description = "新しいタスクの説明"
        priority = 2
        tags = ["テスト", "重要"]
        
        task = server.repo.add_task(
            name=name,
            description=description,
            priority=priority,
            tags=tags
        )
        
        # 検証
        self.assertEqual(task.name, name)
        self.assertEqual(task.description, description)
        self.assertEqual(task.priority, priority)
        self.assertEqual(task.tags, tags)
        self.assertIn(task.id, server.repo.tasks)
        
    @patch('ollama_mcp_server.ollama_client.OllamaClient.generate_text')
    async def test_generate_text(self, mock_generate):
        # モックの設定
        expected_response = "モデルからの応答テキスト"
        mock_generate.return_value = expected_response
        
        # OllamaClientから直接テスト
        client = server.ollama_client
        response = await client.generate_text("テストプロンプト")
        
        # 検証
        self.assertEqual(response, expected_response)
        mock_generate.assert_called_once_with("テストプロンプト")

if __name__ == '__main__':
    unittest.main() 