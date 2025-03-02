import unittest
import json
from datetime import datetime
from ollama_mcp_server.models import Task, Subtask, Result, Evaluation

# datetime型をJSONにシリアライズするためのヘルパー関数
def datetime_to_iso(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not serializable")

class TestModels(unittest.TestCase):
    def test_task_model(self):
        # Taskモデルのインスタンス化とシリアライズのテスト
        task = Task(
            id="task-123",
            name="テストタスク",
            description="テスト用のタスク説明",
            priority=4,
            tags=["重要", "緊急"]
        )
        
        # model_dumpメソッドのテスト（Pydantic 2.0互換）
        task_dict = task.model_dump()
        self.assertEqual(task_dict["id"], "task-123")
        self.assertEqual(task_dict["name"], "テストタスク")
        self.assertEqual(task_dict["priority"], 4)
        self.assertEqual(task_dict["tags"], ["重要", "緊急"])
        self.assertIsNotNone(task_dict["created_at"])
        
        # 旧型のdictメソッドのテスト（後方互換性）
        task_dict_legacy = task.dict()
        self.assertEqual(task_dict, task_dict_legacy)
        
        # JSONシリアライズのテスト（datetime型をカスタムエンコーダーで変換）
        task_json = json.dumps(task_dict, default=datetime_to_iso)
        self.assertIsInstance(task_json, str)
        
    def test_subtask_model(self):
        # Subtaskモデルのインスタンス化とシリアライズのテスト
        subtask = Subtask(
            id="subtask-456",
            task_id="task-123",
            description="サブタスクの説明",
            order=1,
            completed=True
        )
        
        # model_dumpメソッドのテスト
        subtask_dict = subtask.model_dump()
        self.assertEqual(subtask_dict["id"], "subtask-456")
        self.assertEqual(subtask_dict["task_id"], "task-123")
        self.assertEqual(subtask_dict["order"], 1)
        self.assertTrue(subtask_dict["completed"])
        
        # 旧型のdictメソッドのテスト
        subtask_dict_legacy = subtask.dict()
        self.assertEqual(subtask_dict, subtask_dict_legacy)
        
    def test_result_model(self):
        # Resultモデルのインスタンス化とシリアライズのテスト
        result = Result(
            id="result-789",
            task_id="task-123",
            content="タスク結果の内容"
        )
        
        # model_dumpメソッドのテスト
        result_dict = result.model_dump()
        self.assertEqual(result_dict["id"], "result-789")
        self.assertEqual(result_dict["task_id"], "task-123")
        self.assertEqual(result_dict["content"], "タスク結果の内容")
        self.assertIsNotNone(result_dict["created_at"])
        
        # 旧型のdictメソッドのテスト
        result_dict_legacy = result.dict()
        self.assertEqual(result_dict, result_dict_legacy)
        
    def test_evaluation_model(self):
        # Evaluationモデルのインスタンス化とシリアライズのテスト
        evaluation = Evaluation(
            id="eval-101",
            result_id="result-789",
            criteria={"accuracy": 0.6, "completeness": 0.4},
            scores={"accuracy": 90, "completeness": 85},
            feedback="良好な結果です"
        )
        
        # model_dumpメソッドのテスト
        eval_dict = evaluation.model_dump()
        self.assertEqual(eval_dict["id"], "eval-101")
        self.assertEqual(eval_dict["result_id"], "result-789")
        self.assertEqual(eval_dict["criteria"], {"accuracy": 0.6, "completeness": 0.4})
        self.assertEqual(eval_dict["scores"], {"accuracy": 90, "completeness": 85})
        self.assertEqual(eval_dict["feedback"], "良好な結果です")
        self.assertIsNotNone(eval_dict["created_at"])
        
        # 旧型のdictメソッドのテスト
        eval_dict_legacy = evaluation.dict()
        self.assertEqual(eval_dict, eval_dict_legacy)

if __name__ == '__main__':
    unittest.main() 