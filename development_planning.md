# development planning


ご回答ありがとうございます。それでは、提供いただいた要件とサンプルコードを基に、Ollamaを活用したMCPサーバーの実装計画を作成します。

# Ollama-MCP-Server 実装計画

## 1. テスト計画と実装

### 1.1 ユニットテスト
```python
# test_ollama_client.py
import unittest
from unittest.mock import patch, MagicMock
import json
import asyncio
from ollama_client import OllamaClient

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()
        
    @patch('ollama_client.requests.post')
    def test_generate_text_sync(self, mock_post):
        # テスト用のレスポンスを設定
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "生成されたテキスト"}
        mock_post.return_value = mock_response
        
        result = self.client.generate_text_sync("テスト用プロンプト")
        self.assertEqual(result, "生成されたテキスト")
        
    @patch('ollama_client.requests.post')
    def test_generate_json_sync(self, mock_post):
        # テスト用のレスポンスを設定
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"key": "value"}'}
        mock_post.return_value = mock_response
        
        result = self.client.generate_json_sync("テスト用プロンプト")
        self.assertEqual(result, {"key": "value"})

# test_mcp_server.py
import unittest
from unittest.mock import patch, AsyncMock
import json
from mcp_server import app

class TestMCPServer(unittest.TestCase):
    @patch('mcp_server.OllamaClient.generate_text')
    async def test_decompose_task(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "tasks": [
                {"id": 1, "description": "サブタスク1"},
                {"id": 2, "description": "サブタスク2"}
            ]
        })
        
        result = await app.handle_call_tool(
            "decompose-task", 
            {"task_id": "task://123", "granularity": "medium"}
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
```

### 1.2 統合テスト
```python
# test_integration.py
import unittest
import asyncio
import json
from mcp.server.test_utils import MockClient
from main import app

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.client = MockClient(app)
        
    async def test_full_workflow(self):
        # タスクの追加
        add_result = await self.client.call_tool("add-task", {
            "name": "テストタスク",
            "description": "複雑なタスクの例",
        })
        
        # タスクIDの取得
        resources = await self.client.list_resources()
        task_id = resources[0].uri
        
        # タスク分解
        decompose_result = await self.client.call_tool("decompose-task", {
            "task_id": task_id,
            "granularity": "high"
        })
        
        # 結果評価
        eval_result = await self.client.call_tool("evaluate-result", {
            "result_id": f"result://{task_id}",
            "criteria": {"accuracy": 0.4, "completeness": 0.3, "clarity": 0.3}
        })
        
        self.assertIsNotNone(decompose_result)
        self.assertIsNotNone(eval_result)
```

## 2. コアモジュール実装計画

### 2.1 設定管理モジュール
```python
# config.py
import os
from pydantic import BaseSettings

class OllamaSettings(BaseSettings):
    """Ollamaの設定."""
    host: str = "http://localhost:11434"
    default_model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    class Config:
        env_file = ".env"
        env_prefix = "OLLAMA_"

class AppSettings(BaseSettings):
    """アプリケーション全体の設定."""
    log_level: str = "INFO"
    
    # Ollamaの設定
    ollama: OllamaSettings = OllamaSettings()
    
    class Config:
        env_file = ".env"
        env_prefix = "APP_"

settings = AppSettings()
```

### 2.2 Ollamaクライアントの実装
```python
# ollama_client.py
import json
import logging
import aiohttp
import requests
from typing import Dict, Any, Optional, Union
from config import settings

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
```

### 2.3 プロンプトテンプレート管理
```python
# prompts.py
import os
import json

class PromptTemplates:
    """プロンプトテンプレートを管理するクラス."""
    
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    
    @classmethod
    def get_template(cls, name: str) -> str:
        """指定されたテンプレートを取得."""
        file_path = os.path.join(cls.TEMPLATE_DIR, f"{name}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(f"Template not found: {name}")
    
    @classmethod
    def task_decomposition(cls) -> str:
        """タスク分解のプロンプトテンプレートを取得."""
        return cls.get_template("task_decomposition")
    
    @classmethod
    def result_evaluation(cls) -> str:
        """結果評価のプロンプトテンプレートを取得."""
        return cls.get_template("result_evaluation")
```

## 3. MCPサーバー実装計画

### 3.1 データモデル
```python
# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Task(BaseModel):
    """タスクモデル."""
    id: str
    name: str
    description: str
    priority: int = 3
    deadline: Optional[datetime] = None
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    
class Subtask(BaseModel):
    """サブタスクモデル."""
    id: str
    task_id: str
    description: str
    order: int
    completed: bool = False
    
class Result(BaseModel):
    """結果モデル."""
    id: str
    task_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    
class Evaluation(BaseModel):
    """評価モデル."""
    id: str
    result_id: str
    criteria: Dict[str, float]
    scores: Dict[str, float]
    feedback: str
    created_at: datetime = Field(default_factory=datetime.now)
```

### 3.2 リポジトリ
```python
# repository.py
import uuid
from typing import Dict, List, Optional
from models import Task, Subtask, Result, Evaluation

class Repository:
    """データリポジトリ."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.subtasks: Dict[str, List[Subtask]] = {}
        self.results: Dict[str, Result] = {}
        self.evaluations: Dict[str, Evaluation] = {}
        
    def add_task(self, name: str, description: str, **kwargs) -> Task:
        """新しいタスクを追加."""
        task_id = f"task-{uuid.uuid4()}"
        task = Task(id=task_id, name=name, description=description, **kwargs)
        self.tasks[task_id] = task
        return task
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """タスクを取得."""
        return self.tasks.get(task_id)
        
    def add_subtasks(self, task_id: str, subtasks: List[Dict]) -> List[Subtask]:
        """サブタスクを追加."""
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")
            
        result = []
        for i, st in enumerate(subtasks):
            subtask_id = f"subtask-{uuid.uuid4()}"
            subtask = Subtask(
                id=subtask_id,
                task_id=task_id,
                description=st["description"],
                order=i
            )
            if task_id not in self.subtasks:
                self.subtasks[task_id] = []
            self.subtasks[task_id].append(subtask)
            result.append(subtask)
        return result
        
    def add_result(self, task_id: str, content: str) -> Result:
        """結果を追加."""
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")
            
        result_id = f"result-{uuid.uuid4()}"
        result = Result(id=result_id, task_id=task_id, content=content)
        self.results[result_id] = result
        return result
        
    def add_evaluation(
        self, result_id: str, criteria: Dict[str, float], 
        scores: Dict[str, float], feedback: str
    ) -> Evaluation:
        """評価を追加."""
        if result_id not in self.results:
            raise ValueError(f"Result not found: {result_id}")
            
        eval_id = f"eval-{uuid.uuid4()}"
        evaluation = Evaluation(
            id=eval_id,
            result_id=result_id,
            criteria=criteria,
            scores=scores,
            feedback=feedback
        )
        self.evaluations[eval_id] = evaluation
        return evaluation
```

### 3.3 MCPサーバー実装
```python
# server.py
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
from pydantic import AnyUrl
from typing import Dict, List, Any, Sequence, Optional
import json
import asyncio
import logging

from ollama_client import OllamaClient
from repository import Repository
from prompts import PromptTemplates
from config import settings

# ロガーの設定
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger("ollama-mcp-server")

# リポジトリの初期化
repo = Repository()

# Ollamaクライアントの初期化
ollama_client = OllamaClient()

# MCPサーバーの初期化
app = Server("ollama-MCP-server")

@app.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """利用可能なリソースのリストを返す."""
    resources = []
    
    # タスクリソース
    for task_id, task in repo.tasks.items():
        resources.append(
            types.Resource(
                uri=AnyUrl(f"task://{task_id}"),
                name=f"Task: {task.name}",
                description=task.description,
                mimeType="application/json",
            )
        )
    
    # 結果リソース
    for result_id, result in repo.results.items():
        task = repo.get_task(result.task_id)
        resources.append(
            types.Resource(
                uri=AnyUrl(f"result://{result_id}"),
                name=f"Result: {task.name if task else 'Unknown'}",
                description=f"Result for task {result.task_id}",
                mimeType="application/json",
            )
        )
    
    # Ollamaモデルリソース
    resources.append(
        types.Resource(
            uri=AnyUrl(f"model://{settings.ollama.default_model}"),
            name=f"Model: {settings.ollama.default_model}",
            description=f"Ollama model",
            mimeType="application/json",
        )
    )
    
    return resources

@app.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """指定されたリソースの内容を返す."""
    uri_str = str(uri)
    
    if uri_str.startswith("task://"):
        task_id = uri_str[7:]
        task = repo.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        return json.dumps(task.dict())
    
    elif uri_str.startswith("result://"):
        result_id = uri_str[9:]
        result = repo.results.get(result_id)
        if not result:
            raise ValueError(f"Result not found: {result_id}")
        return json.dumps(result.dict())
    
    elif uri_str.startswith("model://"):
        model_name = uri_str[8:]
        # 実際のモデル情報はOllamaから取得するべきだが、
        # 簡略化のために静的な情報を返す
        return json.dumps({
            "name": model_name,
            "description": f"Ollama model: {model_name}",
            "parameters": {
                "temperature": settings.ollama.temperature,
                "max_tokens": settings.ollama.max_tokens
            }
        })
    
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")

@app.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """利用可能なプロンプトのリストを返す."""
    return [
        types.Prompt(
            name="decompose-task",
            description="Breaks complex tasks into manageable subtasks",
            arguments=[
                types.PromptArgument(
                    name="granularity",
                    description="Granularity level (high/medium/low)",
                    required=True,
                )
            ],
        ),
        types.Prompt(
            name="evaluate-result",
            description="Analyzes task results against specified criteria",
            arguments=[
                types.PromptArgument(
                    name="criteria",
                    description="Evaluation criteria with weights",
                    required=True,
                )
            ],
        )
    ]

@app.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """プロンプトを取得する."""
    if name == "decompose-task":
        granularity = (arguments or {}).get("granularity", "medium")
        template = PromptTemplates.task_decomposition()
        prompt = template.format(granularity=granularity)
        
        return types.GetPromptResult(
            description=f"Task decomposition prompt with {granularity} granularity",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=prompt,
                    ),
                )
            ],
        )
    
    elif name == "evaluate-result":
        criteria = (arguments or {}).get("criteria", "{}")
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except json.JSONDecodeError:
                criteria = {}
        
        template = PromptTemplates.result_evaluation()
        prompt = template.format(criteria=json.dumps(criteria))
        
        return types.GetPromptResult(
            description="Result evaluation prompt",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=prompt,
                    ),
                )
            ],
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """利用可能なツールのリストを返す."""
    return [
        types.Tool(
            name="add-task",
            description="Add a new task",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description"],
            },
        ),
        types.Tool(
            name="decompose-task",
            description="Break down a complex task into manageable subtasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "granularity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "max_subtasks": {"type": "number"},
                },
                "required": ["task_id", "granularity"],
            },
        ),
        types.Tool(
            name="evaluate-result",
            description="Evaluate a result against specified criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "criteria": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "detailed": {"type": "boolean"},
                },
                "required": ["result_id", "criteria"],
            },
        ),
        types.Tool(
            name="run-model",
            description="Execute an Ollama model with the specified parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "prompt": {"type": "string"},
                    "temperature": {"type": "number"},
                    "max_tokens": {"type": "number"},
                },
                "required": ["prompt"],
            },
        ),
    ]

@app.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> Sequence[types.TextContent | types.EmbeddedResource]:
    """ツールを実行する."""
    if not arguments:
        raise ValueError("Missing arguments")
    
    if name == "add-task":
        task_name = arguments.get("name")
        description = arguments.get("description")
        
        if not task_name or not description:
            raise ValueError("Missing name or description")
        
        priority = arguments.get("priority", 3)
        tags = arguments.get("tags", [])
        
        task = repo.add_task(
            name=task_name,
            description=description,
            priority=priority,
            tags=tags
        )
        
        # クライアントにリソースが変更されたことを通知
        await app.request_context.session.send_resource_list_changed()
        
        return [
            types.TextContent(
                type="text",
                text=f"Added task '{task_name}' with ID: {task.id}"
            )
        ]
    
    elif name == "decompose-task":
        task_id = arguments.get("task_id")
        if not task_id:
            raise ValueError("Missing task_id")
            
        if task_id.startswith("task://"):
            task_id = task_id[7:]
            
        task = repo.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
            
        granularity = arguments.get("granularity", "medium")
        max_subtasks = arguments.get("max_subtasks", 5)
        
        # タスク分解プロンプトを取得
        template = PromptTemplates.task_decomposition()
        prompt = template.format(
            granularity=granularity,
            task_name=task.name,
            task_description=task.description,
            max_subtasks=max_subtasks
        )
        
        # Ollamaで分解を実行
        json_schema = {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "estimated_complexity": {"type": "number"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }
        
        result = await ollama_client.generate_json(prompt, json_schema)
        
        if not result or "subtasks" not in result:
            raise ValueError("Failed to decompose task")
            
        # サブタスクを保存
        subtasks = repo.add_subtasks(
            task_id, 
            [{"description": st["description"]} for st in result["subtasks"]]
        )
        
        # 結果を保存
        repo.add_result(task_id, json.dumps(result))
        
        # クライアントにリソースが変更されたことを通知
        await app.request_context.session.send_resource_list_changed()
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    elif name == "evaluate-result":
        result_id = arguments.get("result_id")
        if not result_id:
            raise ValueError("Missing result_id")
            
        if result_id.startswith("result://"):
            result_id = result_id[9:]
            
        result = repo.results.get(result_id)
        if not result:
            raise ValueError(f"Result not found: {result_id}")
            
        criteria = arguments.get("criteria", {})
        detailed = arguments.get("detailed", False)
        
        # タスク情報を取得
        task = repo.get_task(result.task_id)
        if not task:
            raise ValueError(f"Task not found for result: {result.task_id}")
        
        # 評価プロンプトを取得
        template = PromptTemplates.result_evaluation()
        prompt = template.format(
            task_name=task.name,
            task_description=task.description,
            result_content=result.content,
            criteria=json.dumps(criteria),
            detailed=str(detailed).lower()
        )
        
        # Ollamaで評価を実行
        json_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                "overall_score": {"type": "number"},
                "feedback": {"type": "string"},
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        eval_result = await ollama_client.generate_json(prompt, json_schema)
        
        if not eval_result or "scores" not in eval_result:
            raise ValueError("Failed to evaluate result")
            
        # 評価を保存
        repo.add_evaluation(
            result_id=result_id,
            criteria=criteria,
            scores=eval_result["scores"],
            feedback=eval_result["feedback"]
        )
        
        # クライアントにリソースが変更されたことを通知
        await app.request_context.session.send_resource_list_changed()
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(eval_result, indent=2)
            )
        ]
    
    elif name == "run-model":
        prompt = arguments.get("prompt")
        if not prompt:
            raise ValueError("Missing prompt")
            
        model = arguments.get("model", settings.ollama.default_model)
        temperature = arguments.get("temperature", settings.ollama.temperature)
        
        # 一時的にモデルとパラメータを変更するためのクライアントを作成
        temp_client = OllamaClient(
            model=model,
            temperature=temperature
        )
        
        response = await temp_client.generate_text(prompt)
        
        return [
            types.TextContent(
                type="text",
                text=response
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")
```

### 3.4 プロンプトテンプレートファイル

#### task_decomposition.txt
```
あなたは高度なタスク分解専門家です。
与えられたタスクを分析し、効率的に実行するために最適なサブタスクに分解してください。

# タスク詳細
- タスク名: {task_name}
- 説明: {task_description}
- 分解の粒度: {granularity}
- 最大サブタスク数: {max_subtasks}

# 粒度の基準
- high: 非常に詳細なサブタスク（初心者向け）
- medium: 標準的な詳細さのサブタスク（一般的なユーザー向け）
- low: 大まかなサブタスク（専門家向け）

# 出力形式
次のJSON形式で出力してください:
{{
  "subtasks": [
    {{
      "description": "サブタスクの説明",
      "estimated_complexity": 数値（1-10のスケール）,
      "dependencies": [依存するサブタスクの番号（1から始まる）のリスト]
    }}
  ]
}}

# 出力要件
1. 最大限厳密にタスクを分解すること
2. 各サブタスクは明確かつ実行可能であること
3. 依存関係は必要な場合のみ指定すること
4. サブタスクは論理的な順序で並べること
```

#### result_evaluation.txt
```
あなたは厳格な品質評価専門家です。
指定された基準に基づいて、タスク結果を詳細に評価してください。

# タスク情報
- タスク名: {task_name}
- 説明: {task_description}

# 評価対象の結果
{result_content}

# 評価基準と重み
{criteria}

# 詳細レベル
詳細な分析: {detailed}

# 出力形式
次のJSON形式で出力してください:
{{
  "scores": {{
    "基準名1": スコア（0-100のスケール）,
    "基準名2": スコア（0-100のスケール）,
    ...
  }},
  "overall_score": 総合スコア（0-100のスケール）,
  "feedback": "総合的なフィードバック",
  "improvements": [
    "改善点1",
    "改善点2",
    ...
  ]
}}

# 評価要件
1. 最大限厳しく評価すること
2. 各基準について詳細な分析を行うこと
3. 具体的かつ実行可能な改善点を提案すること
4. 総合スコアは各基準の重み付き平均で計算すること
```

## 4. エントリーポイント

```python
# main.py
import asyncio
import logging
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp_server import app
from config import settings

async def main():
    """MCPサーバーを起動する."""
    # ロガーの設定
    logging.basicConfig(level=getattr(logging, settings.log_level))
    logger = logging.getLogger("ollama-mcp-server")
    logger.info("Starting Ollama MCP Server...")
    
    # サーバーを起動
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ollama-MCP-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## 5. プロジェクト構造

```
ollama-MCP-server/
├── src/
│   ├── ollama_mcp_server/
│   │   ├── __init__.py
│   │   ├── config.py           # 設定管理
│   │   ├── ollama_client.py    # Ollamaクライアント
│   │   ├── repository.py       # データリポジトリ
│   │   ├── models.py           # データモデル
│   │   ├── prompts.py          # プロンプト管理
│   │   ├── mcp_server.py       # MCPサーバー実装
│   │   ├── templates/          # プロンプトテンプレート
│   │   │   ├── task_decomposition.txt
│   │   │   └── result_evaluation.txt
│   │   └── main.py             # エントリーポイント
│   └── __main__.py             # パッケージエントリーポイント
├── tests/
│   ├── __init__.py
│   ├── test_ollama_client.py   # Ollamaクライアントのテスト
│   ├── test_mcp_server.py      # MCPサーバーのテスト
│   └── test_integration.py     # 統合テスト
├── .env.template               # 環境変数テンプレート
├── pyproject.toml              # プロジェクト設定
├── README.md                   # プロジェクト説明
└── .gitignore                  # Gitの除外設定
```

## 6. 実装フェーズ

### フェーズ1：基本構造とテスト
1. プロジェクト構造を作成
2. 依存関係を定義（pyproject.toml）
3. 設定管理モジュールを実装
4. ユニットテストフレームワークを設定
5. テスト駆動開発で初期テストを作成

### フェーズ2：Ollamaクライアント
1. OllamaClientを実装
2. ユニットテストでクライアントの動作を検証
3. エラーハンドリングとリトライロジックを追加
4. JSON生成機能を強化

### フェーズ3：データモデルとリポジトリ
1. データモデルを定義
2. リポジトリクラスを実装
3. CRUD操作をサポート
4. テストで機能を検証

### フェーズ4：プロンプトテンプレート
1. テンプレートファイルを作成
2. プロンプト管理クラスを実装
3. テンプレートのユニットテストを作成

### フェーズ5：MCPサーバー
1. リソース管理機能を実装
2. プロンプト管理機能を実装
3. ツール実行機能を実装
4. 各ハンドラーのテストを作成

### フェーズ6：統合とテスト
1. すべてのコンポーネントを統合
2. 統合テストで全体の動作を検証
3. エッジケースとエラー処理をテスト
4. パフォーマンステストを実施

### フェーズ7：品質向上とドキュメント
1. コードレビューと最適化
2. ドキュメントの充実
3. READMEとサンプルの作成
4. パッケージングとデプロイ準備


テストファイルの修正と追加
非同期テストの適切な実装
サーバーのコンテキスト管理の課題を解決
モデルのテストケース追加
コード品質の向上
テストカバレッジの向上
Pydantic 2.0互換性の検証
エラー処理の改善

統合テストの強化 - サーバー全体の動作をシミュレートするテスト
実際のOllamaサーバーとの統合テスト
エッジケースと例外処理のテスト追加

## 7. テスト戦略

1. **ユニットテスト**：各モジュールの機能を独立してテスト
2. **モックテスト**：外部依存性（Ollama API）をモック化
3. **統合テスト**：コンポーネント間の連携を検証
4. **エンドツーエンドテスト**：実際の環境での動作確認

## 8. 注意点と考慮事項

1. **設定の外部化**：Ollamaのホスト、モデル名、パラメータなどを設定ファイルで管理
2. **エラーハンドリング**：Ollamaサービスの障害に対する適切な対応
3. **キャッシュ戦略**：頻繁に使用するプロンプトや結果をキャッシュして応答時間を改善
4. **メモリ管理**：大量のタスクや結果を効率的に管理
5. **セキュリティ**：センシティブなプロンプトやデータの適切な扱い
6. **スケーラビリティ**：将来的な機能拡張に備えたモジュラーな設計

この計画に基づき、テスト駆動開発のアプローチでOllama-MCP-Serverを実装していきます。まず最初にテストコードを書き、それに合わせて実装を進めることで、高品質で信頼性の高いシステムを構築できます。

