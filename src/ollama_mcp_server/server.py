import asyncio
import logging
import json

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from typing import Dict, List, Any, Sequence, Optional

from .ollama_client import OllamaClient
from .repository import Repository
from .prompts import PromptTemplates
from .config import settings

# ロガーの設定
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger("ollama-mcp-server")

# リポジトリの初期化
repo = Repository()

# Ollamaクライアントの初期化
ollama_client = OllamaClient()

# MCPサーバーの初期化
server = Server("ollama-MCP-server")

@server.list_resources()
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

@server.read_resource()
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

@server.list_prompts()
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

@server.get_prompt()
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

@server.list_tools()
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

@server.call_tool()
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
        await server.request_context.session.send_resource_list_changed()
        
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
        await server.request_context.session.send_resource_list_changed()
        
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
        await server.request_context.session.send_resource_list_changed()
        
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

async def main():
    """MCPサーバーを起動する."""
    # ロガーの設定
    logging.basicConfig(level=getattr(logging, settings.log_level))
    logger = logging.getLogger("ollama-mcp-server")
    logger.info("Starting Ollama MCP Server...")
    
    # サーバーを起動
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ollama-MCP-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )