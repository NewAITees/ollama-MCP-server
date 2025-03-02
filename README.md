# ollama-MCP-server

Ollamaと通信するModel Context Protocol (MCP) サーバー

## 概要

このMCPサーバーは、ローカルのOllama LLMインスタンスとMCP互換アプリケーションの間でシームレスな統合を可能にし、高度なタスク分解、評価、ワークフロー管理を提供します。

主な機能:
- 複雑な問題のタスク分解
- 結果の評価と検証
- Ollamaモデルの管理と実行
- MCPプロトコルによる標準化された通信
- 高度なエラー処理と詳細なエラーメッセージ
- パフォーマンス最適化（コネクションプーリング、LRUキャッシュ）

## コンポーネント

### リソース

サーバーは以下のリソースを実装しています:
- **task://** - 個別のタスクにアクセスするためのURIスキーム
- **result://** - 評価結果にアクセスするためのURIスキーム
- **model://** - 利用可能なOllamaモデルにアクセスするためのURIスキーム

各リソースには、最適なLLMとの対話のための適切なメタデータとMIMEタイプが設定されています。

### プロンプトとツールの関係

MCPサーバーでは、プロンプトとツールは密接に関連していますが、異なる役割を持っています。

- **プロンプト**：LLMに特定の思考方法や構造を提供するスキーマ（Schema）のような役割
- **ツール**：実際にアクションを実行するハンドラー（Handler）のような役割

各ツールには対応するスキーマ（プロンプト）が必要であり、これによりLLMの思考能力と実際のシステム機能を効果的に連携させることができます。

### プロンプト

サーバーはいくつかの特殊なプロンプトを提供します:
- **decompose-task** - 複雑なタスクを管理しやすいサブタスクに分解
  - タスクの説明と粒度レベルのオプションパラメータを取得
  - 依存関係と推定複雑性を含む構造化された内訳を返す
  
- **evaluate-result** - 指定された基準に対してタスク結果を分析
  - 結果の内容と評価パラメータを取得
  - スコアと改善提案を含む詳細な評価を返す

### ツール

サーバーはいくつかの強力なツールを実装しています:

- **add-task**
  - 必須パラメータ: `name` (文字列), `description` (文字列)
  - オプションパラメータ: `priority` (数値), `deadline` (文字列), `tags` (配列)
  - システムに新しいタスクを作成し、その識別子を返す
  - 対応するスキーマ: タスク作成のためのデータ検証スキーマ

- **decompose-task**
  - 必須パラメータ: `task_id` (文字列), `granularity` (文字列: "high"|"medium"|"low")
  - オプションパラメータ: `max_subtasks` (数値)
  - Ollamaを使用して複雑なタスクを管理可能なサブタスクに分解
  - 対応するスキーマ: 上記の`decompose-task`プロンプト

- **evaluate-result**
  - 必須パラメータ: `result_id` (文字列), `criteria` (オブジェクト)
  - オプションパラメータ: `detailed` (ブール値)
  - 指定された基準に対して結果を評価し、フィードバックを提供
  - 対応するスキーマ: 上記の`evaluate-result`プロンプト

- **run-model**
  - 必須パラメータ: `model` (文字列), `prompt` (文字列)
  - オプションパラメータ: `temperature` (数値), `max_tokens` (数値)
  - 指定されたパラメータでOllamaモデルを実行
  - 対応するスキーマ: Ollamaモデル実行パラメータの検証スキーマ

## 新機能と改善点

### 拡張エラー処理

サーバーは、より詳細で構造化されたエラーメッセージを提供します。これにより、クライアントアプリケーションはエラーをより効果的に処理できます。エラーレスポンスの例:

```json
{
  "error": {
    "message": "Task not found: task-123",
    "status_code": 404,
    "details": {
      "provided_id": "task-123"
    }
  }
}
```

### パフォーマンス最適化

- **コネクションプーリング**: 共有HTTP接続プールを使用することで、リクエストのパフォーマンスが向上し、リソース使用率が低減されます。
- **LRUキャッシュ**: 同一または類似のリクエストに対する応答をキャッシュすることで、レスポンス時間が短縮され、Ollamaサーバーの負荷が軽減されます。

これらの設定は `config.py` で調整できます:

```python
# パフォーマンス関連設定
cache_size: int = 100                 # キャッシュに保存する最大エントリ数
max_connections: int = 10             # 同時接続の最大数
max_connections_per_host: int = 10    # ホストごとの最大接続数
request_timeout: int = 60             # リクエストタイムアウト（秒）
```

## テスト

プロジェクトには包括的なテストスイートが含まれています:

- **ユニットテスト**: 個々のコンポーネントの機能をテスト
- **統合テスト**: エンドツーエンドのワークフローをテスト

テストを実行するには:

```bash
# すべてのテストを実行
python -m unittest discover

# 特定のテストを実行
python -m unittest tests.test_integration
```

## 設定

### 環境変数

```
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3
LOG_LEVEL=info
```

### Ollamaのセットアップ

Ollamaがインストールされ、適切なモデルで実行されていることを確認してください:

```bash
# Ollamaをインストール（まだインストールされていない場合）
curl -fsSL https://ollama.com/install.sh | sh

# 推奨モデルをダウンロード
ollama pull llama3
ollama pull mistral
ollama pull qwen2
```

## クイックスタート

### インストール

```bash
pip install ollama-mcp-server
```

### Claude Desktop設定

#### MacOS
パス: `~/Library/Application\ Support/Claude/claude_desktop_config.json`

#### Windows
パス: `%APPDATA%/Claude/claude_desktop_config.json`

<details>
  <summary>開発/未公開サーバーの設定</summary>
  
  ```json
  "mcpServers": {
    "ollama-MCP-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/ollama-MCP-server",
        "run",
        "ollama-MCP-server"
      ]
    }
  }
  ```
</details>

<details>
  <summary>公開サーバーの設定</summary>
  
  ```json
  "mcpServers": {
    "ollama-MCP-server": {
      "command": "uvx",
      "args": [
        "ollama-MCP-server"
      ]
    }
  }
  ```
</details>

## 使用例

### タスク分解

複雑なタスクを管理可能なサブタスクに分解するには:

```python
result = await mcp.use_mcp_tool({
    "server_name": "ollama-MCP-server",
    "tool_name": "decompose-task",
    "arguments": {
        "task_id": "task://123",
        "granularity": "medium",
        "max_subtasks": 5
    }
})
```

### 結果評価

特定の基準に対して結果を評価するには:

```python
evaluation = await mcp.use_mcp_tool({
    "server_name": "ollama-MCP-server",
    "tool_name": "evaluate-result",
    "arguments": {
        "result_id": "result://456",
        "criteria": {
            "accuracy": 0.4,
            "completeness": 0.3,
            "clarity": 0.3
        },
        "detailed": true
    }
})
```

### Ollamaモデルの実行

Ollamaモデルに対して直接クエリを実行するには:

```python
response = await mcp.use_mcp_tool({
    "server_name": "ollama-MCP-server",
    "tool_name": "run-model",
    "arguments": {
        "model": "llama3",
        "prompt": "量子コンピューティングを簡単な言葉で説明してください",
        "temperature": 0.7
    }
})
```

## 開発

### プロジェクトのセットアップ

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/ollama-MCP-server.git
cd ollama-MCP-server
```

2. 仮想環境を作成してアクティベート:
```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

3. 開発依存関係をインストール:
```bash
uv sync --dev --all-extras
```

### ローカル開発

プロジェクトには便利な開発用スクリプトが含まれています：

#### サーバーの実行

```bash
./run_server.sh
```

オプション:
- `--debug`: デバッグモードで実行（ログレベル: DEBUG）
- `--log=LEVEL`: ログレベルを指定（DEBUG, INFO, WARNING, ERROR, CRITICAL）

#### テストの実行

```bash
./run_tests.sh
```

オプション:
- `--unit`: ユニットテストのみ実行
- `--integration`: 統合テストのみ実行
- `--all`: すべてのテストを実行（デフォルト）
- `--verbose`: 詳細なテスト出力

### ビルドと公開

パッケージを配布用に準備するには:

1. 依存関係を同期してロックファイルを更新:
```bash
uv sync
```

2. パッケージの配布物をビルド:
```bash
uv build
```

これにより、`dist/`ディレクトリにソースとホイールの配布物が作成されます。

3. PyPIに公開:
```bash
uv publish
```

注: PyPI認証情報を環境変数またはコマンドフラグで設定する必要があります:
- トークン: `--token`または`UV_PUBLISH_TOKEN`
- またはユーザー名/パスワード: `--username`/`UV_PUBLISH_USERNAME`と`--password`/`UV_PUBLISH_PASSWORD`

### デバッグ

MCPサーバーはstdioを介して実行されるため、デバッグは難しい場合があります。最適なデバッグ
体験のために、[MCP Inspector](https://github.com/modelcontextprotocol/inspector)の使用を強く推奨します。

[`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)を使用してMCP Inspectorを起動するには、次のコマンドを実行します:

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/ollama-MCP-server run ollama-mcp-server
```

起動時、Inspectorはブラウザでアクセスしてデバッグを開始できるURLを表示します。

## アーキテ

## 貢献

貢献は歓迎します！お気軽にプルリクエストを提出してください。

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細はLICENSEファイルを参照してください。

## 謝辞

- 優れたプロトコル設計を提供した[Model Context Protocol](https://modelcontextprotocol.io)チーム
- ローカルLLM実行をアクセス可能にした[Ollama](https://ollama.com)プロジェクト
- このプロジェクトのすべての貢献者