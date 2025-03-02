#!/bin/bash
# 開発用サーバー起動スクリプト

# 色の設定
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Ollama MCP サーバー起動${NC}"
echo "=================================================="

# 引数チェック
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "使用方法: $0 [オプション]"
  echo ""
  echo "オプション:"
  echo "  --debug     デバッグモードで実行 (ログレベル: DEBUG)"
  echo "  --log=LEVEL ログレベルを指定 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
  echo "  --help, -h  このヘルプを表示"
  exit 0
fi

# デフォルト設定
LOG_LEVEL="INFO"

# 引数処理
for arg in "$@"
do
  case $arg in
    --debug)
      LOG_LEVEL="DEBUG"
      shift
      ;;
    --log=*)
      LOG_LEVEL="${arg#*=}"
      shift
      ;;
  esac
done

# 環境設定
if [ -f ".env" ]; then
  echo -e "${BLUE}INFO:${NC} .env ファイルを読み込みます"
else
  echo -e "${YELLOW}警告:${NC} .env ファイルが見つかりません。デフォルト設定を使用します"
  if [ -f ".env.template" ]; then
    echo -e "${BLUE}INFO:${NC} .env.template から .env ファイルを作成します"
    cp .env.template .env
  fi
fi

# Ollamaサーバーの確認
echo -e "${BLUE}INFO:${NC} Ollamaサーバーの接続を確認しています..."
HOST=$(grep "OLLAMA_HOST" .env 2>/dev/null | cut -d '=' -f2 || echo "http://localhost:11434")
curl -s --connect-timeout 5 "$HOST" > /dev/null
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}警告:${NC} Ollamaサーバーに接続できません。サーバーが実行中であることを確認してください"
  echo -e "ホスト: $HOST"
else
  echo -e "${GREEN}成功:${NC} Ollamaサーバーに接続できました"
fi

# サーバー起動
echo -e "\n${YELLOW}Ollama MCPサーバーを起動中...${NC}"
echo -e "ログレベル: ${LOG_LEVEL}"
echo "=================================================="

# .venv/bin/python または python を使用してサーバーを起動
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

$PYTHON -m ollama_mcp_server --log-level $LOG_LEVEL 