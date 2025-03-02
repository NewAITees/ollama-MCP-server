#!/bin/bash
# テスト実行スクリプト

# 色の設定
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Ollama MCP サーバーテスト${NC}"
echo "=================================================="

# 引数チェック
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "使用方法: $0 [オプション]"
  echo ""
  echo "オプション:"
  echo "  --unit      ユニットテストのみ実行"
  echo "  --integration  統合テストのみ実行"
  echo "  --all       すべてのテストを実行 (デフォルト)"
  echo "  --verbose   詳細なテスト出力"
  echo "  --help, -h  このヘルプを表示"
  exit 0
fi

# デフォルト設定
RUN_UNIT=true
RUN_INTEGRATION=true
VERBOSE=""

# 引数処理
for arg in "$@"
do
  case $arg in
    --unit)
      RUN_UNIT=true
      RUN_INTEGRATION=false
      shift
      ;;
    --integration)
      RUN_UNIT=false
      RUN_INTEGRATION=true
      shift
      ;;
    --all)
      RUN_UNIT=true
      RUN_INTEGRATION=true
      shift
      ;;
    --verbose)
      VERBOSE="-v"
      shift
      ;;
  esac
done

echo "テスト環境を準備中..."

# 成功カウンター
PASSED=0
FAILED=0

# テスト実行関数
run_test() {
  TEST_TYPE=$1
  TEST_FILE=$2
  
  echo -e "\n${YELLOW}$TEST_TYPE テスト: $TEST_FILE${NC}"
  echo "------------------------------------------------"
  
  python -m unittest $VERBOSE $TEST_FILE
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ $TEST_TYPE テスト: $TEST_FILE 成功${NC}"
    PASSED=$((PASSED+1))
  else
    echo -e "${RED}✗ $TEST_TYPE テスト: $TEST_FILE 失敗${NC}"
    FAILED=$((FAILED+1))
  fi
}

# ユニットテスト
if [ "$RUN_UNIT" = true ]; then
  echo -e "\n${YELLOW}ユニットテストを実行中...${NC}"
  
  if [ -f "tests/test_models.py" ]; then
    run_test "モデル" "tests.test_models"
  fi
  
  if [ -f "tests/test_ollama_client.py" ]; then
    run_test "Ollamaクライアント" "tests.test_ollama_client"
  fi
  
  if [ -f "tests/test_mcp_server.py" ]; then
    run_test "MCPサーバー" "tests.test_mcp_server"
  fi
fi

# 統合テスト
if [ "$RUN_INTEGRATION" = true ]; then
  echo -e "\n${YELLOW}統合テストを実行中...${NC}"
  
  if [ -f "tests/test_integration.py" ]; then
    run_test "統合" "tests.test_integration"
  fi
fi

# 結果表示
echo -e "\n${YELLOW}テスト結果${NC}"
echo "=================================================="
echo -e "合計: $((PASSED+FAILED)) テスト実行"
echo -e "${GREEN}成功: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
  echo -e "${RED}失敗: $FAILED${NC}"
  exit 1
else
  echo -e "失敗: $FAILED"
  echo -e "\n${GREEN}すべてのテストが成功しました！${NC}"
  exit 0
fi 