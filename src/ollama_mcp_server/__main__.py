#!/usr/bin/env python3
"""
Ollama MCP サーバーのメインエントリーポイント
"""
import asyncio
import sys
import argparse
import logging
from importlib.metadata import version, PackageNotFoundError

from .server import main
from .config import settings


def parse_args():
    """コマンドライン引数のパース"""
    try:
        pkg_version = version("ollama-mcp-server")
    except PackageNotFoundError:
        pkg_version = "開発版"

    parser = argparse.ArgumentParser(
        description="Ollama MCP サーバー - Ollamaモデルを活用したMCP互換サーバー"
    )
    parser.add_argument(
        "--version", "-v", action="version", 
        version=f"Ollama MCP サーバー {pkg_version}"
    )
    parser.add_argument(
        "--log-level", "-l", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=settings.log_level,
        help="ログレベルを設定"
    )
    
    return parser.parse_args()


def run():
    """メイン関数"""
    args = parse_args()
    
    # ログレベルの設定
    settings.log_level = args.log_level
    
    try:
        # サーバーの起動
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("ユーザーによる中断")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"致命的なエラー: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run() 