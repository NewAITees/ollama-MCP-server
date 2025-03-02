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