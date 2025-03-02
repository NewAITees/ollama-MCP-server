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
    
    def dict(self):
        """互換性のために追加。"""
        return self.model_dump()
    
class Subtask(BaseModel):
    """サブタスクモデル."""
    id: str
    task_id: str
    description: str
    order: int
    completed: bool = False
    
    def dict(self):
        """互換性のために追加。"""
        return self.model_dump()
    
class Result(BaseModel):
    """結果モデル."""
    id: str
    task_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self):
        """互換性のために追加。"""
        return self.model_dump()
    
class Evaluation(BaseModel):
    """評価モデル."""
    id: str
    result_id: str
    criteria: Dict[str, float]
    scores: Dict[str, float]
    feedback: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self):
        """互換性のために追加。"""
        return self.model_dump() 