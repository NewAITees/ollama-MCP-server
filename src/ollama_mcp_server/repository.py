# repository.py
import uuid
from typing import Dict, List, Optional
from .models import Task, Subtask, Result, Evaluation

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