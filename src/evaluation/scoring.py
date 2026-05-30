import os
import json
import time
from typing import Dict, Any, List

class ScoringSystem:
    def __init__(self, settings: Any):
        self.settings = settings
        self.report_dir = self.settings.paths.outputs_reports
        os.makedirs(self.report_dir, exist_ok=True)
        self.history_file = os.path.join(self.report_dir, "score_history.jsonl")

    def log_score(self, japic_code: str, iteration: int, report: Dict[str, Any]):
        """Logs individual iteration results to score_history.jsonl."""
        entry = {
            "timestamp": time.time(),
            "japic_code": japic_code,
            "iteration": iteration,
            "total_score": report.get("total_score", 0),
            "metrics": report.get("metrics", {}),
            "justification": report.get("justification", ""),
            "error_tokens": report.get("error_tokens", [])
        }
        
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Save detailed report per drug iteration
        detail_file = os.path.join(self.report_dir, f"report_{japic_code}_iter_{iteration}.json")
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
            
        print(f"[ScoringSystem] Iteration {iteration} result logged. Total Score: {entry['total_score']}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Reads all logs from score_history.jsonl."""
        if not os.path.exists(self.history_file):
            return []
        
        history = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line.strip()))
        return history
