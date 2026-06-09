import os
import json
import time
from typing import Dict, Any, List, Optional


class ScoringSystem:
    def __init__(self, settings: Any):
        self.settings = settings
        self.report_dir = self.settings.paths.outputs_reports
        os.makedirs(self.report_dir, exist_ok=True)
        self.history_file = os.path.join(self.report_dir, "score_history.jsonl")

    def log_score(self, japic_code: str, iteration: int, report: Dict[str, Any]):
        """개별 이터레이션 결과를 score_history.jsonl에 기록."""
        entry = {
            "timestamp": time.time(),
            "japic_code": japic_code,
            "iteration": iteration,
            "total_score": report.get("total_score", 0),
            "if_score": report.get("if_score"),
            "siori_score": report.get("siori_score"),
            "metrics": report.get("metrics", {}),
            "justification": report.get("justification", ""),
            "error_tokens": report.get("error_tokens", []),
            "error_categories": report.get("error_categories", {}),
        }

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Save detailed report per drug iteration
        detail_file = os.path.join(
            self.report_dir, f"report_{japic_code}_iter_{iteration}.json"
        )
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        print(f"[ScoringSystem] Iteration {iteration} result logged. "
              f"Total Score: {entry['total_score']}")

    def get_history(self) -> List[Dict[str, Any]]:
        """모든 평가 이력을 로드."""
        if not os.path.exists(self.history_file):
            return []

        history = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line.strip()))
        return history

    def get_drug_history(self, japic_code: str) -> List[Dict[str, Any]]:
        """특정 약품의 이터레이션별 평가 이력을 반환."""
        history = self.get_history()
        return [h for h in history if h.get("japic_code") == japic_code]

    def get_score_trend(self, japic_code: Optional[str] = None) -> Dict[str, Any]:
        """점수 추세를 분석.

        Returns:
            scores: 이터레이션별 점수 리스트
            improvement: 초기 대비 최종 점수 향상폭
            best/worst/avg: 통계
            improving: 추세가 향상 중인지 여부
        """
        history = self.get_drug_history(japic_code) if japic_code else self.get_history()

        if not history:
            return {"scores": [], "improvement": 0, "best": 0, "worst": 0, "avg": 0, "improving": False}

        scores = [h["total_score"] for h in history]

        # 최근 3개 점수로 추세 판단
        improving = False
        if len(scores) >= 3:
            recent = scores[-3:]
            improving = recent[-1] > recent[0]

        return {
            "scores": scores,
            "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0,
            "best": max(scores),
            "worst": min(scores),
            "avg": round(sum(scores) / len(scores), 1),
            "improving": improving,
            "total_iterations": len(scores),
        }

    def print_summary(self, japic_code: Optional[str] = None):
        """점수 요약을 콘솔에 출력."""
        trend = self.get_score_trend(japic_code)
        label = f" (Drug: {japic_code})" if japic_code else ""
        print(f"\n[ScoringSystem] === Score Summary{label} ===")
        print(f"  Iterations: {trend['total_iterations']}")
        print(f"  Best: {trend['best']} | Worst: {trend['worst']} | Avg: {trend['avg']}")
        print(f"  Improvement: {trend['improvement']:+d}")
        print(f"  Trend: {'📈 Improving' if trend['improving'] else '📉 Stagnating'}")
        if trend["scores"]:
            print(f"  Score History: {' → '.join(str(s) for s in trend['scores'])}")
        print()
