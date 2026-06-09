import os
import json
import re
from typing import Dict, Any


class JudgeAgent:
    def __init__(self, settings: Any):
        self.settings = settings
        self.device = self.settings.model.device
        self.system_prompt = """You are an expert medical document auditor fluent in English and Japanese.
Your task is to compare the 'Generated Document' with the 'Ground Truth Document' (Interview Form / Kusuri-no-Siori) based on the provided CTD data.

Evaluate the generation across three strict dimensions, scoring each out of 100:
1. Numerical Accuracy (수치 정확성): Are clinical percentages, dosages (mg), and frequencies exactly matched without any hallucination? (Critical)
2. Format Compliance (포맷 준수성): Does it precisely follow the structured medical layout guidelines?
3. Linguistic Appropriateness (언어적 적절성): Is the Japanese medical terminology natural, and does "Kusuri-no-Siori" use patient-friendly language?

Also categorize each error into one of these types:
- "numerical": Wrong numbers, percentages, dosages
- "omission": Missing required sections or data points
- "format": Structural/layout violations
- "terminology": Incorrect medical terms

Output Format MUST be strict JSON as below:
{
  "total_score": 75,
  "metrics": {
    "numerical_accuracy": 70,
    "format_compliance": 80,
    "linguistic_appropriateness": 75
  },
  "justification": "Detailed reason why points were deducted.",
  "error_tokens": ["5mg", "12.0%"],
  "error_categories": {
    "numerical": ["5mg should be 0.5mg"],
    "omission": ["Missing 薬物動態 section"],
    "format": [],
    "terminology": []
  }
}"""

    def evaluate(self, generated_doc: str, ground_truth: str) -> Dict[str, Any]:
        """생성된 문서를 정답과 비교하여 평가.

        1차: Gemini API를 사용하여 LLM-as-a-Judge 평가
        2차: API 실패 시 규칙 기반 폴백 평가
        """
        user_content = f"""[Ground Truth Document]
{ground_truth}

[Generated Document]
{generated_doc}

Please evaluate the Generation and return the JSON score report."""

        print("[JudgeAgent] Running evaluation...")

        # 1. Gemini API 평가
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    'gemini-2.0-flash',
                    system_instruction=self.system_prompt,
                )
                response = model.generate_content(user_content)
                parsed = self._parse_json_safely(response.text)
                if parsed:
                    # error_categories가 없으면 기본값 추가
                    if "error_categories" not in parsed:
                        parsed["error_categories"] = {
                            "numerical": [], "omission": [],
                            "format": [], "terminology": [],
                        }
                    return parsed
            except Exception as e:
                print(f"[JudgeAgent] API Call failed: {e}. Falling back to rule-based scoring.")

        # 2. Rule-based 폴백 평가
        return self._rule_based_fallback_evaluation(generated_doc, ground_truth)

    def evaluate_dual(self, generated_if: str, true_if: str,
                      generated_siori: str, true_siori: str,
                      if_weight: float = 0.6, siori_weight: float = 0.4) -> Dict[str, Any]:
        """IF와 Siori를 독립적으로 평가하고 가중 합산.

        Args:
            if_weight: IF 평가의 가중치 (기본 0.6)
            siori_weight: Siori 평가의 가중치 (기본 0.4)

        Returns:
            통합 평가 리포트
        """
        print("[JudgeAgent] === Dual Evaluation (IF + Siori) ===")

        # IF 평가
        print("  [IF] Evaluating Interview Form...")
        if_report = self.evaluate(generated_if, true_if)
        if_score = if_report.get("total_score", 0)
        print(f"  [IF] Score: {if_score}")

        # Siori 평가
        print("  [Siori] Evaluating Kusuri-no-Siori...")
        siori_report = self.evaluate(generated_siori, true_siori)
        siori_score = siori_report.get("total_score", 0)
        print(f"  [Siori] Score: {siori_score}")

        # 가중 합산
        combined_score = int(if_score * if_weight + siori_score * siori_weight)

        # 에러 토큰 병합
        all_error_tokens = list(set(
            if_report.get("error_tokens", []) + siori_report.get("error_tokens", [])
        ))

        # 에러 카테고리 병합
        combined_categories = {"numerical": [], "omission": [], "format": [], "terminology": []}
        for category in combined_categories:
            if_cats = if_report.get("error_categories", {}).get(category, [])
            siori_cats = siori_report.get("error_categories", {}).get(category, [])
            combined_categories[category] = list(set(if_cats + siori_cats))

        combined_report = {
            "total_score": combined_score,
            "if_score": if_score,
            "siori_score": siori_score,
            "if_weight": if_weight,
            "siori_weight": siori_weight,
            "metrics": {
                "numerical_accuracy": int(
                    if_report["metrics"]["numerical_accuracy"] * if_weight +
                    siori_report["metrics"]["numerical_accuracy"] * siori_weight
                ),
                "format_compliance": int(
                    if_report["metrics"]["format_compliance"] * if_weight +
                    siori_report["metrics"]["format_compliance"] * siori_weight
                ),
                "linguistic_appropriateness": int(
                    if_report["metrics"]["linguistic_appropriateness"] * if_weight +
                    siori_report["metrics"]["linguistic_appropriateness"] * siori_weight
                ),
            },
            "justification": (
                f"[IF] {if_report.get('justification', '')}\n"
                f"[Siori] {siori_report.get('justification', '')}"
            ),
            "error_tokens": all_error_tokens,
            "error_categories": combined_categories,
            "sub_reports": {
                "interview_form": if_report,
                "siori": siori_report,
            },
        }

        print(f"  [Combined] Weighted Score: {combined_score} "
              f"(IF: {if_score}×{if_weight} + Siori: {siori_score}×{siori_weight})")

        return combined_report

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        try:
            # Extract JSON block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception:
            return None

    def _rule_based_fallback_evaluation(self, gen: str, gt: str) -> Dict[str, Any]:
        """규칙 기반 폴백 평가: 수치, 포맷, 언어적 분석."""
        numerical_score = 100
        format_score = 90
        linguistic_score = 95
        error_tokens = []
        error_categories = {"numerical": [], "omission": [], "format": [], "terminology": []}
        justification_parts = []

        # 1. 수치 정확성 검증 (퍼센트, 용량)
        num_patterns = re.findall(r"\b\d+\.?\d*%\b|\b\d+\.?\d*\s*mg\b", gt)
        for pattern in num_patterns:
            if pattern not in gen:
                numerical_score -= 10
                error_tokens.append(pattern)
                error_categories["numerical"].append(f"Missing: {pattern}")

        # 2. 포맷 검증 (주요 섹션 존재 여부)
        required_sections_if = ["副作用", "用法", "用量"]
        required_sections_siori = ["副作用", "用法", "用量"]
        for section in required_sections_if:
            if section in gt and section not in gen:
                format_score -= 15
                error_categories["omission"].append(f"Missing section: {section}")

        # 3. 점수 하한 설정
        numerical_score = max(numerical_score, 10)
        format_score = max(format_score, 10)

        if error_tokens:
            justification_parts.append(f"Numerical mismatches: {error_tokens}")
        if error_categories["omission"]:
            justification_parts.append(f"Missing sections: {error_categories['omission']}")

        justification = "; ".join(justification_parts) if justification_parts else "All clinical entities matched."

        total_score = int(0.4 * numerical_score + 0.3 * format_score + 0.3 * linguistic_score)

        return {
            "total_score": total_score,
            "metrics": {
                "numerical_accuracy": numerical_score,
                "format_compliance": format_score,
                "linguistic_appropriateness": linguistic_score,
            },
            "justification": justification,
            "error_tokens": error_tokens,
            "error_categories": error_categories,
        }
