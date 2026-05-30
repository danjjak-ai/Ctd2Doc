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

Output Format MUST be strict JSON as below:
{
  "total_score": 75,
  "metrics": {
    "numerical_accuracy": 70,
    "format_compliance": 80,
    "linguistic_appropriateness": 75
  },
  "justification": "Detailed reason why points were deducted.",
  "error_tokens": ["5mg", "12대 부작용"]
}"""

    def evaluate(self, generated_doc: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluates generated doc against ground truth."""
        user_content = f"""[Ground Truth Document]
{ground_truth}

[Generated Document]
{generated_doc}

Please evaluate the Generation and return the JSON score report."""

        # In Colab/CUDA, we can use Gemma-2-27b-it to judge, or use API Fallbacks
        # Here we mock/implement parser with regex verification
        print("[JudgeAgent] Running evaluation...")
        
        # 1. API call fallback checking (Gemini API)
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=self.system_prompt)
                response = model.generate_content(user_content)
                parsed = self._parse_json_safely(response.text)
                if parsed:
                    return parsed
            except Exception as e:
                print(f"[JudgeAgent] API Call failed: {e}. Falling back to default scoring.")

        # 2. Local Fallback/Mock Evaluation Strategy based on string analysis
        return self._rule_based_fallback_evaluation(generated_doc, ground_truth)

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
        """Performs logical diff comparison for fallback scoring."""
        numerical_score = 100
        format_score = 90
        linguistic_score = 95
        error_tokens = []
        justification = "All clinical entities matched."

        # Scan for percentages and dosages matching
        # Regex to find numbers/percentages/dosages
        num_patterns = re.findall(r"\b\d+\.?\d*%\b|\b\d+\.?\d*\s*mg\b", gt)
        for pattern in num_patterns:
            if pattern not in gen:
                numerical_score -= 10
                error_tokens.append(pattern)
        
        numerical_score = max(numerical_score, 10)
        if error_tokens:
            justification = f"Mismatch or omissions found in crucial values: {error_tokens}"
        
        total_score = int(0.4 * numerical_score + 0.3 * format_score + 0.3 * linguistic_score)
        
        return {
            "total_score": total_score,
            "metrics": {
                "numerical_accuracy": numerical_score,
                "format_compliance": format_score,
                "linguistic_appropriateness": linguistic_score
            },
            "justification": justification,
            "error_tokens": error_tokens
        }
