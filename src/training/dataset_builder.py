import os
import json
from typing import Dict, Any, List

class DatasetBuilder:
    def __init__(self, settings: Any):
        self.settings = settings
        self.dataset_dir = self.settings.paths.sft_dataset
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.dataset_file = os.path.join(self.dataset_dir, "sft_train.jsonl")

    def build_sft_sample(self, ctd_md: str, true_doc: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """Builds a structured instruction-tuning dataset pair.
        Injects error token guides in instructions to penalize mistakes.
        """
        justification = report.get("justification", "")
        error_tokens = report.get("error_tokens", [])
        
        # Instruction construction integrating target-guided feedback
        error_guide = ""
        if error_tokens:
            error_guide = f"特に以下の用語/数値について正確に記述してください: {', '.join(error_tokens)}.\n피드백: {justification}\n"
            
        instruction = (
            "あなたは日本の医薬品規制に精通した専門の医薬品文書作成エキスパートです。\n"
            "与えられたCTDデータから、数値を正確に反映させて文書を作成してください。\n"
            f"{error_guide}"
        )
        
        return {
            "instruction": instruction,
            "input": ctd_md[:4000], # limit chunk length context size
            "output": true_doc
        }

    def append_and_update(self, ctd_md: str, true_if: str, true_siori: str, report: Dict[str, Any]):
        """Creates training pairs for IF and Siori separately, and appends them to target dataset."""
        if_sample = self.build_sft_sample(ctd_md, true_if, report)
        siori_sample = self.build_sft_sample(ctd_md, true_siori, report)
        
        # Read existing samples to avoid duplication or overwrite
        existing_samples = []
        if os.path.exists(self.dataset_file):
            try:
                with open(self.dataset_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            existing_samples.append(json.loads(line.strip()))
            except Exception as e:
                print(f"[DatasetBuilder] Warning reading dataset file: {e}")

        # Add new sample pair
        existing_samples.append(if_sample)
        existing_samples.append(siori_sample)

        # Save cumulative dataset
        with open(self.dataset_file, "w", encoding="utf-8") as f:
            for sample in existing_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                
        print(f"[DatasetBuilder] Cumulative training dataset updated. Total samples: {len(existing_samples)}")
