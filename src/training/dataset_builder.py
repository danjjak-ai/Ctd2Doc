import os
import json
import hashlib
from typing import Dict, Any, List


class DatasetBuilder:
    def __init__(self, settings: Any):
        self.settings = settings
        self.dataset_dir = self.settings.paths.sft_dataset
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.dataset_file = os.path.join(self.dataset_dir, "sft_train.jsonl")
        self._seen_hashes: set = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        """기존 데이터셋에서 중복 방지를 위한 해시를 로드."""
        if os.path.exists(self.dataset_file):
            try:
                with open(self.dataset_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line.strip())
                            h = self._hash_sample(sample)
                            self._seen_hashes.add(h)
            except Exception as e:
                print(f"[DatasetBuilder] Warning reading dataset file: {e}")

    @staticmethod
    def _hash_sample(sample: dict) -> str:
        """데이터 샘플의 고유 해시를 생성 (중복 방지용)."""
        key = f"{sample.get('instruction', '')}|{sample.get('input', '')[:200]}|{sample.get('output', '')[:200]}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def build_sft_sample(self, ctd_md: str, true_doc: str,
                         report: Dict[str, Any]) -> Dict[str, Any]:
        """SFT 학습용 instruction-tuning 데이터 페어를 생성.

        error_tokens 기반으로 오류 수정 가이드를 instruction에 주입하여
        모델이 특정 수치/용어에 집중하도록 유도.
        """
        justification = report.get("justification", "")
        error_tokens = report.get("error_tokens", [])

        # Instruction에 에러 기반 피드백 주입
        error_guide = ""
        if error_tokens:
            error_guide = (
                f"\n\n【重要な修正ポイント】\n"
                f"以下の用語・数値について特に正確に記述してください: "
                f"{', '.join(error_tokens)}.\n"
                f"前回のフィードバック: {justification}\n"
            )

        instruction = (
            "あなたは日本の医薬品規制に精通した専門の医薬品文書作成エキスパートです。\n"
            "与えられたCTDデータから、数値を正確に反映させて文書を作成してください。"
            f"{error_guide}"
        )

        return {
            "instruction": instruction,
            "input": ctd_md[:6000],  # Gemma 4의 넓은 컨텍스트 활용 (4096→6000)
            "output": true_doc,
        }

    def append_and_update(self, ctd_md: str, true_if: str, true_siori: str,
                          report: Dict[str, Any]):
        """IF와 Siori 학습 페어를 생성하고 데이터셋에 중복 없이 추가."""
        if_sample = self.build_sft_sample(ctd_md, true_if, report)
        siori_sample = self.build_sft_sample(ctd_md, true_siori, report)

        new_samples = []
        for sample in [if_sample, siori_sample]:
            h = self._hash_sample(sample)
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                new_samples.append(sample)

        if not new_samples:
            print("[DatasetBuilder] No new unique samples to add (duplicates skipped).")
            return

        # Append 모드로 신규 샘플만 추가 (기존 데이터 보존)
        with open(self.dataset_file, "a", encoding="utf-8") as f:
            for sample in new_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        total_count = len(self._seen_hashes)
        print(f"[DatasetBuilder] Added {len(new_samples)} new samples. "
              f"Total dataset size: {total_count}")

    def get_dataset_stats(self) -> Dict[str, Any]:
        """현재 데이터셋의 통계 정보를 반환."""
        if not os.path.exists(self.dataset_file):
            return {"total_samples": 0, "file_size_kb": 0}

        count = 0
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1

        file_size = os.path.getsize(self.dataset_file)
        return {
            "total_samples": count,
            "file_size_kb": round(file_size / 1024, 1),
            "dataset_path": self.dataset_file,
        }
