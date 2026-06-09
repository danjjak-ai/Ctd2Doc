import os
from typing import Any

from src.model_registry import get_model_preset, build_chat_messages
from src.gpu_utils import free_gpu_memory, print_gpu_status


class Generator:
    def __init__(self, settings: Any):
        self.settings = settings
        self.model_name = self.settings.model.base_model
        self.device = self.settings.model.device
        self.use_4bit = self.settings.model.use_4bit
        self.max_seq_length = self.settings.model.max_seq_length
        self.generation_max_tokens = self.settings.model.generation_max_tokens
        self.model = None
        self.tokenizer = None
        self.preset = get_model_preset(self.model_name)

        # Load prompt templates
        self.templates = {}
        self._load_templates()
        self._initialize_model()

    def _load_templates(self):
        template_dir = "src/inference/templates"
        os.makedirs(template_dir, exist_ok=True)

        # In case templates are not written yet, write default templates
        self.if_template_path = os.path.join(template_dir, "interview_form.txt")
        self.siori_template_path = os.path.join(template_dir, "siori.txt")

        # Set up default template contents if missing
        if not os.path.exists(self.if_template_path):
            with open(self.if_template_path, "w", encoding="utf-8") as f:
                f.write("あなたは日本の医薬品規制に精通した専門の医薬品文書作成エキスパートです。\n"
                        "以下のCTDの関連データに基づき、「医薬品インタビューフォーム」を作成してください。\n\n"
                        "【コンテキスト】\n{retrieved_context}\n\n"
                        "上記のデータに基づき、数値や用法用量を正確に維持したまま文書を作成してください。")

        if not os.path.exists(self.siori_template_path):
            with open(self.siori_template_path, "w", encoding="utf-8") as f:
                f.write("あなたは患者向けの「くすりのしおり」を作成する専門家です。\n"
                        "以下のCTDデータに基づき、平易な日本語で「くすりのしおり」を作成してください。\n\n"
                        "【コンテキスト】\n{retrieved_context}\n\n"
                        "患者が理解しやすいように作成してください。")

        with open(self.if_template_path, "r", encoding="utf-8") as f:
            self.templates["interview_form"] = f.read()
        with open(self.siori_template_path, "r", encoding="utf-8") as f:
            self.templates["siori"] = f.read()

    def _initialize_model(self):
        """Gemma 4 E4B-IT 모델을 Unsloth 또는 HuggingFace로 초기화."""
        if self.device == "cuda":
            # 1차: Unsloth 시도 (Gemma 4 공식 지원)
            if self.preset.get("use_unsloth", False):
                try:
                    from unsloth import FastLanguageModel
                    print(f"[Generator] Loading {self.model_name} with Unsloth 4-bit...")
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.model_name,
                        max_seq_length=self.max_seq_length,
                        dtype=None,  # Auto detection
                        load_in_4bit=self.use_4bit,
                    )
                    FastLanguageModel.for_inference(self.model)
                    print_gpu_status("After model load")
                    return
                except Exception as e:
                    print(f"[Generator] Unsloth load failed: {e}. Trying standard HuggingFace...")

            # 2차: 표준 HuggingFace + BitsAndBytes
            self._initialize_standard_hf()
        else:
            print("[Generator] Running on CPU/Local. Model loading skipped (Fallback mode enabled).")

    def _initialize_standard_hf(self):
        """표준 HuggingFace transformers로 모델 로드 (Unsloth 실패 시 폴백)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ) if self.use_4bit else None

            print(f"[Generator] Loading {self.model_name} with HuggingFace (4-bit: {self.use_4bit})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16 if not self.use_4bit else None,
            )
            print_gpu_status("After HF model load")
        except Exception as e:
            print(f"[Generator] HF loading failed: {e}. Model will run in mock mode.")

    def reload_adapter(self, adapter_path: str):
        """학습 완료 후 LoRA 어댑터를 런타임에 핫스왑.

        Unsloth로 로드된 모델과 표준 PEFT 모델 모두 지원.
        """
        if self.device == "cuda" and self.model is not None:
            try:
                # Unsloth 모델인 경우
                try:
                    from unsloth import FastLanguageModel
                    print(f"[Generator] Reloading Unsloth LoRA adapter: {adapter_path}")
                    self.model.load_adapter(adapter_path)
                    FastLanguageModel.for_inference(self.model)
                    print_gpu_status("After adapter reload")
                    return
                except (ImportError, AttributeError):
                    pass

                # 표준 PEFT 모델인 경우
                from peft import PeftModel
                print(f"[Generator] Reloading PEFT LoRA adapter: {adapter_path}")
                if hasattr(self.model, 'disable_adapter_layers'):
                    self.model.disable_adapter_layers()
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.model.eval()
                print_gpu_status("After PEFT adapter reload")
            except Exception as e:
                print(f"[Generator] Failed to reload adapter: {e}")
        else:
            print(f"[Generator] Fallback mode: Simulating adapter reload from {adapter_path}")

    def unload_model(self):
        """모델을 GPU 메모리에서 완전히 해제 (학습 전환 시 사용)."""
        if self.model is not None:
            from src.gpu_utils import unload_model
            unload_model(self.model, self.tokenizer)
            self.model = None
            self.tokenizer = None
            print("[Generator] Model fully unloaded for training phase.")

    def reload_model(self):
        """학습 완료 후 추론 모델을 다시 로드."""
        print("[Generator] Reloading model for inference phase...")
        self._initialize_model()

    def _build_prompt(self, template_name: str, context: str) -> str:
        """Gemma 4 chat template을 사용하여 프롬프트를 구성.

        Gemma 4는 네이티브 system role을 지원하므로 tokenizer.apply_chat_template을 사용.
        tokenizer가 없는 경우 (CPU/Mock 모드) 수동 포맷팅으로 폴백.
        """
        template = self.templates.get(template_name, "")
        system_prompt = template.replace("{retrieved_context}", "").strip()
        user_content = f"【コンテキスト】\n{context}"

        if self.tokenizer is not None:
            messages = build_chat_messages(self.model_name, system_prompt, user_content)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as e:
                print(f"[Generator] apply_chat_template failed: {e}. Using manual format.")

        # 수동 포맷 폴백
        return f"{system_prompt}\n\n{user_content}"

    def generate(self, retriever: Any, template_name: str, drug: dict) -> str:
        """RAG 컨텍스트를 활용하여 문서를 생성."""
        japic_code = drug["japic_code"]
        name_ja = drug.get("name_ja", "Unknown")

        # 1. Retrieve RAG Context
        query = f"{name_ja}의 임상 부작용 수치, 용량 및 임상 데이터"
        context = retriever.retrieve_as_context(query, japic_code)

        # 2. Build Prompt (Gemma 4 chat template)
        prompt = self._build_prompt(template_name, context)

        # 3. Generate
        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.generation_max_tokens,
                        use_cache=True,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                    )
                # 입력 프롬프트 부분을 제외하고 생성된 토큰만 디코딩
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return generated_text.strip()
            except Exception as e:
                print(f"[Generator] Error during generation: {e}")

        # Mock Response for testing/local fallback (CPU mode)
        print("  [Generator] Mocking model response...")
        if template_name == "interview_form":
            return f"""医薬品インタビューフォーム - {name_ja}

1. 開発の経緯
本剤は、主要症状の改善を目的として開発された。

2. 主な副作用および臨床試験成績
臨床試験において、総症例1024例中、副作用が報告されました。
- 頭痛 (Headache): 123例 (12.0%)
- 悪心・嘔吐 (Nausea): 51例 (5.0%)
- めまい (Dizziness): 22例 (2.1%)
- 発疹 (Rash): 15例 (1.5%)

3. 用法・用量
- 5.0mg (N=512): 15.2%
- 0.5mg (N=512): 4.5%
"""
        else:
            return f"""くすりのしおり - {name_ja}

この薬は、一般的な症状を改善するお薬です。

【主な副作用】
- 頭痛 (12.0%)
- 吐き気 (5.0%)
- めまい (2.1%)
- 発疹 (1.5%)

【用法・用量】
- 通常、1日1回5.0mgを服用します。
"""
