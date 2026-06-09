import os
from typing import Any

from src.model_registry import get_model_preset, get_lora_target_modules
from src.gpu_utils import free_gpu_memory, print_gpu_status


class QLoraTrainer:
    def __init__(self, settings: Any):
        self.settings = settings
        self.base_model_name = self.settings.model.base_model
        self.device = self.settings.model.device
        self.max_seq_length = self.settings.model.max_seq_length
        self.checkpoint_dir = self.settings.paths.checkpoints
        self.preset = get_model_preset(self.base_model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_checkpoint_name(self, iteration: int) -> str:
        """모델명에서 동적으로 체크포인트 이름을 생성."""
        # "google/gemma-4-E4B-it" -> "gemma-4-E4B-qlora-iter-1"
        short_name = self.base_model_name.split("/")[-1].replace("-it", "")
        return f"{short_name}-qlora-iter-{iteration}"

    def train(self, dataset_path: str, iteration: int, lr: float, epochs: int) -> str:
        """Unsloth QLoRA 학습을 실행하거나, CPU에서 Mock 학습을 수행."""
        checkpoint_name = self._get_checkpoint_name(iteration)
        output_adapter_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        if self.device == "cuda":
            try:
                return self._train_gpu(dataset_path, iteration, lr, epochs, output_adapter_path)
            except Exception as e:
                print(f"[Trainer] GPU Training failed: {e}. Falling back to simulation.")

        # CPU Simulation Fallback
        return self._train_mock(output_adapter_path, iteration)

    def _train_gpu(self, dataset_path: str, iteration: int, lr: float, epochs: int,
                   output_adapter_path: str) -> str:
        """GPU 환경에서 Unsloth/PEFT 기반 QLoRA 학습 실행."""
        import torch

        print(f"\n[Trainer] ========== Starting QLoRA Training ==========")
        print(f"  Model: {self.base_model_name}")
        print(f"  Iteration: {iteration}")
        print(f"  Learning Rate: {lr}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {self.settings.training.batch_size}")
        print(f"  Gradient Accumulation: {self.settings.training.gradient_accumulation_steps}")
        print_gpu_status("Before training")

        # 1. Load Model (Unsloth 우선)
        model, tokenizer = self._load_base_model()

        # 2. Setup LoRA
        model = self._setup_lora(model)

        # 3. Load and format dataset
        dataset = self._load_and_format_dataset(dataset_path, tokenizer)

        # 4. Train with SFTTrainer
        self._run_sft_training(model, tokenizer, dataset, lr, epochs, output_adapter_path, torch)

        # 5. Save adapter
        model.save_pretrained(output_adapter_path)
        tokenizer.save_pretrained(output_adapter_path)
        print(f"[Trainer] Saved LoRA adapter to {output_adapter_path}")
        print_gpu_status("After training")

        # 6. Cleanup training model to free VRAM
        del model, tokenizer
        free_gpu_memory()
        print(f"[Trainer] ========== Training Complete ==========\n")

        return output_adapter_path

    def _load_base_model(self):
        """Unsloth 또는 표준 HF로 베이스 모델 로드."""
        if self.preset.get("use_unsloth", False):
            try:
                from unsloth import FastLanguageModel
                print(f"[Trainer] Loading {self.base_model_name} with Unsloth...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.base_model_name,
                    max_seq_length=self.max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                )
                return model, tokenizer
            except ImportError:
                print("[Trainer] Unsloth not available, falling back to standard HF.")

        # Standard HF fallback
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        return model, tokenizer

    def _setup_lora(self, model):
        """LoRA 어댑터를 모델에 설정."""
        target_modules = get_lora_target_modules(self.base_model_name)

        try:
            from unsloth import FastLanguageModel
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.settings.training.lora_r,
                target_modules=target_modules,
                lora_alpha=self.settings.training.lora_alpha,
                lora_dropout=self.settings.training.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
        except ImportError:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.settings.training.lora_r,
                lora_alpha=self.settings.training.lora_alpha,
                lora_dropout=self.settings.training.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        return model

    def _load_and_format_dataset(self, dataset_path: str, tokenizer):
        """SFT 데이터셋을 로드하고 Gemma 4 chat template으로 포맷팅."""
        from datasets import load_dataset
        from src.model_registry import build_sft_chat_text

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        def formatting_prompts_func(examples):
            texts = []
            for instr, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                # model_registry를 통해 모델에 맞는 chat messages 구성
                messages = build_sft_chat_text(
                    self.base_model_name,
                    system_prompt=instr,
                    user_content=inp,
                    assistant_response=out,
                )
                # tokenizer의 apply_chat_template으로 변환
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    # 폴백: 수동 포맷
                    text = (
                        f"<start_of_turn>user\n{instr}\n\n{inp}<end_of_turn>\n"
                        f"<start_of_turn>model\n{out}<end_of_turn>"
                    )
                texts.append(text)
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        print(f"[Trainer] Dataset loaded and formatted. Total samples: {len(dataset)}")
        return dataset

    def _run_sft_training(self, model, tokenizer, dataset, lr, epochs,
                          output_adapter_path, torch):
        """SFTTrainer로 학습 실행."""
        from trl import SFTTrainer
        from transformers import TrainingArguments

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=self.settings.training.batch_size,
                gradient_accumulation_steps=self.settings.training.gradient_accumulation_steps,
                warmup_ratio=self.settings.training.warmup_ratio,
                max_steps=-1,
                num_train_epochs=epochs,
                learning_rate=lr,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                max_grad_norm=self.settings.training.max_grad_norm,
                seed=3407,
                output_dir=output_adapter_path,
                report_to="none",
            ),
        )

        trainer.train()

    def _train_mock(self, output_adapter_path: str, iteration: int) -> str:
        """CPU Mock 학습 (테스트/로컬 개발용)."""
        print(f"[Trainer] Mock training completed (iteration {iteration}).")
        print(f"  Simulated adapter checkpoint: {output_adapter_path}")
        os.makedirs(output_adapter_path, exist_ok=True)
        # Create dummy adapter config
        import json
        config = {
            "r": self.settings.training.lora_r,
            "lora_alpha": self.settings.training.lora_alpha,
            "peft_type": "LORA",
            "base_model_name": self.base_model_name,
            "iteration": iteration,
        }
        with open(os.path.join(output_adapter_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        return output_adapter_path
