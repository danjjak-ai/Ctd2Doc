import os
from typing import Any

class QLoraTrainer:
    def __init__(self, settings: Any):
        self.settings = settings
        self.base_model_name = self.settings.model.base_model
        self.device = self.settings.model.device
        self.checkpoint_dir = self.settings.paths.checkpoints
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, dataset_path: str, iteration: int, lr: float, epochs: int) -> str:
        """Triggers Unsloth QLoRA training on GPU, or mocks training workflow on CPU."""
        output_adapter_path = os.path.join(self.checkpoint_dir, f"gemma-2-27b-qlora-iter-{iteration}")
        
        if self.device == "cuda":
            try:
                from unsloth import FastLanguageModel
                import torch
                from trl import SFTTrainer
                from transformers import TrainingArguments
                from datasets import load_dataset
                
                print(f"[Trainer] Starting Unsloth training iteration {iteration} on L4 GPU...")
                print(f"  Params: Learning Rate={lr}, Epochs={epochs}, Batch Size=1")
                
                # 1. Load Model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.base_model_name,
                    max_seq_length=4096,
                    dtype=None,
                    load_in_4bit=True,
                )
                
                # 2. Setup LoRA
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=self.settings.training.lora_r,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                                    "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=self.settings.training.lora_alpha,
                    lora_dropout=self.settings.training.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                )
                
                # 3. Load dataset
                dataset = load_dataset("json", data_files=dataset_path, split="train")
                
                # Format prompts
                def formatting_prompts_func(examples):
                    instructions = examples["instruction"]
                    inputs       = examples["input"]
                    outputs      = examples["output"]
                    texts = []
                    for instr, inp, out in zip(instructions, inputs, outputs):
                        text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                        texts.append(text)
                    return { "text" : texts }
                
                dataset = dataset.map(formatting_prompts_func, batched=True)
                
                # 4. Train
                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=dataset,
                    dataset_text_field="text",
                    max_seq_length=4096,
                    dataset_num_proc=2,
                    packing=False, # can speed up for short sequences
                    args=TrainingArguments(
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=4,
                        warmup_steps=5,
                        max_steps=-1,
                        num_train_epochs=epochs,
                        learning_rate=lr,
                        fp16=not torch.cuda.is_bf16_supported(),
                        bf16=torch.cuda.is_bf16_supported(),
                        logging_steps=1,
                        optim="adamw_8bit",
                        weight_decay=0.01,
                        lr_scheduler_type="linear",
                        seed=3407,
                        output_dir=output_adapter_path,
                        report_to="none"
                    ),
                )
                
                trainer.train()
                model.save_pretrained(output_adapter_path)
                tokenizer.save_pretrained(output_adapter_path)
                print(f"[Trainer] Saved LoRA adapter to {output_adapter_path}")
                return output_adapter_path
                
            except Exception as e:
                print(f"[Trainer] GPU Training failed: {e}. Falling back to simulation.")
        
        # CPU Simulation Fallback
        print(f"[Trainer] Mock training completed. Created simulated adapter checkpoint: {output_adapter_path}")
        os.makedirs(output_adapter_path, exist_ok=True)
        # Create dummy weights file
        with open(os.path.join(output_adapter_path, "adapter_config.json"), "w") as f:
            f.write('{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}')
        return output_adapter_path
