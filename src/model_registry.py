"""모델 레지스트리: 지원 모델별 프리셋 및 chat template 관리."""
from typing import Any, Dict, List, Optional


# 지원 모델 프리셋 정의
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "google/gemma-4-E4B-it": {
        "family": "gemma4",
        "display_name": "Gemma 4 E4B-IT",
        "params_billions": 4,
        "max_context_length": 131072,  # 128K
        "recommended_seq_length": 8192,
        "estimated_vram_4bit_mb": 4500,
        "estimated_vram_qlora_mb": 17000,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "supports_system_prompt": True,
        "chat_template_method": "apply_chat_template",
        "use_unsloth": True,
        "eos_token": "<end_of_turn>",
    },
    "google/gemma-2-27b-it": {
        "family": "gemma2",
        "display_name": "Gemma 2 27B-IT",
        "params_billions": 27,
        "max_context_length": 8192,
        "recommended_seq_length": 4096,
        "estimated_vram_4bit_mb": 16000,
        "estimated_vram_qlora_mb": 40000,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "supports_system_prompt": False,
        "chat_template_method": "manual",
        "use_unsloth": True,
        "eos_token": "<end_of_turn>",
    },
    "google/gemma-2-9b-it": {
        "family": "gemma2",
        "display_name": "Gemma 2 9B-IT",
        "params_billions": 9,
        "max_context_length": 8192,
        "recommended_seq_length": 4096,
        "estimated_vram_4bit_mb": 6000,
        "estimated_vram_qlora_mb": 20000,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "supports_system_prompt": False,
        "chat_template_method": "manual",
        "use_unsloth": True,
        "eos_token": "<end_of_turn>",
    },
}


def get_model_preset(model_name: str) -> Dict[str, Any]:
    """모델명에 해당하는 프리셋을 반환. 미등록 모델은 기본값 제공."""
    if model_name in MODEL_PRESETS:
        return MODEL_PRESETS[model_name]

    # 미등록 모델에 대한 기본 프리셋
    print(f"[ModelRegistry] Warning: '{model_name}' is not registered. Using default preset.")
    return {
        "family": "unknown",
        "display_name": model_name,
        "params_billions": 0,
        "max_context_length": 4096,
        "recommended_seq_length": 4096,
        "estimated_vram_4bit_mb": 8000,
        "estimated_vram_qlora_mb": 24000,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "supports_system_prompt": False,
        "chat_template_method": "manual",
        "use_unsloth": False,
        "eos_token": "</s>",
    }


def get_lora_target_modules(model_name: str) -> List[str]:
    """모델에 대한 LoRA target modules 리스트를 반환."""
    preset = get_model_preset(model_name)
    return preset["lora_target_modules"]


def supports_system_prompt(model_name: str) -> bool:
    """모델이 네이티브 system prompt를 지원하는지 여부."""
    preset = get_model_preset(model_name)
    return preset["supports_system_prompt"]


def build_chat_messages(
    model_name: str,
    system_prompt: str,
    user_content: str,
) -> list[dict[str, str]]:
    """모델에 맞는 chat messages 리스트를 구성.

    Gemma 4는 네이티브 system role을 지원하므로 system 메시지를 별도 role로 전달.
    Gemma 2 이하는 system prompt를 user 메시지에 병합.
    """
    preset = get_model_preset(model_name)

    if preset["supports_system_prompt"]:
        # Gemma 4: system role 지원
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        # Gemma 2 이하: system을 user에 병합
        merged = f"{system_prompt}\n\n{user_content}"
        return [
            {"role": "user", "content": merged},
        ]


def build_sft_chat_text(
    model_name: str,
    system_prompt: str,
    user_content: str,
    assistant_response: str,
) -> list[dict[str, str]]:
    """SFT 학습용 chat messages를 구성 (assistant 응답 포함).

    tokenizer.apply_chat_template으로 텍스트 변환 시 사용.
    """
    preset = get_model_preset(model_name)

    if preset["supports_system_prompt"]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "model", "content": assistant_response},
        ]
    else:
        merged = f"{system_prompt}\n\n{user_content}"
        return [
            {"role": "user", "content": merged},
            {"role": "model", "content": assistant_response},
        ]
