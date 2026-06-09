"""GPU 메모리 관리 유틸리티 (Google Colab 환경 최적화)"""
import gc
import os
from typing import Optional


def free_gpu_memory():
    """GPU 메모리 캐시 해제 및 가비지 컬렉션 수행."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[GPU] Memory cache cleared.")
    except ImportError:
        pass


def get_gpu_memory_info() -> dict:
    """현재 GPU VRAM 사용량 정보를 반환."""
    info = {
        "available": False,
        "total_mb": 0,
        "used_mb": 0,
        "free_mb": 0,
        "utilization_pct": 0.0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total - reserved
            info.update({
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "total_mb": round(total / 1024**2),
                "used_mb": round(allocated / 1024**2),
                "free_mb": round(free / 1024**2),
                "utilization_pct": round(allocated / total * 100, 1),
            })
    except ImportError:
        pass
    return info


def print_gpu_status(label: str = ""):
    """GPU 상태를 콘솔에 출력."""
    info = get_gpu_memory_info()
    prefix = f"[GPU:{label}]" if label else "[GPU]"
    if info["available"]:
        print(f"{prefix} {info['device_name']} | "
              f"Used: {info['used_mb']}MB / {info['total_mb']}MB "
              f"({info['utilization_pct']}%) | Free: {info['free_mb']}MB")
    else:
        print(f"{prefix} No GPU available (CPU mode).")


def unload_model(model, tokenizer=None):
    """모델과 토크나이저를 메모리에서 명시적으로 해제."""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    free_gpu_memory()
    print("[GPU] Model unloaded and memory freed.")
