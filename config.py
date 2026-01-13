"""
config.py - Centralized configuration for the MCFP MoA scheduling system.

All configuration values can be overridden via environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from entities import ModelCard


@dataclass
class SchedulerConfig:
    """
    Configuration for the scheduling system.
    
    Scheduling intervals (per logic_devemopment_revised.md):
    - delta_t1: Request scheduling interval (default 1s)
    - delta_t2: GPU reconfiguration interval (default 5s)
    """
    
    # Scheduling intervals
    delta_t1: float = 1.0   # Request scheduling (seconds)
    delta_t2: float = 5.0   # GPU reconfiguration (seconds)
    
    # GPU settings
    alpha: float = 0.4      # Max VRAM fraction for slept weights
    beta: float = 1.0       # Wait cost multiplier
    
    # vLLM settings
    vllm_host: str = "127.0.0.1"
    vllm_port_base: int = 9000
    vllm_startup_timeout_s: float = 600.0
    vllm_request_timeout_s: float = 300.0
    vllm_extra_args: List[str] = field(default_factory=list)
    
    # Model card persistence
    model_card_path: str = "model_card.json"
    
    # Scheduler limits
    max_decisions_per_step: int = 1024
    timing_ema_alpha: float = 0.2
    
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        """
        Create configuration from environment variables.
        
        Environment variable names match field names in uppercase:
        - DELTA_T1, DELTA_T2
        - ALPHA, BETA
        - VLLM_HOST, VLLM_PORT_BASE, VLLM_STARTUP_TIMEOUT_S
        - MODEL_CARD_PATH
        - etc.
        """
        def get_float(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))
        
        def get_int(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))
        
        def get_str(name: str, default: str) -> str:
            return os.getenv(name, default)
        
        def get_list(name: str, default: List[str]) -> List[str]:
            val = os.getenv(name)
            if val:
                return val.split(",")
            return default
        
        return cls(
            delta_t1=get_float("DELTA_T1", 1.0),
            delta_t2=get_float("DELTA_T2", 5.0),
            alpha=get_float("ALPHA", 0.4),
            beta=get_float("BETA", 1.0),
            vllm_host=get_str("VLLM_HOST", "127.0.0.1"),
            vllm_port_base=get_int("VLLM_PORT_BASE", 9000),
            vllm_startup_timeout_s=get_float("VLLM_STARTUP_TIMEOUT_S", 600.0),
            vllm_request_timeout_s=get_float("VLLM_REQUEST_TIMEOUT_S", 300.0),
            vllm_extra_args=get_list("VLLM_EXTRA_ARGS", []),
            model_card_path=get_str("MODEL_CARD_PATH", "model_card.json"),
            max_decisions_per_step=get_int("MAX_DECISIONS_PER_STEP", 1024),
            timing_ema_alpha=get_float("TIMING_EMA_ALPHA", 0.2),
            server_host=get_str("SERVER_HOST", "0.0.0.0"),
            server_port=get_int("SERVER_PORT", 8000),
        )


# -----------------------------
# Model Card Persistence
# -----------------------------

def read_model_cards(path: str) -> Dict[str, ModelCard]:
    """
    Load model cards from JSON file.
    
    Expected format:
    {
        "model_id": {
            "tp_min": 1,
            "t_wake_s": 2.0,
            "t_sleep_s": 2.0,
            "t_load_s": 90.0,
            "t_offload_s": 3.0,
            "slept_mem_tp1_MB": 2048.0,
            "slept_mem_tpgt1_MB": 4096.0,
            "avg_service_s": 0.2
        },
        ...
    }
    """
    if not os.path.exists(path):
        return {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    
    cards: Dict[str, ModelCard] = {}
    for model_id, cfg in data.items():
        cards[model_id] = ModelCard.from_dict(model_id, cfg)
    
    return cards


def write_model_cards(path: str, cards: Dict[str, ModelCard]) -> None:
    """Persist model cards to JSON file."""
    data = {model_id: card.to_dict() for model_id, card in cards.items()}
    
    # Atomic write
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def ensure_model_card(
    cards: Dict[str, ModelCard],
    model_id: str,
    tp_min: int = 1,
    persist_path: Optional[str] = None,
) -> ModelCard:
    """
    Ensure model card exists, creating with defaults if needed.
    
    If persist_path is provided, saves updated cards to disk.
    """
    if model_id in cards:
        return cards[model_id]
    
    card = ModelCard(
        model_id=model_id,
        tp_min=tp_min,
        t_wake_s=2.0,
        t_sleep_s=2.0,
        t_load_s=90.0,
        t_offload_s=3.0,
        slept_mem_tp1_MB=2048.0,
        slept_mem_tpgt1_MB=4096.0,
        avg_service_s=0.2,
    )
    cards[model_id] = card
    
    if persist_path:
        write_model_cards(persist_path, cards)
    
    return card
