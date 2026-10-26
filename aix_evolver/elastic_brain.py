"""
Elastic Brain Engine: Tracks parameter frugality, detects capacity plateaus,
and dynamically expands modular LoRA capacity on demand.
"""

from dataclasses import dataclass, field
import json
import os
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpansionEvent:
    timestamp: float
    trigger_reason: str
    category: str
    expert_name: str
    prior_params: int
    added_params: int
    new_total_params: int
    growth_percent: float


@dataclass
class ModularExpert:
    name: str
    target_category: str
    lora_rank: int
    params_count: int
    creation_time: float
    usage_count: int = 0
    accuracy_impact: float = 0.0
    is_active: bool = True


class ElasticBrain:
    """Manages the parameter economy of the model.

    Enforces minimum parameter footprint by default, detects representational
    plateaus, and dynamically spawns modular LoRA experts only when strictly necessary.
    """

    BASE_1_5B_PARAMS = 1_540_000_000  # Baseline 1.5B parameter count

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or os.path.expanduser("~/.cache/aix_evolver/elastic_brain.json")
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        self.base_params = self.BASE_1_5B_PARAMS
        self.modular_experts: Dict[str, ModularExpert] = {}
        self.expansion_history: List[ExpansionEvent] = []
        self.category_history: Dict[str, List[float]] = {}
        self._load_state()

    @property
    def total_active_parameters(self) -> int:
        added = sum(exp.params_count for exp in self.modular_experts.values() if exp.is_active)
        return self.base_params + added

    @property
    def added_parameter_footprint_mb(self) -> float:
        """Approximates memory in MB for added 16-bit LoRA adapter weights."""
        added_params = sum(exp.params_count for exp in self.modular_experts.values() if exp.is_active)
        # 2 bytes per FP16 parameter
        return (added_params * 2) / (1024 * 1024)

    def record_performance(self, category: str, reward: float):
        """Records category reward to track performance curves and detect plateaus."""
        if category not in self.category_history:
            self.category_history[category] = []
        self.category_history[category].append(reward)

    def detect_plateau(self, category: str, window: int = 4, threshold: float = 0.05) -> Tuple[bool, str]:
        """Detects whether the model's current parameter size has plateaued in a specific domain.

        A plateau occurs when:
        1. At least `window` attempts were evaluated.
        2. Average reward remains stagnant or negative (<= 0.0).
        3. Variance / improvement delta is less than `threshold`.
        """
        history = self.category_history.get(category, [])
        if len(history) < window:
            return False, "Insufficient history"

        recent = history[-window:]
        avg_reward = sum(recent) / len(recent)
        delta = max(recent) - min(recent)

        if avg_reward < 0.2 and delta < threshold:
            reason = (
                f"Plateau detected in '{category}': {window} recent rollouts stagnant "
                f"(avg_reward={avg_reward:.2f}, delta={delta:.2f} < {threshold})"
            )
            return True, reason

        return False, "Performance progressing normally"

    def spawn_modular_expert(
        self,
        category: str,
        reason: str,
        lora_rank: int = 8
    ) -> ExpansionEvent:
        """Dynamically expands neural capacity by spawning a modular LoRA expert for the struggling category."""
        expert_idx = len(self.modular_experts) + 1
        expert_name = f"expert_{category}_{expert_idx}"

        # Standard Qwen/DeepSeek 28-layer attention projection LoRA parameter math:
        # Rank r=8 across q, k, v, o projections (1536 dim): ~2.5M params (~5 MB)
        added_params = 28 * 4 * (1536 * lora_rank * 2)
        prior = self.total_active_parameters

        expert = ModularExpert(
            name=expert_name,
            target_category=category,
            lora_rank=lora_rank,
            params_count=added_params,
            creation_time=time.time(),
            is_active=True
        )
        self.modular_experts[expert_name] = expert

        growth_pct = (added_params / prior) * 100

        event = ExpansionEvent(
            timestamp=time.time(),
            trigger_reason=reason,
            category=category,
            expert_name=expert_name,
            prior_params=prior,
            added_params=added_params,
            new_total_params=prior + added_params,
            growth_percent=growth_pct
        )
        self.expansion_history.append(event)
        self._save_state()
        return event

    def prune_inactive_experts(self, min_usage: int = 5, min_accuracy_gain: float = 0.02) -> List[str]:
        """Prunes modular experts that failed to provide accuracy gains, shedding unnecessary weight."""
        pruned = []
        for name, expert in list(self.modular_experts.items()):
            if expert.is_active and expert.usage_count >= min_usage:
                if expert.accuracy_impact < min_accuracy_gain:
                    expert.is_active = False
                    pruned.append(name)
        if pruned:
            self._save_state()
        return pruned

    def _save_state(self):
        data = {
            "base_params": self.base_params,
            "experts": {
                name: {
                    "target_category": exp.target_category,
                    "lora_rank": exp.lora_rank,
                    "params_count": exp.params_count,
                    "creation_time": exp.creation_time,
                    "usage_count": exp.usage_count,
                    "accuracy_impact": exp.accuracy_impact,
                    "is_active": exp.is_active
                }
                for name, exp in self.modular_experts.items()
            },
            "expansion_count": len(self.expansion_history),
            "category_history": self.category_history
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.base_params = data.get("base_params", self.BASE_1_5B_PARAMS)
                    self.category_history = data.get("category_history", {})
                    for name, exp_data in data.get("experts", {}).items():
                        self.modular_experts[name] = ModularExpert(
                            name=name,
                            target_category=exp_data["target_category"],
                            lora_rank=exp_data["lora_rank"],
                            params_count=exp_data["params_count"],
                            creation_time=exp_data["creation_time"],
                            usage_count=exp_data.get("usage_count", 0),
                            accuracy_impact=exp_data.get("accuracy_impact", 0.0),
                            is_active=exp_data.get("is_active", True)
                        )
            except Exception:
                pass
