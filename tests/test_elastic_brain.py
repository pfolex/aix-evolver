"""Tests for ElasticBrain plateau detection, dynamic modular growth, and pruning."""

import os
import tempfile
import pytest
from aix_evolver.elastic_brain import ElasticBrain


def test_elastic_brain_initial_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "brain.json")
        brain = ElasticBrain(state_file=state_file)

        assert brain.base_params == 1_540_000_000
        assert brain.total_active_parameters == 1_540_000_000
        assert brain.added_parameter_footprint_mb == 0.0


def test_plateau_detection_and_spawn():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "brain.json")
        brain = ElasticBrain(state_file=state_file)

        # Record stagnant negative rewards
        for _ in range(5):
            brain.record_performance("cognitive_trap", reward=-0.8)

        is_plateau, reason = brain.detect_plateau("cognitive_trap", window=4)
        assert is_plateau
        assert "Plateau detected" in reason

        # Spawn modular expert
        event = brain.spawn_modular_expert("cognitive_trap", reason, lora_rank=8)
        assert event.expert_name in brain.modular_experts
        assert brain.total_active_parameters > brain.base_params
        assert brain.added_parameter_footprint_mb > 0.0
        assert event.growth_percent < 1.0  # Should be < 1% parameter expansion


def test_expert_pruning():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "brain.json")
        brain = ElasticBrain(state_file=state_file)

        event = brain.spawn_modular_expert("math", "Test", lora_rank=8)
        expert = brain.modular_experts[event.expert_name]
        expert.usage_count = 10
        expert.accuracy_impact = 0.001  # Insignificant impact

        pruned = brain.prune_inactive_experts(min_usage=5, min_accuracy_gain=0.01)
        assert event.expert_name in pruned
        assert not expert.is_active
        assert brain.total_active_parameters == brain.base_params
