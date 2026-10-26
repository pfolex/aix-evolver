"""Tests for ElasticTrainer."""

import os
import tempfile
import pytest
from aix_evolver.trainer import ElasticTrainer


def test_dry_run_training_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ElasticTrainer(checkpoint_dir=tmpdir, dry_run=True)
        metrics = trainer.train_on_rollouts(
            problem_id="TEST-01",
            question="What is 2+2?",
            winning_rollout="2+2 is 4",
            losing_rollout="2+2 is 5",
            adapter_name="test_expert"
        )
        assert metrics.step == 1
        assert metrics.loss > 0.0
        assert metrics.adapter_name == "test_expert"

        ckpt = trainer.save_checkpoint("test_expert")
        assert os.path.exists(ckpt)
