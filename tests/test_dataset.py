"""Tests for ProblemManager and curriculum logic."""

import pytest
from aix_evolver.dataset import Problem, ProblemManager


def test_problem_plateau_detection():
    prob = Problem(id="TEST-1", category="trap", question="Q", expected="5")
    assert not prob.is_plateaued

    prob.history = [False, False]
    assert not prob.is_plateaued

    prob.history = [False, False, False]
    assert prob.is_plateaued

    prob.history.append(True)
    assert not prob.is_plateaued


def test_problem_manager_curated_load():
    manager = ProblemManager()
    assert len(manager.problems) >= 8
    batch = manager.get_batch(count=3)
    assert len(batch) == 3


def test_problem_manager_record_attempt():
    manager = ProblemManager()
    prob_id = list(manager.problems.keys())[0]
    manager.record_attempt(prob_id, success=True)
    assert manager.problems[prob_id].success_count == 1
    assert manager.problems[prob_id].total_attempts == 1
