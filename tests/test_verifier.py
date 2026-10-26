"""Tests for ReasoningVerifier, reflection scoring, and consensus."""

import pytest
from aix_evolver.verifier import ReasoningVerifier


def test_extract_boxed_answer():
    verifier = ReasoningVerifier()
    text = "After careful calculations, the final cost is \\boxed{5} cents."
    assert verifier.extract_answer(text) == "5"

    text2 = "Therefore we have \\boxed{$64}."
    assert verifier.extract_answer(text2) == "64"


def test_extract_hash_answer():
    verifier = ReasoningVerifier()
    text = "Step 1: 10 + 8 = 18\n#### 18"
    assert verifier.extract_answer(text) == "18"


def test_reflection_intensity():
    verifier = ReasoningVerifier()
    plain = "The ball costs 5 cents because 1.10 - 1.00 is 0.10 and 0.10 divided by 2 is 0.05."
    assert verifier.compute_reflection_intensity(plain) == 0.0

    reflective = "Wait, let me double check that assumption. Actually, that would lead to a contradiction! Let me correct this mistake."
    intensity = verifier.compute_reflection_intensity(reflective)
    assert intensity > 0.4


def test_repetition_ratio():
    verifier = ReasoningVerifier()
    clean = "First we add two numbers, then we multiply the result by three."
    assert verifier.compute_repetition_ratio(clean) == 0.0

    loop = "let me check let me check let me check let me check let me check let me check"
    assert verifier.compute_repetition_ratio(loop) > 0.5


def test_consensus():
    verifier = ReasoningVerifier()
    rollouts = [
        ("The answer is 5.", 50),
        ("Let's see: \\boxed{5}", 60),
        ("I think it is 50.", 40),
        ("Total is \\boxed{5}", 55),
    ]
    ans, conf = verifier.find_consensus(rollouts)
    assert ans == "5"
    assert conf == 0.75
