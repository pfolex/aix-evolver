"""
Multi-signal verification engine: Evaluates correctness, reflection intensity, repetition, and consensus.
"""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class VerificationScore:
    predicted_answer: str
    expected_answer: str
    is_correct: bool
    reflection_intensity: float
    repetition_ratio: float
    length_penalty: float
    reward: float


class ReasoningVerifier:
    """Evaluates the cognitive quality and factual correctness of reasoning rollouts."""

    REFLECTION_MARKERS = [
        r"\bwait\b",
        r"\bactually\b",
        r"\blet me re-?read\b",
        r"\blet me check\b",
        r"\bdouble-?check\b",
        r"\bhowever\b",
        r"\bon the other hand\b",
        r"\bthat would mean\b",
        r"\bcontradiction\b",
        r"\bcorrecting\b",
        r"\bmistake\b",
    ]

    def __init__(self, alpha_length: float = 0.001, beta_reflection: float = 0.2):
        self.alpha_length = alpha_length
        self.beta_reflection = beta_reflection

    def extract_answer(self, text: str) -> str:
        """Extracts the final answer with priority for LaTeX boxed format."""
        if not text:
            return ""

        # Priority 1: Check \boxed{...} LaTeX expression
        boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed_matches:
            ans = boxed_matches[-1].strip()
            # Clean currency, spaces, units
            clean = re.sub(r"[\$,\\%]", "", ans).strip()
            # Look for number or short word
            num = re.search(r"-?\d+(?:\.\d+)?", clean)
            if num:
                return num.group(0)
            return clean

        # Priority 2: Look for '#### <number>' or 'answer is <number>'
        hash_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if hash_match:
            return hash_match.group(1)

        ans_patterns = [
            r"(?i)(?:the answer is|equals?|total is|result is)\s*[:\$]?\s*(-?\d+(?:\.\d+)?)",
            r"(?i)(?:the answer is|it is)\s*[:\$]?\s*([a-zA-Z0-9\(\)]+)",
        ]
        for pat in ans_patterns:
            m = re.findall(pat, text)
            if m:
                return m[-1].strip()

        # Priority 3: Extract last number
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "")

        # Fallback to last line
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return lines[-1][:40] if lines else ""

    def compute_reflection_intensity(self, text: str) -> float:
        """Quantifies how actively the model tested and verified its assumptions."""
        if not text:
            return 0.0
        matches = 0
        for marker in self.REFLECTION_MARKERS:
            matches += len(re.findall(marker, text, re.IGNORECASE))
        # Normalized by length (per 100 words)
        words = len(text.split())
        if words == 0:
            return 0.0
        rate = (matches / words) * 100
        return min(rate / 3.0, 1.0)  # Max out at 1.0

    def compute_repetition_ratio(self, text: str, n_gram: int = 4) -> float:
        """Detects spinning loops or repetitive token loops."""
        words = text.lower().split()
        if len(words) < n_gram:
            return 0.0
        ngrams = [tuple(words[i:i + n_gram]) for i in range(len(words) - n_gram + 1)]
        counts = Counter(ngrams)
        duplicates = sum(count - 1 for count in counts.values() if count > 1)
        return duplicates / len(ngrams)

    def is_equivalent(self, pred: str, expected: str) -> bool:
        """Checks mathematical and semantic equivalence."""
        pred_clean = pred.strip().lower().replace(",", "")
        exp_clean = expected.strip().lower().replace(",", "")

        if pred_clean == exp_clean:
            return True

        # Try floating point equivalence
        try:
            return math.isclose(float(pred_clean), float(exp_clean), rel_tol=1e-3, abs_tol=1e-3)
        except (ValueError, TypeError):
            pass

        # String substring fallback for word problems e.g. "O(log n)" vs "O(log(n))"
        if exp_clean in pred_clean:
            return True

        return False

    def score_rollout(self, rollout_text: str, expected: str, total_tokens: int) -> VerificationScore:
        """Computes a multi-signal reward for a rollout."""
        pred = self.extract_answer(rollout_text)
        correct = self.is_equivalent(pred, expected)
        refl = self.compute_reflection_intensity(rollout_text)
        rep = self.compute_repetition_ratio(rollout_text)

        # Length penalty: penalizes unnecessarily verbose rollouts
        len_penalty = math.exp(-self.alpha_length * max(0, total_tokens - 200))

        # Reward formulation:
        # Base: +1.0 for correct, -1.0 for wrong
        # Bonus: reflection intensity (rewarding self-verification)
        # Penalty: repetitive looping
        if correct:
            reward = 1.0 + (self.beta_reflection * refl) - (1.5 * rep)
            reward *= len_penalty
        else:
            reward = -1.0 + (0.1 * refl) - (2.0 * rep)

        return VerificationScore(
            predicted_answer=pred,
            expected_answer=expected,
            is_correct=correct,
            reflection_intensity=refl,
            repetition_ratio=rep,
            length_penalty=len_penalty,
            reward=reward
        )

    def find_consensus(self, rollouts: List[Tuple[str, int]]) -> Tuple[Optional[str], float]:
        """Detects multi-path consensus across a group of rollouts (text, token_count)."""
        answers = [self.extract_answer(text) for text, _ in rollouts if text]
        if not answers:
            return None, 0.0

        counts = Counter(answers)
        most_common_ans, count = counts.most_common(1)[0]
        confidence = count / len(answers)
        return most_common_ans, confidence
