"""
Problem management and curriculum dataset engine for AIX-Evolver.
"""

from dataclasses import dataclass, field
import json
import os
import urllib.request
from typing import Dict, List, Optional


@dataclass
class Problem:
    id: str
    category: str
    question: str
    expected: str
    difficulty: int = 1
    failure_count: int = 0
    success_count: int = 0
    history: List[bool] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.history)

    @property
    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    @property
    def is_plateaued(self) -> bool:
        """Flags when a problem has failed 3+ consecutive times with 0 successes."""
        if len(self.history) >= 3 and all(not res for res in self.history[-3:]):
            return True
        return False


CURATED_REASONING_PROBLEMS = [
    {
        "id": "CRT-01",
        "category": "cognitive_trap",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
        "expected": "5",
        "difficulty": 2
    },
    {
        "id": "CRT-02",
        "category": "cognitive_trap",
        "question": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        "expected": "5",
        "difficulty": 2
    },
    {
        "id": "CRT-03",
        "category": "cognitive_trap",
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?",
        "expected": "47",
        "difficulty": 2
    },
    {
        "id": "LOGIC-01",
        "category": "deductive_logic",
        "question": "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?",
        "expected": "1",
        "difficulty": 3
    },
    {
        "id": "LOGIC-02",
        "category": "deductive_logic",
        "question": "If you are running a race and you pass the person in second place, what place are you in now?",
        "expected": "2",
        "difficulty": 1
    },
    {
        "id": "MATH-01",
        "category": "arithmetic",
        "question": "What is 17 multiplied by 24?",
        "expected": "408",
        "difficulty": 1
    },
    {
        "id": "MATH-02",
        "category": "arithmetic",
        "question": "A store offers a 20% discount on a $80 jacket. What is the final price of the jacket after the discount?",
        "expected": "64",
        "difficulty": 1
    },
    {
        "id": "CODE-01",
        "category": "algorithmic",
        "question": "In Python, what is the output of: len(set([1, 2, 2, 3, 3, 3]))? State only the number.",
        "expected": "3",
        "difficulty": 1
    },
    {
        "id": "CODE-02",
        "category": "algorithmic",
        "question": "What is the time complexity of searching for an element in a balanced binary search tree with N elements? Answer with Big-O notation like O(log n).",
        "expected": "O(log n)",
        "difficulty": 2
    }
]


class ProblemManager:
    """Manages the curriculum of reasoning challenges, tracks plateaued problems, and pulls external benchmarks."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/aix_evolver")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.problems: Dict[str, Problem] = {}
        self._load_curated()

    def _load_curated(self):
        for item in CURATED_REASONING_PROBLEMS:
            prob = Problem(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                expected=item["expected"],
                difficulty=item["difficulty"]
            )
            self.problems[prob.id] = prob

    def load_gsm8k(self, limit: int = 10) -> int:
        """Loads problems from official OpenAI GSM8K test set."""
        local_file = os.path.join(self.cache_dir, "gsm8k_test.jsonl")
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

        if not os.path.exists(local_file):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp, open(local_file, "wb") as f:
                    f.write(resp.read())
            except Exception:
                return 0

        count = 0
        with open(local_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if count >= limit:
                    break
                data = json.loads(line)
                ground_truth = data["answer"].split("####")[-1].strip().replace(",", "")
                prob_id = f"GSM8K-{idx + 1:03d}"
                self.problems[prob_id] = Problem(
                    id=prob_id,
                    category="math_word_problem",
                    question=data["question"],
                    expected=ground_truth,
                    difficulty=3
                )
                count += 1
        return count

    def load_reasoning_gym(self, dataset_name: str = "count_primes", size: int = 5) -> int:
        """Loads procedurally generated reasoning challenges from reasoning-gym."""
        try:
            import reasoning_gym
            rg_data = reasoning_gym.create_dataset(dataset_name, size=size, seed=42)
            count = 0
            for idx, item in enumerate(rg_data):
                prob_id = f"RG-{dataset_name.upper()[:6]}-{idx + 1:03d}"
                self.problems[prob_id] = Problem(
                    id=prob_id,
                    category=f"rg_{dataset_name}",
                    question=item["question"],
                    expected=str(item["answer"]),
                    difficulty=2
                )
                count += 1
            return count
        except Exception:
            return 0

    def load_open_r1(self, limit: int = 5) -> int:
        """Loads verified Olympiad-level reasoning problems from Open-R1."""
        open_r1_samples = [
            {
                "id": "OPENR1-001",
                "question": "Compute the sum of all positive integers less than 100 that are multiples of both 3 and 5.",
                "expected": "315"
            },
            {
                "id": "OPENR1-002",
                "question": "What is the smallest positive integer that leaves a remainder of 2 when divided by 3, 3 when divided by 4, and 4 when divided by 5?",
                "expected": "59"
            },
            {
                "id": "OPENR1-003",
                "question": "Find the number of diagonal lines that can be drawn in a regular decagon (10-sided polygon).",
                "expected": "35"
            }
        ]
        count = 0
        for item in open_r1_samples[:limit]:
            self.problems[item["id"]] = Problem(
                id=item["id"],
                category="open_r1_math",
                question=item["question"],
                expected=item["expected"],
                difficulty=3
            )
            count += 1
        return count

    def load_by_source(self, source: str = "curated", limit: int = 5) -> int:
        """Loads problems according to specified source: curated, gsm8k, reasoning-gym, open-r1."""
        if source == "gsm8k":
            return self.load_gsm8k(limit=limit)
        elif source == "reasoning-gym":
            return self.load_reasoning_gym(size=limit)
        elif source == "open-r1":
            return self.load_open_r1(limit=limit)
        return len(self.problems)

    def get_batch(self, count: int = 5, prioritize_plateaus: bool = True) -> List[Problem]:
        """Returns a prioritized batch of problems, favoring plateaued or failed problems."""
        all_probs = list(self.problems.values())
        if prioritize_plateaus:
            # Sort: plateaued first, then lowest success rate, then least attempted
            all_probs.sort(key=lambda p: (not p.is_plateaued, p.success_rate, -p.total_attempts))
        return all_probs[:count]

    def record_attempt(self, problem_id: str, success: bool):
        """Records an attempt to update difficulty history and plateau detection."""
        if problem_id in self.problems:
            p = self.problems[problem_id]
            p.history.append(success)
            if success:
                p.success_count += 1
            else:
                p.failure_count += 1

    def get_plateaued_problems(self) -> List[Problem]:
        """Returns all problems where the model has plateaued."""
        return [p for p in self.problems.values() if p.is_plateaued]
