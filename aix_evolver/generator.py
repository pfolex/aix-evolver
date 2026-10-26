"""
Rollout generator interfacing with local Ollama runtime for multi-path reasoning exploration.
"""

from dataclasses import dataclass
import json
import time
import urllib.request
from typing import List, Optional


@dataclass
class RolloutResult:
    problem_id: str
    thinking: str
    content: str
    tokens: int
    tok_per_sec: float
    elapsed: float

    @property
    def full_text(self) -> str:
        if self.content:
            return self.content
        return self.thinking


class RolloutGenerator:
    """Generates multiple diverse reasoning traces from a local Ollama model."""

    def __init__(self, model_name: str = "aix", api_url: str = "http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.api_url = api_url

    def generate_single(
        self,
        problem_id: str,
        question: str,
        temperature: float = 0.6,
        num_predict: int = 1024,
        timeout: int = 120
    ) -> RolloutResult:
        """Generates a single reasoning rollout via streaming Ollama API."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": question}],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.api_url, data=data, headers={"Content-Type": "application/json"})

        full_thinking = ""
        full_content = ""
        thinking_tokens = 0
        content_tokens = 0
        t0 = time.time()

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                msg = chunk.get("message", {})
                t = msg.get("thinking", "")
                c = msg.get("content", "")
                if t:
                    full_thinking += t
                    thinking_tokens += 1
                if c:
                    full_content += c
                    content_tokens += 1

        elapsed = time.time() - t0
        total_tokens = thinking_tokens + content_tokens
        tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        return RolloutResult(
            problem_id=problem_id,
            thinking=full_thinking.strip(),
            content=full_content.strip(),
            tokens=total_tokens,
            tok_per_sec=tok_per_sec,
            elapsed=elapsed
        )

    def generate_group(
        self,
        problem_id: str,
        question: str,
        group_size: int = 3,
        temperature: float = 0.7,
        num_predict: int = 1024
    ) -> List[RolloutResult]:
        """Generates G independent reasoning traces for self-consistency and relative ranking."""
        rollouts = []
        for _ in range(group_size):
            try:
                res = self.generate_single(
                    problem_id=problem_id,
                    question=question,
                    temperature=temperature,
                    num_predict=num_predict
                )
                rollouts.append(res)
            except Exception as e:
                # Fallback empty rollout on network timeout
                rollouts.append(
                    RolloutResult(
                        problem_id=problem_id,
                        thinking=f"[Error during generation: {e}]",
                        content="",
                        tokens=0,
                        tok_per_sec=0.0,
                        elapsed=0.0
                    )
                )
        return rollouts
