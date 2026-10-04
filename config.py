"""
Configuration management for the AI-assisted unit testing pipeline.
Handles API keys, model settings, and pipeline parameters.
"""

import os
from dataclasses import dataclass


# Structural (graph) coverage criteria the test cases must achieve; prime path is the default
COVERAGE_TYPES = ("node", "edge", "edge_pair", "prime_path")

COVERAGE_NAMES = {
    "node": "Node Coverage",
    "edge": "Edge Coverage",
    "edge_pair": "Edge-Pair Coverage",
    "prime_path": "Prime Path Coverage",
}


@dataclass
class Config:
    """Pipeline configuration."""

    # OpenRouter API settings
    api_key: str = ""
    api_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "cohere/north-mini-code:free"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Fallback models to try if the primary model fails (rate-limited, unavailable, etc.)
    fallback_models: list = None  # Set in __post_init__

    # Pipeline settings
    num_problems: int = 1
    max_problems: int = 50
    timeout_seconds: int = 10  # TLE threshold for a single test case

    # Structural coverage criterion the generated test cases must achieve
    coverage_type: str = "prime_path"

    # MBPP dataset settings
    dataset_name: str = "mbpp"
    dataset_split: str = "test"

    # Output settings
    output_dir: str = "output"

    def __post_init__(self):
        """Load API key from environment or .env file if not set."""
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not self.api_key:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break

        if self.fallback_models is None:
            self.fallback_models = [
                "cohere/north-mini-code:free",
                "google/gemma-4-31b-it:free",
                "google/gemma-4-26b-a4b-it:free",
                "qwen/qwen3.8-27b:free",
                "nvidia/nemotron-3.5-lightning:free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
            ]

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.api_key:
            errors.append(
                "OPENROUTER_API_KEY not set. "
                "Set it via .env file, environment variable, or --api-key CLI argument."
            )
        if self.num_problems < 1 or self.num_problems > self.max_problems:
            errors.append(f"num_problems must be between 1 and {self.max_problems}.")
        if self.coverage_type not in COVERAGE_TYPES:
            errors.append(
                f"Invalid coverage criterion '{self.coverage_type}'. "
                f"Must be one of: {', '.join(COVERAGE_TYPES)}."
            )
        return errors

    def coverage_label(self) -> str:
        """Human-readable name of the selected coverage criterion."""
        return COVERAGE_NAMES[self.coverage_type]

    def as_dict(self) -> dict:
        """Settings recorded in the output (the API key is never written)."""
        return {
            "model": self.model,
            "fallback_models": self.fallback_models,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "coverage_criterion": self.coverage_label(),
            "num_problems": self.num_problems,
            "timeout_seconds_per_test": self.timeout_seconds,
            "dataset": f"{self.dataset_name} ({self.dataset_split} split)",
        }
