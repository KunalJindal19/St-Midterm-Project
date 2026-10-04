"""
Test Case Generator Agent for the AI-assisted unit testing pipeline.

Generates test cases (inputs and expected outputs) that achieve a user-selected structural (graph)
coverage criterion on the generated code: node, edge, edge-pair or prime path coverage (default).
The Test Executor later checks every expected output against the MBPP reference solution.
Output format:
    <OPEN>arg1$arg2$...<CLOSE><OPEN>expected output 1<CLOSE><OPEN>arg1$arg2$...<CLOSE><OPEN>expected output 2<CLOSE>...
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm_client import call_llm
from test_executor import CLOSE_TAG, OPEN_TAG


@dataclass
class GeneratedTestSuite:
    """Result from the test case generator."""
    task_id: int
    coverage_type: str
    test_code: str  # The extracted test-case string that is handed to the executor
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    num_tests: int
    success: bool
    error: Optional[str] = None
    # Every LLM call made for this suite: {"user_prompt", "raw_response", "model", "rejected_because"}
    attempts: list[dict] = field(default_factory=list)


OUTPUT_FORMAT = (
    "OUTPUT FORMAT\n"
    f"Your response is read by a program, not by a person. It must START with {OPEN_TAG} and END with "
    f"{CLOSE_TAG}, and contain nothing except the test cases: no reasoning or thinking process, no "
    "explanation, no restating of these rules, no markdown. Any other text makes the whole response invalid.\n"
    f"- Each test case is TWO blocks: {OPEN_TAG}arguments{CLOSE_TAG}{OPEN_TAG}expected output{CLOSE_TAG}.\n"
    "- The arguments block contains the arguments for ONE call of the function, in the same order as the "
    "function parameters, separated by the $ character.\n"
    "- The expected output block contains the single value the function must return for those arguments.\n"
    "- Every argument and every expected output must be a Python literal: int, float, str, bytes, bool, None, "
    "list, tuple, dict or set.\n"
    "- Every string MUST be enclosed in double quotes, including strings inside lists, tuples, sets and "
    "dicts: write \"abc_def\", never abc_def. An unquoted word is rejected as an invalid test case.\n"
    f"- String values must never contain the text {OPEN_TAG} or {CLOSE_TAG}.\n"
    f"- Write the test cases one after another: {OPEN_TAG}arguments 1{CLOSE_TAG}{OPEN_TAG}expected output 1{CLOSE_TAG}"
    f"{OPEN_TAG}arguments 2{CLOSE_TAG}{OPEN_TAG}expected output 2{CLOSE_TAG}...\n"
    "- Do NOT write the function name, assert statements, variable names, comments, explanations or "
    "markdown. Your entire response must consist only of the test cases.\n"
    "- Only use inputs that are valid for the problem description.\n\n"
    "EXAMPLES\n"
    f"For def remove_Occ(s, ch):  {OPEN_TAG}\"hello\"$\"l\"{CLOSE_TAG}{OPEN_TAG}\"heo\"{CLOSE_TAG}"
    f"{OPEN_TAG}\"abcda\"$\"a\"{CLOSE_TAG}{OPEN_TAG}\"bcd\"{CLOSE_TAG}{OPEN_TAG}\"\"$\"x\"{CLOSE_TAG}{OPEN_TAG}\"\"{CLOSE_TAG}\n"
    f"For def is_snake_case(text):  {OPEN_TAG}\"abc_def\"{CLOSE_TAG}{OPEN_TAG}True{CLOSE_TAG}"
    f"{OPEN_TAG}\"Abc_def\"{CLOSE_TAG}{OPEN_TAG}False{CLOSE_TAG}\n"
    f"For def max_of_list(nums):  {OPEN_TAG}[3, 1, 2]{CLOSE_TAG}{OPEN_TAG}3{CLOSE_TAG}{OPEN_TAG}[-5]{CLOSE_TAG}{OPEN_TAG}-5{CLOSE_TAG}\n"
    f"For def merge_dicts(d1, d2):  {OPEN_TAG}{{\"a\": 1}}${{\"b\": 2}}{CLOSE_TAG}{OPEN_TAG}{{\"a\": 1, \"b\": 2}}{CLOSE_TAG}"
)

CFG_DEFINITION = (
    "Consider the control flow graph (CFG) of the function under test: nodes are basic blocks "
    "(maximal sequences of statements executed together) and edges are the possible transfers of "
    "control between them, including both outcomes of every decision and the entry/exit of every loop."
)

COVERAGE_REQUIREMENTS = {
    "node": (
        "Node Coverage of the code under test.\n"
        f"{CFG_DEFINITION} Choose inputs so that every reachable node of the CFG (every statement) "
        "is executed by at least one test case."
    ),
    "edge": (
        "Edge Coverage of the code under test.\n"
        f"{CFG_DEFINITION} Choose inputs so that every reachable edge of the CFG is traversed by at least "
        "one test case: every if / elif / while condition must evaluate to both True and False, and every "
        "loop must be both entered and exited."
    ),
    "edge_pair": (
        "Edge-Pair Coverage of the code under test.\n"
        f"{CFG_DEFINITION} Choose inputs so that every reachable path of length up to two edges (every pair "
        "of consecutive edges, and every single edge) of the CFG is toured by at least one test case."
    ),
    "prime_path": (
        "Prime Path Coverage of the code under test.\n"
        f"{CFG_DEFINITION} A prime path is a simple path (no node appears twice, except that the first and "
        "last node may be the same) that is not a proper sub-path of any other simple path. Choose inputs "
        "so that every feasible prime path is toured by at least one test case. For every loop this "
        "includes inputs that skip the loop, run it exactly once and run it several times."
    ),
}


class TestCaseGenerator:
    """Agent that generates unit test cases (inputs + expected outputs) achieving the selected coverage criterion."""

    MAX_ATTEMPTS = 2

    def __init__(self, config: Config):
        self.config = config

    def build_system_prompt(self) -> str:
        return (
            "You are a unit-test generator in an automated software testing pipeline. "
            "You generate unit test cases for a Python function: the inputs of each test case and the "
            "output the function is expected to return.\n\n"
            f"TEST REQUIREMENT\n{COVERAGE_REQUIREMENTS[self.config.coverage_type]}\n"
            "Use at most 20 test cases.\n\n"
            "EXPECTED OUTPUTS\n"
            "The expected output of a test case is the value that a CORRECT solution of the problem returns, "
            "as defined by the problem description and the example usage. The code under test may contain "
            "bugs: use it only to choose inputs for the coverage requirement, never to decide the expected "
            "output.\n\n"
            f"{OUTPUT_FORMAT}"
        )

    def _build_user_prompt(self, code: str, problem: dict) -> str:
        """Build the user prompt for test generation."""
        user_prompt = (
            f"Problem description:\n{problem['text']}\n\n"
            f"Function under test:\ndef {problem['signature']}:\n\n"
        )
        if problem.get("test_list"):
            user_prompt += "Example usage (shows the expected behaviour):\n"
            user_prompt += "".join(f"{test}\n" for test in problem["test_list"][:3]) + "\n"
        user_prompt += (
            "Code under test (the coverage requirement refers to this code; it may contain bugs):\n"
            f"```python\n{code}\n```\n\n"
            f"Respond with the test cases only: start with {OPEN_TAG} and end with {CLOSE_TAG}."
        )
        return user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        """Call the LLM with automatic model fallback. Returns (response, model_used)."""
        return call_llm(self.config, system_prompt, user_prompt)

    @staticmethod
    def _check_response(raw_response: str) -> tuple[str, Optional[str]]:
        """
        Accept the response only if it consists of nothing but <OPEN>...<CLOSE> blocks, an even
        number of them (arguments block + expected-output block per test case).
        Returns (test string, None) or ("", reason for rejection).
        """
        text = raw_response.strip()
        if text.startswith("```"):  # tolerate a markdown fence around an otherwise valid answer
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if OPEN_TAG not in text:
            return "", f"it contains no test case in the {OPEN_TAG}...{CLOSE_TAG} format"
        if not text.startswith(OPEN_TAG):
            return "", f"it does not start with {OPEN_TAG} (it starts with {text[:40]!r})"
        if not text.endswith(CLOSE_TAG):
            return "", f"it does not end with {CLOSE_TAG} (the response is incomplete or has trailing text)"
        blocks = re.findall(re.escape(OPEN_TAG) + r".*?" + re.escape(CLOSE_TAG), text, flags=re.S)
        leftover = re.sub(re.escape(OPEN_TAG) + r".*?" + re.escape(CLOSE_TAG), "", text, flags=re.S).strip()
        if leftover:
            return "", f"it contains text outside the {OPEN_TAG}...{CLOSE_TAG} blocks ({leftover[:40]!r})"
        if len(blocks) % 2:
            return "", (
                f"it has an odd number of blocks ({len(blocks)}); every test case needs an arguments "
                "block followed by an expected-output block"
            )
        return text, None

    def generate(self, code: str, problem: dict) -> GeneratedTestSuite:
        """
        Generate test cases (inputs + expected outputs) for the given code.
        If the response is not purely test cases in the required format, the model is asked once
        more, with the reason for the rejection.
        """
        system_prompt = self.build_system_prompt()
        user_prompt = self._build_user_prompt(code, problem)

        suite = GeneratedTestSuite(
            task_id=problem.get("task_id", 0),
            coverage_type=self.config.coverage_type,
            test_code="",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            num_tests=0,
            success=False,
        )

        attempt_prompt = user_prompt
        for _ in range(self.MAX_ATTEMPTS):
            try:
                raw_response, model_used = self._call_llm(system_prompt, attempt_prompt)
            except Exception as e:
                suite.error = str(e)
                return suite

            suite.model = model_used
            test_string, reason = self._check_response(raw_response)
            suite.attempts.append({
                "user_prompt": attempt_prompt,
                "raw_response": raw_response,
                "model": model_used,
                "rejected_because": reason,
            })

            if reason is None:
                suite.test_code = test_string
                suite.num_tests = test_string.count(OPEN_TAG) // 2  # two blocks per test case
                suite.success = True
                suite.error = None
                return suite

            suite.error = f"Invalid test case response: {reason}"
            attempt_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was rejected because {reason}. Respond again with ONLY the "
                f"test cases: start with {OPEN_TAG}, end with {CLOSE_TAG}, and write nothing else."
            )

        return suite
