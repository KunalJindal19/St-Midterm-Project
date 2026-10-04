"""
Code Generator Agent for the AI-assisted unit testing pipeline.
Generates a single Python function from a natural language problem description.
The Test Executor defines this function and calls it with the parsed test-case arguments.
"""

import ast
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm_client import call_llm


@dataclass
class GeneratedCode:
    """Result from the code generator."""
    task_id: int
    prompt: str
    generated_code: str
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    success: bool
    error: Optional[str] = None
    # Every LLM call made for this problem: {"user_prompt", "raw_response", "model", "rejected_because"}
    attempts: list[dict] = field(default_factory=list)


def validate_generated_code(code: str, function_name: str) -> Optional[str]:
    """
    Return None if the response is exactly one function definition named `function_name`,
    otherwise the reason it is rejected. Imports and helpers must be inside the function body.
    """
    if not code.strip():
        return "the response is empty"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"the response is not valid Python code (SyntaxError: {e.msg}, line {e.lineno})"

    if not tree.body:
        return "the response contains no function definition"
    extra = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if extra:
        return (
            f"line {extra[0].lineno} is a top-level `{type(extra[0]).__name__}` statement; the response "
            "must be a single function definition (put imports and helpers inside the function body)"
        )
    if len(tree.body) > 1:
        return (
            f"the response defines {len(tree.body)} top-level functions; it must be a single function "
            "(define any helper functions inside it)"
        )
    if tree.body[0].name != function_name:
        return f"the function is named `{tree.body[0].name}` but must be named `{function_name}`"
    return None


def _strip_fence(text: str) -> str:
    """Remove a markdown fence if the model wrapped its whole answer in one anyway."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class CodeGenerator:
    """
    Agent that generates Python code from MBPP problem descriptions.
    Uses an LLM via OpenRouter API to generate functional code.
    """

    MAX_ATTEMPTS = 2

    SYSTEM_PROMPT = (
        "You are a Python code generator in an automated unit-testing pipeline. "
        "Your response is loaded by a program that calls your function with test inputs, "
        "so it must be exactly one Python 3 function definition and nothing else.\n\n"
        "Rules:\n"
        "1. Write exactly one function, with exactly the name and parameters given in the user message.\n"
        "2. Put any import statements and helper functions INSIDE the function body. Nothing may appear "
        "before or after the function.\n"
        "3. The function must return its result. Do not print, do not read input(), and do not include "
        "example usage, test cases or assert statements.\n"
        "4. Do not write any text outside the function: no reasoning or thinking process, no explanations "
        "and no markdown code fences (```).\n"
        "5. The first line of your response must be `def <function name>(`."
    )

    def __init__(self, config: Config):
        self.config = config

    def _build_user_prompt(self, problem: dict) -> str:
        """Build the user prompt from an MBPP problem."""
        prompt_text = problem.get("text", problem.get("prompt", ""))
        test_list = problem.get("test_list", [])

        user_prompt = (
            f"Problem description:\n{prompt_text}\n\n"
            f"Required function signature:\ndef {problem['signature']}:\n"
        )

        if test_list:
            user_prompt += "\nExample usage (shows the expected behaviour):\n"
            for test in test_list[:3]:
                user_prompt += f"{test}\n"

        user_prompt += "\nWrite the function now."
        return user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        """Call the LLM with automatic model fallback. Returns (response, model_used)."""
        return call_llm(self.config, system_prompt, user_prompt)

    def generate(self, problem: dict) -> GeneratedCode:
        """
        Generate code for a given MBPP problem.

        The response must be exactly one function definition with the required name. If it is not
        (syntax error, anything outside the function, wrong name), the model is asked once more
        with the rejection reason appended to the prompt.
        """
        task_id = problem.get("task_id", 0)
        prompt_text = problem.get("text", problem.get("prompt", ""))
        function_name = problem["function_name"]
        user_prompt = self._build_user_prompt(problem)

        result = GeneratedCode(
            task_id=task_id,
            prompt=prompt_text,
            generated_code="",
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            success=False,
        )

        attempt_prompt = user_prompt
        for _ in range(self.MAX_ATTEMPTS):
            try:
                raw_response, model_used = self._call_llm(self.SYSTEM_PROMPT, attempt_prompt)
            except Exception as e:
                result.error = str(e)
                return result

            code = _strip_fence(raw_response)
            reason = validate_generated_code(code, function_name)
            result.attempts.append({
                "user_prompt": attempt_prompt,
                "raw_response": raw_response,
                "model": model_used,
                "markdown_fence_removed": code != raw_response.strip(),
                "rejected_because": reason,
            })
            result.model = model_used

            if reason is None:
                result.generated_code = code
                result.success = True
                result.error = None
                return result

            result.error = f"Invalid code: {reason}"
            attempt_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was rejected because {reason}. "
                "Respond again with only the function definition."
            )

        return result
