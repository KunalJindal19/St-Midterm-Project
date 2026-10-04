"""
Pipeline Driver — orchestrates the Code Generator, Test Case Generator, and Test Executor.
Loads the MBPP dataset, runs the full pipeline, and collects results.
"""

import ast
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from code_generator import CodeGenerator, GeneratedCode
from test_case_generator import TestCaseGenerator, GeneratedTestSuite
from test_executor import TestExecutor, ExecutionResult, Verdict, format_coverage


@dataclass
class PipelineResult:
    """Result for a single MBPP problem through the full pipeline."""
    task_id: int
    problem_text: str
    reference_code: str
    function_name: str = ""
    signature: str = ""
    error: Optional[str] = None  # Set when the pipeline stopped before executing the tests

    generated_code: Optional[GeneratedCode] = None
    test_suite: Optional[GeneratedTestSuite] = None
    execution: Optional[ExecutionResult] = None


@dataclass
class PipelineSummary:
    """Summary of the entire pipeline run."""
    config: dict
    total_problems: int
    counts: dict
    total_time: float = 0.0
    results: list[PipelineResult] = field(default_factory=list)


def to_json(obj, indent: int = 0) -> str:
    """
    Pretty-print JSON like json.dumps(indent=2), but keep short lists (paths such as [1, 3, 4, 6],
    parsed arguments, lists of paths) on a single line so the results stay readable.
    """
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{inner}{json.dumps(str(k))}: {to_json(v, indent + 2)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        flat = all(
            not isinstance(x, (dict, list)) or (isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x))
            for x in obj
        )
        if flat:
            return json.dumps(obj)
        return "[\n" + ",\n".join(inner + to_json(x, indent + 2) for x in obj) + "\n" + pad + "]"
    return json.dumps(obj)


def resolve_function(problem: dict) -> tuple[str, str]:
    """
    Find the function under test and its signature from the reference solution.
    The name is the function called in the dataset's assert statements.
    """
    tree = ast.parse(problem["code"])
    functions = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError("the reference solution defines no top-level function")

    called = set(re.findall(r"([A-Za-z_]\w*)\s*\(", " ".join(problem.get("test_list", []))))
    candidates = [f for f in functions if f.name in called]
    node = candidates[0] if candidates else functions[-1]
    return node.name, f"{node.name}({ast.unparse(node.args)})"


class Pipeline:
    """
    Main driver that orchestrates the three agents:
      1. Code Generator — generates a Python function from an MBPP problem
      2. Test Case Generator — generates test cases (inputs + expected outputs) achieving the
         selected coverage criterion
      3. Test Executor — parses the tests, validates each expected output with the reference
         solution, calls the generated function on every valid test case and asserts the result
    """

    def __init__(self, config: Config, progress_callback=None):
        """
        Args:
            config: Pipeline configuration.
            progress_callback: Optional callable(step: str, task_id: int, detail: str)
                               for reporting progress to the CLI.
        """
        self.config = config
        self.code_gen = CodeGenerator(config)
        self.test_gen = TestCaseGenerator(config)
        self.executor = TestExecutor(config)
        self.progress = progress_callback or (lambda *a, **kw: None)

    def load_dataset(self) -> list[dict]:
        """Load the MBPP dataset and return the requested number of problems."""
        from datasets import load_dataset

        self.progress("dataset", 0, "Loading MBPP dataset...")
        ds = load_dataset("google-research-datasets/mbpp", "full", split=self.config.dataset_split)

        problems = []
        for i, item in enumerate(ds):
            if i >= self.config.num_problems:
                break
            problems.append({
                "task_id": item["task_id"],
                "text": item["text"],
                "code": item["code"].replace("\r\n", "\n"),
                "test_list": item["test_list"],
                "test_setup_code": item.get("test_setup_code", "") or "",
            })

        self.progress("dataset", 0, f"Loaded {len(problems)} problem(s) from MBPP dataset.")
        return problems

    def run_single(self, problem: dict) -> PipelineResult:
        """Run the full pipeline for a single MBPP problem."""
        task_id = problem["task_id"]
        result = PipelineResult(task_id=task_id, problem_text=problem["text"], reference_code=problem["code"])

        try:
            result.function_name, result.signature = resolve_function(problem)
        except (SyntaxError, ValueError) as e:
            result.error = f"Cannot use the reference solution: {e}"
            self.progress("pipeline", task_id, result.error)
            return result
        problem = dict(problem, function_name=result.function_name, signature=result.signature)

        # Step 1: Generate code
        self.progress("code_gen", task_id, "Generating code...")
        result.generated_code = self.code_gen.generate(problem)
        if not result.generated_code.success:
            result.error = f"Code generation failed: {result.generated_code.error}"
            self.progress("code_gen", task_id, result.error)
            return result
        self.progress("code_gen", task_id, "Code generated successfully.")

        # Step 2: Generate test cases
        self.progress("test_gen", task_id, f"Generating tests ({self.config.coverage_label()})...")
        result.test_suite = self.test_gen.generate(result.generated_code.generated_code, problem)
        if not result.test_suite.success:
            result.error = f"Test generation failed: {result.test_suite.error}"
            self.progress("test_gen", task_id, result.error)
            return result
        self.progress("test_gen", task_id, f"Generated {result.test_suite.num_tests} test case(s).")

        # Step 3: Execute tests
        self.progress("executor", task_id, "Executing tests...")
        result.execution = self.executor.execute(
            task_id=task_id,
            function_name=result.function_name,
            generated_code=result.generated_code.generated_code,
            reference_code=problem["code"],
            test_string=result.test_suite.test_code,
            setup_code=problem.get("test_setup_code", ""),
        )
        e = result.execution
        self.progress(
            "executor", task_id,
            f"{e.verdict.value} ({e.passed}/{e.valid_tests} valid tests passed, {e.invalid} invalid)",
        )
        return result

    def run(self) -> PipelineSummary:
        """Run the full pipeline for all problems."""
        start_time = time.time()
        problems = self.load_dataset()

        results = []
        for i, problem in enumerate(problems):
            self.progress("pipeline", problem["task_id"], f"Processing problem {i + 1}/{len(problems)}...")
            results.append(self.run_single(problem))

        summary = PipelineSummary(
            config=self.config.as_dict(),
            total_problems=len(problems),
            counts=self._count(results),
            results=results,
        )
        summary.total_time = time.time() - start_time
        self.progress("pipeline", 0, f"Pipeline completed in {summary.total_time:.1f}s")
        return summary

    @staticmethod
    def _count(results: list[PipelineResult]) -> dict:
        counts = {
            "code_generation_failed": 0,
            "test_generation_failed": 0,
            "verdicts": {v.value: 0 for v in Verdict},
            "coverage_criterion_met": 0,      # problems whose test cases achieve the selected criterion
            "coverage_criterion_not_met": 0,
            "test_cases": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "timeouts": 0, "invalid": 0},
        }
        for r in results:
            if r.generated_code is None or not r.generated_code.success:
                counts["code_generation_failed"] += 1
            elif r.test_suite is None or not r.test_suite.success:
                counts["test_generation_failed"] += 1
            if r.execution:
                counts["verdicts"][r.execution.verdict.value] += 1
                cov = r.execution.coverage
                if cov and "criterion_met" in cov:
                    key = "coverage_criterion_met" if cov["criterion_met"] else "coverage_criterion_not_met"
                    counts[key] += 1
                for k, v in r.execution.counts().items():
                    counts["test_cases"][k] += v
        return counts

    @staticmethod
    def _clear_previous_results(out_dir: str):
        """Remove only files written by a previous pipeline run, never anything else."""
        for name in os.listdir(out_dir):
            path = os.path.join(out_dir, name)
            if name == "summary.json" and os.path.isfile(path):
                os.remove(path)
            elif name.startswith("task_") and os.path.isdir(path):
                shutil.rmtree(path)

    def save_results(self, summary: PipelineSummary):
        """Save pipeline results to the output folder."""
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        self._clear_previous_results(out_dir)

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({
                "config": summary.config,
                "total_problems": summary.total_problems,
                "counts": summary.counts,
                "total_time_seconds": round(summary.total_time, 1),
            }, f, indent=2)

        for result in summary.results:
            self._save_task(result, os.path.join(out_dir, f"task_{result.task_id}"))

        self.progress("save", 0, f"Results saved to {out_dir}/")

    def _save_task(self, result: PipelineResult, task_dir: str):
        os.makedirs(task_dir, exist_ok=True)

        with open(os.path.join(task_dir, "reference_code.py"), "w") as f:
            f.write(result.reference_code)

        gen = result.generated_code
        if gen:
            if gen.success:
                with open(os.path.join(task_dir, "generated_code.py"), "w") as f:
                    f.write(gen.generated_code + "\n")
            with open(os.path.join(task_dir, "code_gen_prompts.json"), "w") as f:
                json.dump({
                    "system_prompt": gen.system_prompt,
                    "user_prompt": gen.user_prompt,
                    "model": gen.model,
                    "temperature": gen.temperature,
                    "max_tokens": self.config.max_tokens,
                    "attempts": gen.attempts,
                }, f, indent=2)

        suite = result.test_suite
        if suite:
            if suite.success:
                with open(os.path.join(task_dir, "test_cases.txt"), "w") as f:
                    f.write(suite.test_code + "\n")
            with open(os.path.join(task_dir, "test_gen_prompts.json"), "w") as f:
                json.dump({
                    "system_prompt": suite.system_prompt,
                    "user_prompt": suite.user_prompt,
                    "model": suite.model,
                    "temperature": suite.temperature,
                    "max_tokens": self.config.max_tokens,
                    "attempts": suite.attempts,
                }, f, indent=2)

        data = {
            "task_id": result.task_id,
            "function": result.signature or None,
            "coverage_criterion": self.config.coverage_label(),
        }
        e = result.execution
        if e is None:
            data["verdict"] = "NOT RUN"
            data["error"] = result.error
        else:
            data["verdict"] = e.verdict.value
            if e.error_message:
                data["error"] = e.error_message
            data["summary"] = e.counts()
            data["coverage"] = format_coverage(e.coverage)
            data["test_cases"] = [t.to_dict() for t in e.test_results]

        with open(os.path.join(task_dir, "execution_results.json"), "w") as f:
            f.write(to_json(data) + "\n")
