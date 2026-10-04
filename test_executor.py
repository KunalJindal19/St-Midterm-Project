"""
Test Executor Agent for the AI-assisted unit testing pipeline.

For every test suite produced by the Test Case Generator it:
  1. Parses the test-case string
         <OPEN>arg1$arg2$...<CLOSE><OPEN>expected output 1<CLOSE><OPEN>arg1$arg2$...<CLOSE><OPEN>expected output 2<CLOSE>...
     into (arguments, expected output) pairs (every argument and output is a Python literal).
  2. Validates each test case with the MBPP reference solution: the test case is valid only if the
     reference solution's output on the arguments equals the generated expected output.
     Otherwise it is an INVALID test case and the generated code is not run on it.
  3. Runs the generated code on each valid test case and asserts that its output equals the
     expected output -> PASS / FAIL (wrong answer) / ERROR (exception) / TLE (time limit).
  4. Builds the control flow graph of the generated code, records the CFG path of every valid
     test case and checks node, edge, edge-pair and prime path coverage (coverage_probe.py).

Reference and generated code each run in their own worker process; every single call has
its own time limit, and a worker that exceeds it is killed and restarted for the next test.
"""

import ast
import math
import multiprocessing
import pickle
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import Config

OPEN_TAG = "<OPEN>"
CLOSE_TAG = "<CLOSE>"
ARG_SEPARATOR = "$"

_MAX_MESSAGE = 500


class Verdict(Enum):
    """Verdict of a single test case, or of a whole suite."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TLE = "TIME LIMIT EXCEEDED"
    INVALID = "INVALID TEST"


# ---------------------------------------------------------------------------
# Parsing of the test-case string
# ---------------------------------------------------------------------------

@dataclass
class ParsedTestCase:
    """One test case as parsed by the executor."""
    index: int
    raw: str  # Raw arguments block
    raw_expected: Optional[str] = None  # Raw expected-output block
    args: Optional[list] = None
    expected: object = None
    parse_error: Optional[str] = None


def _skip_string(text: str, pos: int) -> int:
    """`text[pos]` is a quote; return the index just past the end of that string literal."""
    quote = text[pos] * 3 if text.startswith(text[pos] * 3, pos) else text[pos]
    pos += len(quote)
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
        elif text.startswith(quote, pos):
            return pos + len(quote)
        else:
            pos += 1
    return len(text)


def extract_test_blocks(test_string: str) -> list[tuple[str, Optional[str]]]:
    """
    Split the generator output into the raw contents of each <OPEN>...<CLOSE> block.
    The tags are reserved words (the generator is told never to use them inside values), so block
    boundaries are found from the tags alone: a malformed value, such as an unbalanced quote, then
    only invalidates its own test case instead of swallowing the following ones.
    Returns (content, error) pairs.
    """
    blocks = []
    pos = test_string.find(OPEN_TAG)
    while pos != -1:
        start = pos + len(OPEN_TAG)
        close = test_string.find(CLOSE_TAG, start)
        next_open = test_string.find(OPEN_TAG, start)
        if close == -1 or (next_open != -1 and next_open < close):
            stop = next_open if next_open != -1 else len(test_string)
            blocks.append((test_string[start:stop], f"missing {CLOSE_TAG}"))
            pos = next_open
        else:
            blocks.append((test_string[start:close], None))
            pos = test_string.find(OPEN_TAG, close + len(CLOSE_TAG))
    return blocks


def split_arguments(block: str) -> list[str]:
    """Split a block on `$` that is outside string literals and brackets."""
    if not block.strip():
        return []
    parts, depth, current_start, i = [], 0, 0, 0
    while i < len(block):
        ch = block[i]
        if ch in "\"'":
            i = _skip_string(block, i)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ARG_SEPARATOR and depth == 0:
            parts.append(block[current_start:i])
            current_start = i + 1
        i += 1
    parts.append(block[current_start:])
    return parts


_SAFE_CALLS = {
    "set": set, "frozenset": frozenset, "float": float, "int": int, "str": str,
    "tuple": tuple, "list": list, "dict": dict, "range": range, "bool": bool,
    "complex": complex, "bytes": bytes,
}
_SAFE_NODES = (
    ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Load,
    ast.UnaryOp, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Call, ast.Name,
)


def parse_value(text: str):
    """Parse one argument: a Python literal (plus a few safe constructors like set())."""
    text = text.strip()
    if not text:
        raise ValueError("empty argument")
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    tree = ast.parse(text, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"not a Python literal: {text!r}")
        if isinstance(node, ast.Name) and node.id not in _SAFE_CALLS:
            raise ValueError(f"unknown name '{node.id}' in {text!r}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError(f"not a Python literal: {text!r}")
    return eval(compile(tree, "<test-case>", "eval"), {"__builtins__": {}}, dict(_SAFE_CALLS))


def parse_test_cases(test_string: str) -> list[ParsedTestCase]:
    """
    Parse the full generator output into test cases. Blocks come in pairs: the first block of a
    pair holds the `$`-separated arguments, the second holds the expected output.
    """
    blocks = extract_test_blocks(test_string)
    parsed = []
    for index, i in enumerate(range(0, len(blocks), 2), start=1):
        args_block, args_error = blocks[i]
        case = ParsedTestCase(index=index, raw=args_block.strip())
        if i + 1 >= len(blocks):
            case.parse_error = "missing expected output block"
            parsed.append(case)
            continue
        expected_block, expected_error = blocks[i + 1]
        case.raw_expected = expected_block.strip()

        if args_error or expected_error:
            case.parse_error = args_error or expected_error
        else:
            try:
                case.args = [parse_value(part) for part in split_arguments(args_block)]
            except Exception as e:
                case.parse_error = f"arguments: {type(e).__name__}: {e}"
            else:
                try:
                    case.expected = parse_value(expected_block)
                except Exception as e:
                    case.parse_error = f"expected output: {type(e).__name__}: {e}"
        parsed.append(case)
    return parsed


# ---------------------------------------------------------------------------
# Worker process that loads one piece of code and calls its function on demand
# ---------------------------------------------------------------------------

class Unpicklable:
    """Stand-in for a return value that cannot be sent between processes."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return self.text


def _short(message: str) -> str:
    return message if len(message) <= _MAX_MESSAGE else message[:_MAX_MESSAGE] + "..."


def _describe_unpicklable(value) -> Unpicklable:
    """Text form of a return value that cannot be pickled (e.g. a generator or map object)."""
    if hasattr(value, "__next__"):
        try:
            items = list(value)
            return Unpicklable(_short(f"<{type(value).__name__} yielding {items!r}>"))
        except Exception:
            pass
    return Unpicklable(_short(re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))))


def _worker_main(conn, source, setup_code, func_name, filename, measure_coverage):
    """Entry point of a worker process."""
    import io
    import sys

    sys.stdout, sys.stderr, sys.stdin = io.StringIO(), io.StringIO(), io.StringIO("")

    def load(code):
        namespace = {"__name__": "unit_under_test"}
        if setup_code:
            exec(compile(setup_code, "<setup>", "exec"), namespace)
        exec(code, namespace)
        func = namespace.get(func_name)
        if not callable(func):
            raise NameError(f"the code does not define a function named '{func_name}'")
        return namespace, func

    try:
        _, func = load(compile(source, filename, "exec"))
    except BaseException as e:
        conn.send(("load_error", _short(f"{type(e).__name__}: {e}"), 0.0))
        return

    # Coverage uses a separately instrumented copy of the code. If instrumenting fails, only the
    # coverage is lost: the verdict is always computed with the original function above.
    cov_func = recorder = meta = coverage_error = None
    if measure_coverage:
        try:
            from coverage_probe import PathRecorder, instrument_function
            tree, meta = instrument_function(source, func_name)
            recorder = PathRecorder()
            namespace = {"__name__": "unit_under_test"}
            recorder.install(namespace)
            if setup_code:
                exec(compile(setup_code, "<setup>", "exec"), namespace)
            exec(compile(tree, filename, "exec"), namespace)
            cov_func = namespace[func_name]
        except BaseException as e:
            cov_func = recorder = meta = None
            coverage_error = _short(f"{type(e).__name__}: {e}")
    conn.send(("loaded", {"meta": meta, "coverage_error": coverage_error}, 0.0))

    while True:
        try:
            command, args = conn.recv()
        except (EOFError, OSError):
            return
        if command == "exit":
            return
        if command == "call":
            start = time.perf_counter()
            try:
                value = func(*args)
                elapsed = time.perf_counter() - start
                try:
                    pickle.dumps(value)
                except Exception:
                    value = _describe_unpicklable(value)
                conn.send(("ok", value, elapsed))
            except BaseException as e:
                elapsed = time.perf_counter() - start
                conn.send(("error", _short(f"{type(e).__name__}: {e}"), elapsed))
        elif command == "cover":
            if recorder is None:
                conn.send(("cover", None, 0.0))
                continue
            recorder.reset()
            try:
                cov_func(*args)
            except BaseException:
                pass
            conn.send(("cover", recorder.snapshot(), 0.0))


class _CodeRunner:
    """Parent-side handle of a worker process, with a time limit on every request."""

    def __init__(self, source, setup_code, func_name, filename, timeout, measure_coverage=False):
        self.args = (source, setup_code, func_name, filename, measure_coverage)
        self.timeout = timeout
        self.ctx = multiprocessing.get_context("spawn")
        self.process = None
        self.conn = None
        self.load_error: Optional[str] = None
        self.coverage_meta: Optional[dict] = None
        self.coverage_error: Optional[str] = None

    def _kill(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=2)
        if self.conn is not None:
            self.conn.close()
        self.process = self.conn = None

    def _start(self) -> bool:
        parent_conn, child_conn = self.ctx.Pipe()
        process = self.ctx.Process(
            target=_worker_main, args=(child_conn, *self.args), daemon=True
        )
        try:
            process.start()
        except Exception as e:
            parent_conn.close()
            child_conn.close()
            self.load_error = f"could not start worker process: {type(e).__name__}: {e}"
            return False
        self.process = process
        child_conn.close()
        self.conn = parent_conn
        # Allow for interpreter start-up on top of the time limit for loading the code
        status, payload, _ = self._receive(self.timeout + 10)
        if status == "loaded":
            self.coverage_meta = payload["meta"]
            self.coverage_error = payload["coverage_error"]
            return True
        self.load_error = payload if status == "load_error" else (
            "loading the code exceeded the time limit" if status == "timeout" else payload
        )
        self._kill()
        return False

    def _receive(self, timeout):
        try:
            if self.conn.poll(timeout):
                return self.conn.recv()
            return ("timeout", None, float(timeout))
        except (EOFError, OSError):
            return ("error", "worker process crashed", 0.0)

    def request(self, command: str, args: list) -> tuple:
        """Send a command; returns (status, payload, elapsed_seconds)."""
        if self.load_error:
            return ("load_error", self.load_error, 0.0)
        if self.process is None and not self._start():
            return ("load_error", self.load_error, 0.0)
        try:
            self.conn.send((command, args))
        except (OSError, BrokenPipeError):
            self._kill()
            return ("error", "worker process crashed", 0.0)
        status, payload, elapsed = self._receive(self.timeout)
        if status == "timeout" or (status == "error" and payload == "worker process crashed"):
            self._kill()  # restarted lazily for the next request
        return status, payload, elapsed

    def close(self):
        if self.process is not None and self.process.is_alive():
            try:
                self.conn.send(("exit", None))
            except (OSError, BrokenPipeError):
                pass
        self._kill()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def show(value) -> str:
    """Python representation of a value as written in the results."""
    return value.text if isinstance(value, Unpicklable) else repr(value)


def outputs_match(actual, expected) -> bool:
    """Python == semantics, except floats are compared with a small tolerance."""
    if isinstance(actual, Unpicklable) or isinstance(expected, Unpicklable):
        return show(actual) == show(expected)
    numbers = (int, float)
    if (
        (isinstance(actual, float) or isinstance(expected, float))
        and isinstance(actual, numbers) and isinstance(expected, numbers)
        and not isinstance(actual, bool) and not isinstance(expected, bool)
    ):
        if math.isnan(actual) and math.isnan(expected):
            return True
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, (list, tuple)) and type(actual) is type(expected):
        return len(actual) == len(expected) and all(
            outputs_match(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            outputs_match(actual[k], expected[k]) for k in actual
        )
    try:
        return bool(actual == expected)
    except Exception:
        return False


@dataclass
class TestCaseResult:
    """Result of one parsed test case."""
    test_id: int
    raw_input: str
    parsed_arguments: Optional[list[str]] = None
    call: Optional[str] = None
    expected_output: Optional[str] = None  # Generated by the Test Case Generator
    reference_output: Optional[str] = None  # Output of the reference solution on the same input
    actual_output: Optional[str] = None  # Output of the generated code
    assertion: Optional[str] = None
    verdict: Verdict = Verdict.INVALID
    error_message: Optional[str] = None
    path: Optional[list[int]] = None  # CFG test path executed by the generated code
    recursive_call_paths: Optional[list[list[int]]] = None
    covers: Optional[list[list[int]]] = None  # Target-criterion requirements toured by this test

    def to_dict(self) -> dict:
        data = {
            "id": self.test_id,
            # Parsed arguments; the raw text is shown only if the test case could not be parsed
            "input": self.parsed_arguments if self.parsed_arguments is not None else self.raw_input,
            "expected": self.expected_output,
            "reference_output": self.reference_output,
            "actual": self.actual_output,
            "verdict": self.verdict.value,
            "error": self.error_message,
        }
        if self.path is not None:
            data["path"] = self.path
            if self.recursive_call_paths:
                data["recursive_call_paths"] = self.recursive_call_paths
            data["covers"] = self.covers
        return data


COVERAGE_NAMES = {
    "node": "Node Coverage",
    "edge": "Edge Coverage",
    "edge_pair": "Edge-Pair Coverage",
    "prime_path": "Prime Path Coverage",
}


def format_coverage(coverage: Optional[dict]) -> Optional[dict]:
    """
    Coverage in the form written to execution_results.json:
    target criterion, whether it is met, covered/total for all four criteria, the uncovered
    requirements, and the CFG (node table + edges) that the paths refer to.
    """
    if not coverage or "error" in coverage:
        return coverage
    formatted = {
        "target_criterion": COVERAGE_NAMES[coverage["target"]],
        "criterion_met": coverage["criterion_met"],
    }
    uncovered = {}
    for crit, data in coverage["criteria"].items():
        if "error" in data:
            formatted[crit] = data["error"]
        elif data["total"]:
            formatted[crit] = f"{data['covered']}/{data['total']} ({data['percent']}%)"
            if data["uncovered"]:
                uncovered[crit] = data["uncovered"]
        else:
            formatted[crit] = "n/a"
    if uncovered:
        formatted["uncovered"] = uncovered
        formatted["note"] = (
            "Uncovered requirements may be infeasible (no input can execute them). "
            "Paths are lists of CFG node ids; see cfg.nodes."
        )
    formatted["cfg"] = coverage["cfg"]
    return formatted


@dataclass
class ExecutionResult:
    """Result of executing the test suite of one problem."""
    task_id: int
    function_name: str
    verdict: Verdict = Verdict.INVALID
    test_results: list[TestCaseResult] = field(default_factory=list)
    coverage: Optional[dict] = None
    error_message: Optional[str] = None

    def _count(self, verdict: Verdict) -> int:
        return sum(1 for t in self.test_results if t.verdict == verdict)

    @property
    def total_tests(self) -> int:
        return len(self.test_results)

    @property
    def valid_tests(self) -> int:
        return self.total_tests - self.invalid

    @property
    def passed(self) -> int:
        return self._count(Verdict.PASS)

    @property
    def failed(self) -> int:
        return self._count(Verdict.FAIL)

    @property
    def errors(self) -> int:
        return self._count(Verdict.ERROR)

    @property
    def tle_count(self) -> int:
        return self._count(Verdict.TLE)

    @property
    def invalid(self) -> int:
        return self._count(Verdict.INVALID)

    def counts(self) -> dict:
        return {
            "total": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "timeouts": self.tle_count,
            "invalid": self.invalid,
        }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class TestExecutor:
    """
    Agent that parses generated test cases, validates their expected outputs with the reference
    solution and asserts the generated code against the valid ones.
    """

    def __init__(self, config: Config):
        self.config = config

    def execute(
        self,
        task_id: int,
        function_name: str,
        generated_code: str,
        reference_code: str,
        test_string: str,
        setup_code: str = "",
    ) -> ExecutionResult:
        result = ExecutionResult(task_id=task_id, function_name=function_name)

        parsed = parse_test_cases(test_string or "")
        if not parsed:
            result.error_message = f"No test cases found in the {OPEN_TAG}...{CLOSE_TAG} format."
            return result

        timeout = self.config.timeout_seconds
        reference = _CodeRunner(reference_code, setup_code, function_name, "<reference>", timeout)
        generated = _CodeRunner(
            generated_code, setup_code, function_name, "<generated>", timeout, measure_coverage=True
        )

        traces = {}  # test id -> CFG paths recorded by the instrumented copy

        try:
            for case in parsed:
                tr = self._run_case(case, function_name, reference, generated)
                result.test_results.append(tr)
                if tr.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.ERROR) and not generated.load_error:
                    status, snapshot, _ = generated.request("cover", case.args)
                    if status == "cover" and snapshot is not None:
                        traces[case.index] = snapshot
            if generated.coverage_meta is not None and traces:
                self._attach_coverage(result, generated.coverage_meta, traces)
            elif generated.coverage_error:
                result.coverage = {"error": f"Coverage not measured: {generated.coverage_error}"}
        finally:
            reference.close()
            generated.close()

        if generated.load_error:
            result.error_message = f"Generated code failed to load: {generated.load_error}"
        elif reference.load_error:
            result.error_message = f"Reference solution failed to load: {reference.load_error}"

        result.verdict = self._suite_verdict(result)
        return result

    def _attach_coverage(self, result: ExecutionResult, meta: dict, traces: dict):
        """Check the test paths against the test requirements of all four criteria."""
        from coverage_probe import analyze
        try:
            analysis = analyze(meta, traces, self.config.coverage_type)
        except Exception as e:
            result.coverage = {"error": f"Coverage not measured: {type(e).__name__}: {e}"}
            return
        for tr in result.test_results:
            info = analysis["per_test"].get(tr.test_id)
            if info:
                tr.path = info["path"]
                tr.recursive_call_paths = info["recursive_call_paths"]
                tr.covers = info["covers"]
        del analysis["per_test"]
        result.coverage = analysis

    def _run_case(self, case: ParsedTestCase, function_name: str, reference, generated) -> TestCaseResult:
        tr = TestCaseResult(test_id=case.index, raw_input=case.raw)
        if case.parse_error:
            tr.expected_output = case.raw_expected
            tr.error_message = f"Parse error: {case.parse_error}"
            return tr

        tr.parsed_arguments = [show(a) for a in case.args]
        tr.call = f"{function_name}({', '.join(tr.parsed_arguments)})"
        expected = case.expected
        tr.expected_output = show(expected)
        tr.assertion = f"assert {tr.call} == {tr.expected_output}"

        # Validity check: the reference solution must produce the generated expected output
        status, payload, _ = reference.request("call", case.args)
        if status != "ok":
            reason = {
                "timeout": "exceeded the time limit",
                "load_error": f"failed to load ({payload})",
            }.get(status, f"raised {payload}")
            tr.error_message = f"Invalid input: the reference solution {reason} on this input."
            return tr
        tr.reference_output = show(payload)
        if not outputs_match(payload, expected):
            tr.error_message = (
                f"Wrong expected output: the test case expects {tr.expected_output}, "
                f"but the reference solution returns {tr.reference_output}."
            )
            return tr

        # Unit under test
        status, payload, _ = generated.request("call", case.args)
        if status == "timeout":
            tr.verdict = Verdict.TLE
            tr.error_message = f"Generated code exceeded the {self.config.timeout_seconds}s time limit."
        elif status == "load_error":
            tr.verdict = Verdict.ERROR
            tr.error_message = f"Generated code failed to load: {payload}"
        elif status != "ok":
            tr.verdict = Verdict.ERROR
            tr.error_message = f"Generated code raised {payload}"
        else:
            actual = payload
            tr.actual_output = show(actual)
            try:
                assert outputs_match(actual, expected), (
                    f"expected {tr.expected_output}, got {tr.actual_output}"
                )
                tr.verdict = Verdict.PASS
            except AssertionError as e:
                tr.verdict = Verdict.FAIL
                tr.error_message = f"AssertionError: {e}"
        return tr

    @staticmethod
    def _suite_verdict(result: ExecutionResult) -> Verdict:
        """Verdict of the generated code on the valid test cases of this suite."""
        if result.valid_tests == 0:
            return Verdict.INVALID
        for verdict in (Verdict.FAIL, Verdict.ERROR, Verdict.TLE):
            if result._count(verdict):
                return verdict
        return Verdict.PASS
