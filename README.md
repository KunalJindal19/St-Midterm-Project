# AI-Assisted Unit Testing Pipeline

**CSE731: Software Testing — Mid-term Project**

Repository: https://github.com/KunalJindal19/St-Midterm-Project

## Quick Start

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/KunalJindal19/St-Midterm-Project.git
cd St-Midterm-Project
pip install -r requirements.txt
```

### 2. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```
Or set the environment variable:
```bash
export OPENROUTER_API_KEY=your_key_here
```

### 3. Run the pipeline

**Interactive mode** (asks for the coverage criterion and the number of problems):
```bash
python3 cli.py
```

**Batch mode**:
```bash
python3 cli.py run                  # Prime Path Coverage (default), 1 problem
python3 cli.py run -c edge -n 5     # Edge Coverage, 5 problems
python3 cli.py run -k <api-key>     # pass the OpenRouter key on the command line
```
Coverage criteria: `node`, `edge`, `edge_pair`, `prime_path` (default).

**Show pipeline info**:
```bash
python3 cli.py info
```

Fixed settings (in `config.py`): model `cohere/north-mini-code:free` (with free fallbacks),
temperature 0.7, max tokens 4096, 10 s time limit per test case, MBPP test split, output folder `output/`.

## How the pipeline works

1. **Code Generator** — the LLM returns exactly one Python function (imports and helpers go inside
   its body, nothing before or after it). The response is checked with `ast`; if it is invalid (syntax
   error, anything outside the function, wrong function name) the model is asked once more.
2. **Test Case Generator** — the LLM returns test cases (inputs **and** expected outputs) that achieve the
   selected structural coverage criterion on the generated code:
   `<OPEN>arg1$arg2<CLOSE><OPEN>expected output 1<CLOSE><OPEN>arg1$arg2<CLOSE><OPEN>expected output 2<CLOSE>...`
   (every argument and output is a Python literal). It sees the problem description, the MBPP example asserts
   and the generated code, but not the reference solution.
   The response must consist of nothing but test cases (start with `<OPEN>`, end with `<CLOSE>`, even
   number of blocks, no other text); otherwise it is rejected and the model is asked once more with the reason.
3. **Test Executor** — parses every test case and validates it with the MBPP reference solution: a test case
   is valid only if the reference solution returns the same expected output. It then defines the generated
   function and calls it on every valid test case: `assert generated(*args) == expected_output`. Each call
   runs in a separate process with its own time limit.
4. **Coverage verification** (part of the executor) — builds the control flow graph of the generated
   function from its AST (basic blocks as nodes, one decision node per `if`/`while` condition, an
   `exit` node), records the CFG path each valid test case executes, and checks the test requirements of
   all four criteria (node, edge, edge-pair, prime path). The selected criterion is reported as **met** or
   **not met**, with the uncovered requirements listed.

Per-test verdicts: `PASS`, `FAIL` (wrong answer), `ERROR` (exception), `TIME LIMIT EXCEEDED`, and
`INVALID TEST` (the test does not parse, its expected output differs from the reference solution's output, or the
reference solution fails on that input — the generated code is not run on invalid test cases).

## Structural Coverage Criteria

| Criterion | Requirement |
|-----------|-------------|
| Node Coverage | Every node (statement) of the control flow graph is executed |
| Edge Coverage | Every edge (branch outcome) of the control flow graph is traversed |
| Edge-Pair Coverage | Every path of length up to two edges is toured |
| Prime Path Coverage (default) | Every prime path is toured |

## Project Structure

```
Midterm_Project/
├── cli.py                  # CLI frontend (interactive & batch)
├── config.py               # Configuration management
├── code_generator.py       # Code Generator Agent
├── test_case_generator.py  # Test Case Generator Agent
├── test_executor.py        # Test Executor Agent (parser, validation, assertions)
├── coverage_probe.py       # CFG construction, test paths, coverage verification
├── pipeline.py             # Pipeline driver (orchestrator, output)
├── llm_client.py           # OpenRouter client with model fallback
├── requirements.txt        # Python dependencies
├── .env.example            # Example .env file
├── report.tex              # LaTeX report
└── output/
    ├── summary.json
    └── task_<id>/
        ├── generated_code.py, reference_code.py
        ├── code_gen_prompts.json, test_gen_prompts.json   # prompts, settings, raw LLM responses
        ├── test_cases.txt                                 # generated test cases (inputs + expected outputs)
        └── execution_results.json   # verdict, CFG + coverage check, parsed test cases with paths
```

## Example `execution_results.json`

```json
{
  "task_id": 11,
  "function": "remove_Occ(s, ch)",
  "coverage_criterion": "Prime Path Coverage",
  "verdict": "PASS",
  "summary": {
    "total": 9,
    "passed": 9,
    "failed": 0,
    "errors": 0,
    "timeouts": 0,
    "invalid": 0
  },
  "coverage": {
    "target_criterion": "Prime Path Coverage",
    "criterion_met": true,
    "node": "4/4 (100.0%)",
    "edge": "4/4 (100.0%)",
    "edge_pair": "2/2 (100.0%)",
    "prime_path": "2/2 (100.0%)",
    "cfg": {
      "nodes": {
        "1": "lines 2-4: first = s.find(ch); last = s.rfind(ch); if first == -1 or last == -1:",
        "2": "line 5: return s",
        "3": "line 6: return s[:first] + s[first+1:last] + s[last+1:]",
        "4": "exit"
      },
      "edges": [[1, 2], [1, 3], [2, 4], [3, 4]],
      "initial_node": 1,
      "final_node": 4
    }
  },
  "test_cases": [
    {
      "id": 1,
      "input": ["'hello'", "'l'"],
      "expected": "'heo'",
      "reference_output": "'heo'",
      "actual": "'heo'",
      "verdict": "PASS",
      "error": null,
      "path": [1, 3, 4],
      "covers": [[1, 3, 4]]
    },
    {
      "id": 4,
      "input": ["'hello'", "'x'"],
      "expected": "'hello'",
      "reference_output": "'hello'",
      "actual": "'hello'",
      "verdict": "PASS",
      "error": null,
      "path": [1, 2, 4],
      "covers": [[1, 2, 4]]
    }
  ]
}
```
(abridged: the real file has nine test cases)

- `input` and `expected` are the test case as parsed by the executor (generated by the LLM);
  `reference_output` is what the MBPP reference solution returns (the test case is valid only if it equals
  `expected`); `actual` is what the generated function returns.
- `path` is the CFG test path of the test case (node ids, see `coverage.cfg.nodes`) and `covers` lists the
  requirements of the selected criterion that it tours.
- `coverage.criterion_met` says whether the selected criterion is achieved; when it is not, `uncovered`
  lists the missing requirements for each criterion (some may be infeasible).
- If code or test generation fails, the file only contains `"verdict": "NOT RUN"` and the `error`.

## Notes

- Coverage is verified on the CFG of the generated function. Uncovered requirements may be infeasible (no input
  can execute them), so a criterion can be reported as not met even for the best possible test suite.
- Full coverage of the generated code does not guarantee a strong test suite: coverage only exercises
  paths that exist in the code, so missing behaviour in the generated function can go undetected.
- Works on Python 3.9+.
- The Hugging Face warning about unauthenticated requests is harmless; set `HF_TOKEN` to silence it.
- Free OpenRouter models have per-minute and per-day request limits; on rate limits the client falls back
  to other free models. Responses cut off at the token limit are discarded and the next model is tried.

## API Key

Get a free API key from [OpenRouter](https://openrouter.ai/).
