#!/usr/bin/env python3
"""
CLI Frontend for the AI-assisted Unit Testing Pipeline.
Provides an interactive command-line interface to configure and run the pipeline.
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.markup import escape
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.rule import Rule
from rich import box

from config import Config, COVERAGE_TYPES, COVERAGE_NAMES
from pipeline import Pipeline, PipelineSummary, PipelineResult
from test_executor import Verdict, format_coverage

console = Console(emoji=False)  # icons are literal unicode; keeps ":free" in model names intact


BANNER = r"""
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║         AI-Assisted Unit Testing Pipeline                    ║
║         CSE731: Software Testing — Midterm Project           ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
"""

VERDICT_STYLE = {
    Verdict.PASS: "[bold green]PASS[/bold green]",
    Verdict.FAIL: "[bold red]FAIL[/bold red]",
    Verdict.ERROR: "[bold yellow]ERROR[/bold yellow]",
    Verdict.TLE: "[bold magenta]TIME LIMIT EXCEEDED[/bold magenta]",
    Verdict.INVALID: "[bold white]INVALID TEST[/bold white]",
}

VERDICT_ICON = {
    Verdict.PASS: "✅",
    Verdict.FAIL: "❌",
    Verdict.ERROR: "💥",
    Verdict.TLE: "⏱️ ",
    Verdict.INVALID: "⚪",
}


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI mode."""
    parser = argparse.ArgumentParser(
        description="AI-Assisted Unit Testing Pipeline — CSE731 Midterm Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py                          # Interactive mode\n"
            "  python cli.py run                      # Prime path coverage, 1 problem\n"
            "  python cli.py run -c edge -n 5         # Edge coverage, 5 problems\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "--coverage", "-c",
        choices=list(COVERAGE_TYPES),
        default="prime_path",
        help="Structural coverage criterion (default: prime_path)",
    )
    run_parser.add_argument(
        "--num-problems", "-n",
        type=int, default=1,
        help="Number of MBPP problems to process (1-50, default: 1)",
    )
    run_parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var / .env file)",
    )

    subparsers.add_parser("interactive", help="Run in interactive mode")
    subparsers.add_parser("info", help="Show pipeline information")

    return parser


def show_banner():
    """Display the application banner."""
    console.print(BANNER)


def show_config(config: Config):
    """Display the current configuration."""
    table = Table(title="Pipeline Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Coverage Criterion", config.coverage_label())
    table.add_row("Number of Problems", str(config.num_problems))
    table.add_row("Model", config.model)
    table.add_row("Temperature", str(config.temperature))
    table.add_row("Time Limit per Test", f"{config.timeout_seconds}s")
    table.add_row("Dataset", f"{config.dataset_name} ({config.dataset_split} split)")
    table.add_row("Output Directory", config.output_dir)
    table.add_row("API Key", "***" + config.api_key[-4:] if config.api_key else "[red]NOT SET[/red]")

    console.print(table)


def progress_callback(step: str, task_id: int, detail: str):
    """Callback for pipeline progress updates."""
    icons = {
        "dataset": "📦",
        "pipeline": "🔄",
        "code_gen": "🔨",
        "test_gen": "🧪",
        "executor": "▶️ ",
        "save": "💾",
    }
    icon = icons.get(step, "  ")
    task_str = f"[dim]Task {task_id}[/dim] " if task_id else ""
    console.print(f"  {icon} {task_str}{escape(detail)}", highlight=False)


def display_result(result: PipelineResult):
    """Display results for a single problem."""
    console.print(Rule(f"Task {result.task_id}", style="blue"))
    console.print(Panel(
        result.problem_text,
        title="[bold]Problem Description[/bold]",
        border_style="blue",
        padding=(0, 1),
    ))

    if result.generated_code and result.generated_code.success:
        console.print(Syntax(
            result.generated_code.generated_code,
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        ))

    exec_res = result.execution
    if exec_res is None:
        console.print(f"  [red]{escape(result.error or 'Not executed')}[/red]")
        return

    console.print(
        f"\n  Verdict: {VERDICT_STYLE[exec_res.verdict]}  "
        f"({exec_res.passed}/{exec_res.valid_tests} valid tests passed, "
        f"{exec_res.failed} failed, {exec_res.errors} errors, {exec_res.tle_count} TLE, "
        f"{exec_res.invalid} invalid)"
    )
    _display_coverage(format_coverage(exec_res.coverage))
    if exec_res.error_message:
        console.print(f"  [red]{escape(exec_res.error_message)}[/red]", highlight=False)

    for tr in exec_res.test_results:
        shown = tr.assertion or tr.call or tr.raw_input
        console.print(f"    {VERDICT_ICON[tr.verdict]} {shown[:100]}", markup=False, highlight=False)
        if tr.path is not None:
            covers = f"  covers {tr.covers}" if tr.covers else ""
            console.print(f"       path {tr.path}{covers}", style="dim", markup=False, highlight=False)
        if tr.error_message:
            console.print(f"       {tr.error_message}", style="dim red", markup=False, highlight=False)


def _display_coverage(coverage):
    """Show the CFG, the coverage of all four criteria and whether the selected one is met."""
    if not coverage:
        return
    if "error" in coverage:
        console.print(f"  [yellow]{escape(coverage['error'])}[/yellow]")
        return
    met = "[bold green]MET[/bold green]" if coverage["criterion_met"] else "[bold red]NOT MET[/bold red]"
    console.print(f"  {coverage['target_criterion']}: {met}")
    console.print(
        "  Coverage: " + ", ".join(f"{c} {coverage[c]}" for c in COVERAGE_TYPES), highlight=False
    )
    console.print("  CFG nodes:", highlight=False)
    for nid, label in coverage["cfg"]["nodes"].items():
        console.print(f"    {nid}: {label}", markup=False, highlight=False)
    console.print(f"  CFG edges: {coverage['cfg']['edges']}", markup=False, highlight=False)
    for crit, paths in coverage.get("uncovered", {}).items():
        console.print(f"  Uncovered {crit}: {paths}", style="yellow", markup=False, highlight=False)


def display_summary(summary: PipelineSummary):
    """Display the overall pipeline summary."""
    console.print("\n")
    console.print(Rule("Pipeline Summary", style="bold green"))

    c = summary.counts
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Coverage Criterion", summary.config["coverage_criterion"])
    table.add_row("Total Problems", str(summary.total_problems))
    table.add_row("Code generation failed", str(c["code_generation_failed"]))
    table.add_row("Test generation failed", str(c["test_generation_failed"]))
    table.add_row("", "")
    table.add_row("[bold]Verdicts (generated code)[/bold]", "")
    for verdict in Verdict:
        table.add_row(f"  {verdict.value}", str(c["verdicts"][verdict.value]))
    table.add_row("", "")
    table.add_row("[bold]Coverage criterion[/bold]", "")
    table.add_row("  Met", f"[green]{c['coverage_criterion_met']}[/green]")
    table.add_row("  Not met", f"[red]{c['coverage_criterion_not_met']}[/red]")
    table.add_row("", "")
    t = c["test_cases"]
    table.add_row("[bold]Test Cases[/bold]", str(t["total"]))
    table.add_row("  Passed", f"[green]{t['passed']}[/green]")
    table.add_row("  Failed", f"[red]{t['failed']}[/red]")
    table.add_row("  Errors", f"[yellow]{t['errors']}[/yellow]")
    table.add_row("  Timeouts", f"[magenta]{t['timeouts']}[/magenta]")
    table.add_row("  Invalid", str(t["invalid"]))
    table.add_row("", "")
    table.add_row("Total Time", f"{summary.total_time:.1f}s")

    console.print(table)


def run_interactive():
    """Run the pipeline in interactive mode."""
    show_banner()

    config = Config()
    if not config.api_key:
        config.api_key = Prompt.ask("[yellow]Enter your OpenRouter API key[/yellow]", password=True)

    console.print("\n[bold]Select Structural Coverage Criterion:[/bold]")
    for i, cov in enumerate(COVERAGE_TYPES, start=1):
        default = " (default)" if cov == "prime_path" else ""
        console.print(f"  [cyan]{i}[/cyan] — {COVERAGE_NAMES[cov]}{default}")
    default_choice = str(COVERAGE_TYPES.index("prime_path") + 1)
    choice = Prompt.ask(
        "Your choice", choices=[str(i) for i in range(1, len(COVERAGE_TYPES) + 1)], default=default_choice
    )
    config.coverage_type = COVERAGE_TYPES[int(choice) - 1]

    config.num_problems = IntPrompt.ask("Number of MBPP problems to process", default=1)
    config.num_problems = max(1, min(config.num_problems, config.max_problems))

    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"  [red]✗ {err}[/red]")
        return

    console.print()
    show_config(config)
    console.print()

    if not Confirm.ask("Proceed with this configuration?", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        return

    _execute_pipeline(config)


def _execute_pipeline(config: Config):
    """Execute the pipeline with the given configuration."""
    console.print()
    console.print(Rule("Running Pipeline", style="bold green"))
    console.print()

    pipeline = Pipeline(config, progress_callback=progress_callback)

    try:
        summary = pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        return
    except Exception as e:
        console.print(f"\n[red]Pipeline error: {escape(str(e))}[/red]")
        return

    console.print()
    console.print(Rule("Results", style="bold blue"))

    for result in summary.results:
        display_result(result)

    display_summary(summary)

    pipeline.save_results(summary)
    console.print(f"\n[green]Results saved to {config.output_dir}/[/green]")


def run_from_args(args):
    """Run the pipeline from parsed command-line arguments."""
    config = Config(
        api_key=args.api_key or "",
        num_problems=args.num_problems,
        coverage_type=args.coverage,
    )

    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"  [red]✗ {err}[/red]")
        sys.exit(1)

    show_banner()
    show_config(config)
    _execute_pipeline(config)


def show_info():
    """Display pipeline information."""
    show_banner()

    info = """
## Pipeline Architecture

The pipeline consists of three AI agents:

1. **Code Generator** — generates a single Python function from an MBPP problem description
2. **Test Case Generator** — generates test cases (inputs and expected outputs) achieving the selected
   structural coverage criterion on the generated code, in the format
   `<OPEN>arg1$arg2<CLOSE><OPEN>expected output<CLOSE>...`
3. **Test Executor** — parses the test cases, keeps only those whose expected output matches the MBPP
   reference solution's output (the others are INVALID TEST), then calls the generated function on each
   valid test case and asserts the result (PASS / FAIL / ERROR / TLE); it also builds the control flow
   graph of the generated code and checks node, edge, edge-pair and prime path coverage

## Structural Coverage Criteria

| Criterion | Requirement |
|-----------|-------------|
| Node Coverage | Every node (statement) of the control flow graph is executed |
| Edge Coverage | Every edge (branch outcome) of the control flow graph is traversed |
| Edge-Pair Coverage | Every path of length up to two edges is toured |
| Prime Path Coverage (default) | Every prime path is toured |

## Dataset

- **MBPP** (Most Basic Python Problems) — the reference solution of each problem validates the generated test cases

## Usage

```bash
python cli.py                      # Interactive mode
python cli.py run                  # Prime path coverage, 1 problem
python cli.py run -c edge -n 5     # Edge coverage, 5 problems
```
"""
    console.print(Markdown(info))


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_from_args(args)
    elif args.command == "info":
        show_info()
    else:
        # Default to interactive mode
        run_interactive()


if __name__ == "__main__":
    main()
