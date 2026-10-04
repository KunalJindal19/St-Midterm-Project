"""
Structural (graph) coverage measurement for the Test Executor.

1. Control flow graph: the function under test is parsed with `ast` and its CFG is built.
   Statements are grouped into basic blocks (the CFG nodes); a synthetic "exit" node is the
   final node. Every if/elif/while condition is one decision node (`a or b` is not split),
   every loop header is its own node. Nested function/class definitions are single statements.
2. Test paths: a copy of the function is instrumented (in the same pass that builds the CFG)
   so that each call records the sequence of CFG nodes it executes — its test path.
3. Test requirements for all four criteria (Ammann & Offutt) are computed from the CFG and each
   one is checked against the test paths (a requirement is covered when it is a sub-path of a
   test path, i.e. the test path tours it directly):
     node       : every reachable node
     edge       : every reachable edge
     edge_pair  : every reachable path of length up to 2 edges
     prime_path : every prime path (maximal simple path)

The pass/fail verdict is always computed on the original, un-instrumented code; the
instrumented copy is only executed to record test paths.
"""

import ast

PROBE_NODE = "_cfg_probe_node"
PROBE_TEST = "_cfg_probe_test"
PROBE_ITER = "_cfg_probe_iter"
PROBE_ENTER = "_cfg_probe_enter"
PROBE_FAIL = "_cfg_probe_fail"
PROBE_EXIT = "_cfg_probe_exit"

EXIT = 0  # Statement-level id of the synthetic exit node
MAX_SIMPLE_PATHS = 20000  # Safety limit for prime path enumeration
MAX_LISTED_RECURSIVE_PATHS = 10  # Recursive-call paths listed per test case (all are used for coverage)

CRITERIA = ("node", "edge", "edge_pair", "prime_path")

_MATCH_NODE = getattr(ast, "Match", None)  # Python 3.10+
_TRY_NODES = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _call(name: str, args: list, loc: ast.AST) -> ast.expr:
    call = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
    return ast.copy_location(call, loc)


def _probe(name: str, args: list, loc: ast.AST) -> ast.stmt:
    return ast.copy_location(ast.Expr(value=_call(name, args, loc)), loc)


# ---------------------------------------------------------------------------
# CFG construction + instrumentation
# ---------------------------------------------------------------------------

class _CFGBuilder:
    """Builds the statement-level CFG of one function and instruments it in the same pass."""

    def __init__(self, source: str):
        self.lines = source.splitlines()
        self.nodes: dict[int, dict] = {}  # statement-level id -> {"line", "text"}
        self.edges: set[tuple[int, int]] = set()
        self.loop_headers: set[int] = set()
        self.loops: list[tuple[int, list]] = []  # stack of (header id, break nodes)

    def _node(self, stmt: ast.stmt) -> int:
        nid = len(self.nodes) + 1
        text = self.lines[stmt.lineno - 1].strip() if stmt.lineno <= len(self.lines) else ""
        self.nodes[nid] = {"line": stmt.lineno, "text": text}
        return nid

    def _link(self, preds: list, nid: int):
        for p in preds:
            self.edges.add((p, nid))

    def build(self, func: ast.FunctionDef):
        body = list(func.body)
        docstring = [body.pop(0)] if body and _is_docstring(body[0]) else []
        new_body, preds = self._block(body, [])
        self._link(preds, EXIT)
        if not new_body:
            new_body = [ast.Pass()]

        # Record one test path per call: enter -> nodes... -> exit (exit only on normal return)
        handler = ast.ExceptHandler(
            type=ast.Name(id="BaseException", ctx=ast.Load()), name=None,
            body=[_probe(PROBE_FAIL, [], func), ast.Raise(exc=None, cause=None)],
        )
        wrapped = ast.Try(body=new_body, handlers=[handler], orelse=[],
                          finalbody=[_probe(PROBE_EXIT, [], func)])
        func.body = docstring + [_probe(PROBE_ENTER, [], func), ast.copy_location(wrapped, func)]

    def _block(self, stmts: list, preds: list) -> tuple[list, list]:
        out = []
        for stmt in stmts:
            if isinstance(stmt, (ast.Global, ast.Nonlocal)):
                out.append(stmt)
                continue
            new_stmts, preds = self._statement(stmt, preds)
            out.extend(new_stmts)
        return out, preds

    def _loop(self, node: ast.stmt, nid: int) -> list:
        """Body of a while/for loop; returns the break nodes."""
        self.loop_headers.add(nid)
        self.loops.append((nid, []))
        node.body, body_out = self._block(node.body, [nid])
        self._link(body_out, nid)
        return self.loops.pop()[1]

    def _statement(self, s: ast.stmt, preds: list) -> tuple[list, list]:
        if isinstance(s, ast.If):
            n = self._node(s)
            self._link(preds, n)
            s.body, body_out = self._block(s.body, [n])
            if s.orelse:
                s.orelse, else_out = self._block(s.orelse, [n])
            else:
                else_out = [n]
            return [_probe(PROBE_NODE, [ast.Constant(n)], s), s], body_out + else_out

        if isinstance(s, ast.While):
            n = self._node(s)
            self._link(preds, n)
            always_true = isinstance(s.test, ast.Constant) and bool(s.test.value)
            s.test = _call(PROBE_TEST, [ast.Constant(n), s.test], s.test)
            breaks = self._loop(s, n)
            exit_preds = [] if always_true else [n]
            if s.orelse:
                s.orelse, exit_preds = self._block(s.orelse, exit_preds)
            return [s], exit_preds + breaks

        if isinstance(s, ast.For):
            n = self._node(s)
            self._link(preds, n)
            s.iter = _call(PROBE_ITER, [ast.Constant(n), s.iter], s.iter)
            breaks = self._loop(s, n)
            exit_preds = [n]
            if s.orelse:
                s.orelse, exit_preds = self._block(s.orelse, exit_preds)
            return [s], exit_preds + breaks

        if isinstance(s, _TRY_NODES):
            # `try:` itself executes nothing: control goes straight into the body, and an
            # exception can transfer control from any statement of the body to each handler
            first_body_id = len(self.nodes) + 1
            s.body, body_out = self._block(s.body, preds)
            body_nodes = list(range(first_body_id, len(self.nodes) + 1))
            handler_out = []
            for h in s.handlers:
                h.body, h_out = self._block(h.body, body_nodes)
                handler_out += h_out
            if s.orelse:
                s.orelse, body_out = self._block(s.orelse, body_out)
            outs = body_out + handler_out
            if s.finalbody:
                s.finalbody, outs = self._block(s.finalbody, outs)
            return [s], outs

        n = self._node(s)
        self._link(preds, n)
        probe = _probe(PROBE_NODE, [ast.Constant(n)], s)

        if isinstance(s, ast.Break):
            if self.loops:
                self.loops[-1][1].append(n)
            return [probe, s], []
        if isinstance(s, ast.Continue):
            if self.loops:
                self._link([n], self.loops[-1][0])
            return [probe, s], []
        if isinstance(s, (ast.Return, ast.Raise)):
            self._link([n], EXIT)
            return [probe, s], []
        if isinstance(s, ast.With):
            s.body, out = self._block(s.body, [n])
            return [probe, s], out
        if _MATCH_NODE is not None and isinstance(s, _MATCH_NODE):
            outs = [n]  # no case matched
            for case in s.cases:
                case.body, c_out = self._block(case.body, [n])
                outs += c_out
            return [probe, s], outs

        # Simple statement (including nested def/class, async constructs): one node
        return [probe, s], [n]


def instrument_function(source: str, func_name: str) -> tuple[ast.Module, dict]:
    """
    Build the CFG of `func_name` and return (instrumented module AST, CFG metadata).
    The metadata is plain data so it can be sent between processes.
    """
    tree = ast.parse(source)
    func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name), None
    )
    if func is None:
        raise ValueError(f"no top-level function named '{func_name}'")
    builder = _CFGBuilder(source)
    builder.build(func)
    meta = {
        "nodes": builder.nodes,
        "edges": sorted(builder.edges),
        "loop_headers": sorted(builder.loop_headers),
    }
    return ast.fix_missing_locations(tree), meta


# ---------------------------------------------------------------------------
# Runtime: records one test path (list of statement-level node ids) per call
# ---------------------------------------------------------------------------

class PathRecorder:
    """Runtime side of the probes: installed into the instrumented code's namespace."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stack: list[dict] = []  # one frame per active call (handles recursion)
        self.paths: list[list[int]] = []  # completed calls, innermost first

    def install(self, namespace: dict):
        namespace[PROBE_NODE] = self._node
        namespace[PROBE_TEST] = self._test
        namespace[PROBE_ITER] = self._iter
        namespace[PROBE_ENTER] = self._enter
        namespace[PROBE_FAIL] = self._fail
        namespace[PROBE_EXIT] = self._exit

    def snapshot(self) -> list[list[int]]:
        return [list(p) for p in self.paths]

    def _node(self, nid):
        if self.stack:
            self.stack[-1]["path"].append(nid)

    def _test(self, nid, value):
        self._node(nid)
        return value

    def _iter(self, nid, iterable):
        for item in iterable:
            self._node(nid)
            yield item
        self._node(nid)

    def _enter(self):
        self.stack.append({"path": [], "failed": False})

    def _fail(self):
        if self.stack:
            self.stack[-1]["failed"] = True

    def _exit(self):
        if self.stack:
            frame = self.stack.pop()
            self.paths.append(frame["path"] + ([] if frame["failed"] else [EXIT]))


# ---------------------------------------------------------------------------
# Analysis: basic blocks, test requirements, touring
# ---------------------------------------------------------------------------

def _basic_blocks(meta: dict) -> tuple[dict, dict, set, dict]:
    """
    Group statement-level nodes into basic blocks.
    Returns (stmt id -> block id, block id -> label, block edges, block id -> stmt ids).
    Block ids are numbered 1..N in source order; the exit node gets the last id.
    """
    stmt_ids = sorted(int(k) for k in meta["nodes"]) + [EXIT]
    edges = [tuple(e) for e in meta["edges"]]
    succ = {n: [] for n in stmt_ids}
    pred = {n: [] for n in stmt_ids}
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    headers = set(meta["loop_headers"])
    entry = stmt_ids[0]

    def is_leader(n):
        if n in (entry, EXIT) or n in headers or len(pred[n]) != 1:
            return True
        p = pred[n][0]
        return len(succ[p]) != 1 or p in headers

    blocks = []
    for n in stmt_ids:
        if not is_leader(n):
            continue
        block, cur = [n], n
        while len(succ[cur]) == 1 and not is_leader(succ[cur][0]):
            cur = succ[cur][0]
            block.append(cur)
        blocks.append(block)
    blocks.sort(key=lambda b: (b[0] == EXIT, b[0]))

    stmt_to_block, members, labels = {}, {}, {}
    for bid, block in enumerate(blocks, start=1):
        members[bid] = block
        for n in block:
            stmt_to_block[n] = bid
        if block == [EXIT]:
            labels[bid] = "exit"
            continue
        infos = [meta["nodes"][n] if n in meta["nodes"] else meta["nodes"][str(n)] for n in block]
        first, last = infos[0]["line"], infos[-1]["line"]
        where = f"line {first}" if first == last else f"lines {first}-{last}"
        text = "; ".join(i["text"] for i in infos)
        labels[bid] = f"{where}: {text if len(text) <= 100 else text[:97] + '...'}"

    block_edges = set()
    for a, b in edges:
        ba, bb = stmt_to_block[a], stmt_to_block[b]
        if ba != bb or members[bb][0] == b:
            block_edges.add((ba, bb))
    return stmt_to_block, labels, block_edges, members


def _reachable(entry: int, succ: dict) -> set:
    seen, todo = {entry}, [entry]
    while todo:
        for s in succ.get(todo.pop(), ()):
            if s not in seen:
                seen.add(s)
                todo.append(s)
    return seen


def _prime_paths(nodes: list, succ: dict, pred: dict) -> list[tuple]:
    """All prime paths: simple paths that are not a proper sub-path of another simple path."""
    simple, stack = set(), [(n,) for n in nodes]
    while stack:
        path = stack.pop()
        if path in simple:
            continue
        simple.add(path)
        if len(simple) > MAX_SIMPLE_PATHS:
            raise OverflowError(f"more than {MAX_SIMPLE_PATHS} simple paths")
        if len(path) > 1 and path[0] == path[-1]:
            continue  # a simple cycle cannot be extended
        for s in succ[path[-1]]:
            if s not in path or s == path[0]:
                stack.append(path + (s,))

    def extendable(path):
        if len(path) > 1 and path[0] == path[-1]:
            return False
        return any(path + (s,) in simple for s in succ[path[-1]]) or any(
            (r,) + path in simple for r in pred[path[0]]
        )

    return sorted((p for p in simple if not extendable(p)), key=lambda p: (len(p), p))


def _tours(test_path: list, requirement: tuple) -> bool:
    """True if the requirement is a contiguous sub-path of the test path (direct touring)."""
    k = len(requirement)
    return any(tuple(test_path[i:i + k]) == requirement for i in range(len(test_path) - k + 1))


def analyze(meta: dict, traces: dict, target: str) -> dict:
    """
    traces: test id -> list of recorded statement-level paths (one per call, innermost first).
    Returns the CFG, the coverage of all four criteria, and each test's path / covered requirements.
    """
    stmt_to_block, labels, block_edges, members = _basic_blocks(meta)
    leaders = {block[0] for block in members.values()}
    exit_block = stmt_to_block[EXIT]

    succ = {b: [] for b in labels}
    pred = {b: [] for b in labels}
    for a, b in sorted(block_edges):
        succ[a].append(b)
        pred[b].append(a)
    reach = _reachable(1, succ)
    nodes = sorted(reach)
    edges = sorted((a, b) for a, b in block_edges if a in reach and b in reach)
    succ_r = {n: [s for s in succ[n] if s in reach] for n in nodes}
    pred_r = {n: [p for p in pred[n] if p in reach] for n in nodes}

    # Test requirements
    requirements: dict[str, list] = {"node": [(n,) for n in nodes], "edge": edges}
    pairs = [(a, b, c) for a, b in edges for c in succ_r[b]]
    in_pair = {(p[0], p[1]) for p in pairs} | {(p[1], p[2]) for p in pairs}
    epc = pairs + [e for e in edges if e not in in_pair]
    if not edges:
        epc = [(n,) for n in nodes]
    requirements["edge_pair"] = sorted(set(epc))
    prime_error = None
    try:
        requirements["prime_path"] = _prime_paths(nodes, succ_r, pred_r)
    except OverflowError as e:
        requirements["prime_path"] = None
        prime_error = f"Prime paths not computed: {e}"

    # Test paths (block level): a block is entered whenever its first statement executes
    def to_blocks(stmt_path):
        return [stmt_to_block[n] for n in stmt_path if n in leaders and n in stmt_to_block]

    per_test, all_paths = {}, []
    for test_id, stmt_paths in traces.items():
        paths = [to_blocks(p) for p in stmt_paths]
        paths = [p for p in paths if p]
        if not paths:
            continue
        all_paths.extend(paths)
        target_reqs = requirements.get(target) or []
        per_test[test_id] = {
            "path": paths[-1],                 # the call made by the test itself
            "recursive_call_paths": paths[:-1][:MAX_LISTED_RECURSIVE_PATHS],  # nested recursive calls
            "covers": [list(r) for r in target_reqs if any(_tours(p, r) for p in paths)],
        }

    criteria = {}
    for crit in CRITERIA:
        reqs = requirements[crit]
        if reqs is None:
            criteria[crit] = {"error": prime_error}
            continue
        uncovered = [list(r) for r in reqs if not any(_tours(p, r) for p in all_paths)]
        total = len(reqs)
        criteria[crit] = {
            "covered": total - len(uncovered),
            "total": total,
            "percent": round(100.0 * (total - len(uncovered)) / total, 1) if total else None,
            "uncovered": uncovered,
        }

    target_result = criteria[target]
    return {
        "target": target,
        "criterion_met": "error" not in target_result and not target_result["uncovered"],
        "criteria": criteria,
        "cfg": {
            "nodes": {str(b): labels[b] for b in sorted(labels) if b in reach},
            "edges": [list(e) for e in edges],
            "initial_node": 1,
            "final_node": exit_block,
        },
        "per_test": per_test,
    }
