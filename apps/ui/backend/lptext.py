"""lptext.py — a model typed by hand on the Solve page, in a small algebraic language close to the
CPLEX LP file format, parsed by our own code into the Model contract (CLAUDE.md §6) and written as
a .lpm file, so the solver, the verifier and every reference solver read exactly the same numbers.

    maximize  profit: 3 x1 + 5 x2          (max / maximise / min / minimize …; a name is optional)
    subject to                              (st / s.t. / such that)
      plant1:  x1 <= 4                      (<=, >=, =; also =<, =>, <, >; names are optional)
      2 x2 <= 12
      3 x1 + 2 x2 <= 18
      -5 <= x1 - x2 <= 5                    (a ranged row)
      x1 + 2 <= 3 x2                        (terms and constants on both sides)
    bounds                                  (default 0 <= x < +inf)
      x1 <= 100 ;  -inf <= x3 <= 10 ;  x4 free ;  2 <= x5 <= 8 ;  x6 = 1
    integer   x1 x2                         (also: general / generals / int)
    binary    y                             (also: binaries / bin; bounds become [0, 1])
    end

Coefficients may be written 3x, 3 x, 3*x, 1.5e3 x, -x; comments start with \\ # or // (or /* … */); ';' separates statements on one line; a statement may span lines
(a new one starts at 'name:', at a keyword, or on a new line once the previous one is complete).
lp_solve style is accepted too: 'max: 3x + 2y;  c1: x + y <= 4;  int x, y;'. Keywords cannot be
variable names. Errors carry the line number.
"""
from __future__ import annotations

import math
import re

INF = math.inf
_KEY = {
    "max": "max", "maximize": "max", "maximise": "max", "maximum": "max",
    "min": "min", "minimize": "min", "minimise": "min", "minimum": "min",
    "st": "st", "s.t.": "st", "subject": "st", "such": "st",
    "bounds": "bounds", "bound": "bounds",
    "integer": "int", "integers": "int", "general": "int", "generals": "int", "gen": "int", "int": "int",
    "binary": "bin", "binaries": "bin", "bin": "bin",
    "end": "end",
}
_TOKEN = re.compile(r"""
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<inf>[+-]?(?:inf|infinity)\b)
  | (?P<op><=|>=|=<|=>|==|<|>|=)
  | (?P<name>[A-Za-z_][A-Za-z0-9_.\[\]]*)
  | (?P<sign>[+-])
  | (?P<star>\*)
  | (?P<colon>:)
  | (?P<semi>;)
  | (?P<comma>,)
  | (?P<ws>\s+)
""", re.X | re.I)


class LpTextError(ValueError):
    def __init__(self, line: int, msg: str):
        super().__init__(f"line {line}: {msg}")
        self.line = line


def _tokens(text: str):
    """[(kind, value, line)] without comments and blanks."""
    out = []
    text = re.sub(r"/\*.*?\*/", lambda mm: "\n" * mm.group().count("\n"), text, flags=re.S)  # /* block comments */
    for ln, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"\\|#|//", raw, maxsplit=1)[0]
        low = line.strip().lower()
        # multi-word keywords first
        for phrase, key in (("subject to", "st"), ("such that", "st"), ("s.t.", "st")):
            if low.startswith(phrase):
                out.append(("key", key, ln))
                line = line.strip()[len(phrase):]
                break
        pos = 0
        while pos < len(line):
            m = _TOKEN.match(line, pos)
            if not m:
                raise LpTextError(ln, f"unexpected character {line[pos]!r}")
            pos = m.end()
            kind = m.lastgroup
            val = m.group(kind)
            if kind == "ws":
                continue
            colon_next = line[pos:].lstrip().startswith(":")
            if kind == "name" and val.lower() in _KEY and _KEY[val.lower()] in ("max", "min") and colon_next:
                pos = line.index(":", pos) + 1  # lp_solve style 'max: 3x + 2y;'
                out.append(("key", _KEY[val.lower()], ln))
                continue
            if kind == "name" and val.lower() in _KEY and not colon_next:
                if val.lower() == "subject" or val.lower() == "such":  # 'subject to' split across tokens
                    nxt = re.match(r"\s*(to|that)\b", line[pos:], re.I)
                    if nxt:
                        pos += nxt.end()
                out.append(("key", _KEY[val.lower()], ln))
                continue
            if kind == "name" and val.lower() == "free":
                out.append(("free", val, ln))
                continue
            out.append((kind, val, ln))
    return out


def _statements(toks):
    """Split one section's tokens into statements: a new one begins at 'name :' or when a complete
    comparison is followed by a term that is not part of a chained comparison."""
    stmts, cur = [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t[0] == "name" and i + 1 < len(toks) and toks[i + 1][0] == "colon":
            if cur:
                stmts.append(cur)
            cur = [t, toks[i + 1]]
            i += 2
            continue
        if t[0] == "semi":
            if cur:
                stmts.append(cur)
            cur = []
            i += 1
            continue
        if cur and t[2] != cur[-1][2] and _complete(cur) and t[0] != "op":
            stmts.append(cur)
            cur = []
        cur.append(t)
        i += 1
    if cur:
        stmts.append(cur)
    return stmts


def _complete(stmt) -> bool:
    """A statement with at least one comparison and something after its last comparison."""
    ops = [k for k, t in enumerate(stmt) if t[0] == "op"]
    return bool(ops) and ops[-1] < len(stmt) - 1


def _linear(toks, ln):
    """Parse 'terms' → ({var: coef}, constant, order-of-first-appearance)."""
    coefs, const, order = {}, 0.0, []
    i, n = 0, len(toks)
    if n == 0:
        raise LpTextError(ln, "empty expression")
    while i < n:
        sign = 1.0
        while i < n and toks[i][0] == "sign":
            sign *= -1.0 if toks[i][1] == "-" else 1.0
            i += 1
        if i >= n:
            raise LpTextError(ln, "expression ends with a sign")
        if toks[i][0] == "inf":
            raise LpTextError(toks[i][2], "infinity is only allowed as a bound")
        num = None
        if toks[i][0] == "num":
            num = float(toks[i][1])
            i += 1
            if i < n and toks[i][0] == "star":
                i += 1
                if i >= n or toks[i][0] != "name":
                    raise LpTextError(ln, "'*' must be followed by a variable")
        if i < n and toks[i][0] == "name":
            v = toks[i][1]
            if v not in coefs:
                coefs[v] = 0.0
                order.append(v)
            coefs[v] += sign * (num if num is not None else 1.0)
            i += 1
        elif num is not None:
            const += sign * num
        else:
            raise LpTextError(toks[i][2], f"expected a number or a variable, found {toks[i][1]!r}")
        if i < n and toks[i][0] != "sign":
            raise LpTextError(toks[i][2], f"expected '+' or '-' before {toks[i][1]!r}")
    return coefs, const, order


_OPS = {"<=": "<=", "=<": "<=", "<": "<=", ">=": ">=", "=>": ">=", ">": ">=", "=": "=", "==": "="}


def parse(text: str, name: str = "typed"):
    from lpm import PyModel
    toks = _tokens(text)
    sections: list[tuple[str, int, list]] = []
    for t in toks:
        if t[0] == "key":
            sections.append((t[1], t[2], []))
        else:
            if not sections:
                raise LpTextError(t[2], "start with 'maximize' or 'minimize'")
            sections[-1][2].append(t)
    # lp_solve style: 'max: … ;' followed directly by constraints — what follows the objective's ';'
    if sections and sections[0][0] in ("max", "min") and not any(sec[0] == "st" for sec in sections):
        body = sections[0][2]
        semi = next((k for k, t in enumerate(body) if t[0] == "semi"), None)
        if semi is not None and semi + 1 < len(body):
            sections.insert(1, ("st", body[semi + 1][2], body[semi + 1:]))
            sections[0] = (sections[0][0], sections[0][1], body[:semi])
    if not sections or sections[0][0] not in ("max", "min"):
        raise LpTextError(sections[0][1] if sections else 1, "start with 'maximize' or 'minimize'")
    if sum(1 for s in sections if s[0] in ("max", "min")) > 1:
        raise LpTextError(next(s[1] for s in sections[1:] if s[0] in ("max", "min")), "only one objective")

    col_index: dict[str, int] = {}
    cols: list[str] = []

    def col(v: str) -> int:
        if v not in col_index:
            col_index[v] = len(cols)
            cols.append(v)
        return col_index[v]

    sense = -1 if sections[0][0] == "max" else 1
    otoks = [t for t in sections[0][2] if t[0] != "semi"]
    if len(otoks) >= 2 and otoks[0][0] == "name" and otoks[1][0] == "colon":
        otoks = otoks[2:]
    if any(t[0] == "op" for t in otoks):
        raise LpTextError(sections[0][1], "the objective has no comparison; put constraints after 'subject to'")
    ocoef, offset, oorder = _linear(otoks, sections[0][1]) if otoks else ({}, 0.0, [])
    for v in oorder:
        col(v)

    rows = []  # (name, {col: coef}, lower, upper, line)
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    integer: set[int] = set()
    binary: set[int] = set()
    seen_end = False
    for kind, ln, body in sections[1:]:
        if seen_end:
            raise LpTextError(ln, "text after 'end'")
        if kind == "end":
            seen_end = True
            if body:
                raise LpTextError(body[0][2], "text after 'end'")
            continue
        if kind in ("int", "bin"):
            for t in body:
                if t[0] in ("comma", "semi"):
                    continue
                if t[0] != "name":
                    raise LpTextError(t[2], f"'{'integer' if kind == 'int' else 'binary'}' lists variable names, found {t[1]!r}")
                (integer if kind == "int" else binary).add(col(t[1]))
            continue
        for stmt in _statements(body):
            sln = stmt[0][2]
            rname = ""
            if len(stmt) >= 2 and stmt[0][0] == "name" and stmt[1][0] == "colon":
                rname, stmt = stmt[0][1], stmt[2:]
            if kind == "bounds" and len(stmt) == 2 and stmt[0][0] == "name" and stmt[1][0] == "free":
                j = col(stmt[0][1])
                lower[j], upper[j] = -INF, INF
                continue
            parts, ops = [[]], []
            for t in stmt:
                if t[0] == "op":
                    ops.append(_OPS[t[1]])
                    parts.append([])
                else:
                    parts[-1].append(t)
            if not ops:
                raise LpTextError(sln, "expected a comparison (<=, >= or =)")
            if len(ops) > 2 or (len(ops) == 2 and (ops[0] != ops[1] or ops[0] == "=")):
                raise LpTextError(sln, "a ranged row is written  a <= expression <= b  (or with >=)")

            def value(p):
                if len(p) == 1 and p[0][0] == "inf":
                    return -INF if p[0][1].startswith("-") else INF
                if len(p) == 2 and p[0][0] == "sign" and p[1][0] == "inf":
                    return -INF if p[0][1] == "-" else INF
                c, k, _ = _linear(p, sln)
                if any(abs(x) > 0 for x in c.values()):
                    return None
                return k

            if len(ops) == 2:  # a <= expr <= b   or   a >= expr >= b
                lo, hi = value(parts[0]), value(parts[2])
                if lo is None or hi is None:
                    raise LpTextError(sln, "the ends of a ranged row must be numbers")
                c, k, order = _linear(parts[1], sln)
                if ops[0] == ">=":
                    lo, hi = hi, lo
                lo, hi = lo - k, hi - k
                if lo > hi:
                    raise LpTextError(sln, f"empty range [{lo:g}, {hi:g}]")
                op = "range"
            else:
                lhs_c, rhs_c = value(parts[0]), value(parts[1])
                if lhs_c is not None and rhs_c is not None:
                    raise LpTextError(sln, "a comparison needs at least one variable")
                if lhs_c is not None:  # number op expr  →  expr op' number
                    c, k, order = _linear(parts[1], sln)
                    op = {"<=": ">=", ">=": "<=", "=": "="}[ops[0]]
                    b = lhs_c
                elif rhs_c is not None:
                    c, k, order = _linear(parts[0], sln)
                    op, b = ops[0], rhs_c
                else:  # expressions on both sides
                    cl, kl, ol = _linear(parts[0], sln)
                    cr, kr, orr = _linear(parts[1], sln)
                    c = dict(cl)
                    for v, a in cr.items():
                        c[v] = c.get(v, 0.0) - a
                    order = ol + [v for v in orr if v not in cl]
                    k, op, b = kl - kr, ops[0], 0.0
                if math.isinf(b) and op == "=":
                    raise LpTextError(sln, "'= infinity' is not a constraint")
                b -= k
                lo, hi = {"<=": (-INF, b), ">=": (b, INF), "=": (b, b)}[op]
            if kind == "bounds":
                if len(c) != 1 or next(iter(c.values())) != 1.0:
                    raise LpTextError(sln, "a bound is on one variable with coefficient 1 (put other rows under 'subject to')")
                j = col(next(iter(c)))
                if op in ("range", ">=", "="):
                    lower[j] = lo
                if op in ("range", "<=", "="):
                    upper[j] = hi
                continue
            if kind != "st":
                raise LpTextError(sln, f"unexpected statement in section '{kind}'")
            for v in order:
                col(v)
            rows.append((rname, {col(v): a for v, a in c.items() if a != 0.0}, lo, hi, sln))
    if not any(sec[0] == "st" for sec in sections):
        raise LpTextError(sections[0][1], "no constraints: add a 'subject to' section")
    if not rows:
        raise LpTextError(sections[0][1], "no constraints under 'subject to'")

    n, m = len(cols), len(rows)
    for j in binary:
        lower.setdefault(j, 0.0)
        upper[j] = min(upper.get(j, 1.0), 1.0)
        lower[j] = max(lower[j], 0.0)
        integer.add(j)
    names = [r[0] for r in rows]
    used = set(x for x in names if x)
    for i, nm in enumerate(names):
        if not nm:
            k = i + 1
            while f"c{k}" in used:
                k += 1
            names[i] = f"c{k}"
            used.add(names[i])
    if len(set(names)) != len(names):
        dup = next(x for x in names if names.count(x) > 1)
        raise LpTextError([r[4] for r in rows if r[0] == dup][1], f"row name {dup!r} is used twice")
    if set(names) & set(cols):
        clash = sorted(set(names) & set(cols))[0]
        raise LpTextError(next(r[4] for r in rows if r[0] == clash), f"{clash!r} names both a row and a variable")

    M = PyModel()
    M.name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:40] or "typed"
    M.num_rows, M.num_cols, M.sense, M.obj_offset = m, n, sense, offset
    M.obj = [ocoef.get(v, 0.0) for v in cols]
    M.col_lower = [lower.get(j, 0.0) for j in range(n)]
    M.col_upper = [upper.get(j, INF) for j in range(n)]
    M.row_lower = [r[2] for r in rows]
    M.row_upper = [r[3] for r in rows]
    percol: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        for j, a in sorted(r[1].items()):
            percol[j].append((i, a))
    M.col_start, M.row_index, M.value = [0], [], []
    for j in range(n):
        for i, a in percol[j]:
            M.row_index.append(i)
            M.value.append(a)
        M.col_start.append(len(M.value))
    M.is_integer = [1 if j in integer else 0 for j in range(n)] if integer else []
    M.row_names, M.col_names = names, list(cols)
    return M


def summary(M) -> dict:
    return {"rows": M.num_rows, "cols": M.num_cols, "nnz": M.nnz, "integer": sum(M.is_integer) if M.is_integer else 0,
            "sense": "maximize" if M.sense == -1 else "minimize"}


EXAMPLES = {
    "wyndor": """\\ Wyndor Glass (Hillier & Lieberman): two products, three plants
maximize
  profit: 3 doors + 5 windows
subject to
  plant1: doors <= 4
  plant2: 2 windows <= 12
  plant3: 3 doors + 2 windows <= 18
end
""",
    "blend": """\\ a small crude blend: cost per barrel, sulphur and density limits, demand
minimize
  cost: 72 arab_light + 68 arab_heavy + 81 bonny_light + 64 basrah
subject to
  demand:   arab_light + arab_heavy + bonny_light + basrah = 100
  sulphur:  1.8 arab_light + 2.9 arab_heavy + 0.2 bonny_light + 2.6 basrah <= 180
  api_low:  33 arab_light + 27 arab_heavy + 35 bonny_light + 30 basrah >= 3100
  api_high: 33 arab_light + 27 arab_heavy + 35 bonny_light + 30 basrah <= 3300
bounds
  arab_light <= 60
  bonny_light <= 30
  10 <= basrah <= 50
end
""",
    "knapsack": """\\ a 0-1 knapsack with a side constraint (MILP)
maximize
  value: 12 a + 9 b + 7 c + 15 d + 6 e
subject to
  weight: 4 a + 3 b + 2 c + 6 d + 2 e <= 11
  pick:   a + b + c + d + e <= 3
binary
  a b c d e
end
""",
}
