# Interfaces and file formats

The contracts a user or another program relies on. Code references are authoritative
(`include/ps26119/solution.h`, `include/ps26119/ps26119.h`, `src/io/solution_writer.h`,
`tools/lpm.py`); this page collects them in one place.

## Status

| Status | Meaning | What backs it |
|---|---|---|
| `Optimal` | x, y, z returned | LP: re-checked on the original model (primal, dual, gap at the verifier's 1e-6, `check`) plus a rounding-proof bound; MILP: the point re-checked (bounds, rows, integrality) and the gap closed by certified node bounds |
| `Infeasible` | no feasible point exists | LP: a Farkas certificate (`dual_ray`) checked on the original model (or crossed bounds); MILP: exhausted branch-and-bound tree (no certificate) |
| `Unbounded` | the objective improves without limit | a feasible point `x` plus a ray `primal_ray`, checked on the original model (MILP: integer point) |
| `IterationLimit`, `TimeLimit` | stopped at the caller's limit | the best iterate / incumbent is reported, never labelled optimal |
| `NumericalError` | an engine or a check failed | includes every claim whose check failed ("Optimal withdrawn", "Infeasible withdrawn") |
| `NotSolved` | not attempted or refused | invalid model or options, no CUDA backend for `--gpu`, out of memory |

`check`: `PASS` / `FAIL` / empty (not applicable or no certificate returned). A first-order run
with a tolerance looser than the verifier's may report `Optimal` with `check FAIL` — at the
requested tolerance, stated in the message.

## CLI

`ps26119 solve <model.mps|model.lpm> [options]` prints one `key value` line per field:
`model`, `status`, `engine`, `objective`, `residuals`, `check`, `certified` (rounding-proof bound
on the optimum; not printed for Infeasible/Unbounded), `iterations`, `to 1e-4`, `message`.
`ps26119 --help` lists every option.

| Exit code | Meaning |
|---|---|
| 0 | Optimal |
| 1 | Infeasible, Unbounded, IterationLimit, TimeLimit |
| 2 | usage error (unknown option, malformed number, …) |
| 3 | read error (model or warm-start file), with the line number where known |
| 4 | the output file cannot be written (checked before solving) |
| 5 | NumericalError, NotSolved (incl. out of memory) |

## Solution file (`--out`, read by `tools/verify.py` and `--warm`)

```
PS26119-SOLUTION 1
model <fingerprint>            FNV-1a hash of every number of the model (Model::fingerprint)
name <model name>
status <Status>                engine <name>      precision fp64|mixed|dd
objective <value>              dual_objective, certified_bound, primal_residual, dual_residual,
gap, iterations, seconds, setup_seconds, primal_weight, iterations_to_fast, seconds_to_fast
check <PASS|FAIL ...>          message <free text>
COLUMNS <n>
<j> <x_j> <z_j> <name>         %.17g: values round-trip exactly
ROWS <m>
<i> <activity_i> <y_i> <name>
DUAL_RAY <m>                   optional (Infeasible): <index> <value> per line
PRIMAL_RAY <n>                 optional (Unbounded):  <index> <value> per line
END
```
Signs: z = c − Aᵀy (HiGHS convention). The reader requires every number to parse completely,
validates block counts, and never allocates from a count before the lines exist.

## Ranging file (`solve --ranging <file>`)

Cost and right-hand-side ranging at the optimal vertex (`include/ps26119/ranging.h`,
`src/core/ranging.cpp`): the basis is rebuilt from the solution, factorised, and checked for
primal and dual feasibility before any range is computed. LPs only, vertex answers only (simplex,
oracle — an interior first-order answer is refused), up to 25,000 rows. When ranging is refused
the CLI prints `ranging    refused: <reason>` and writes no file; the solve result is unaffected.

```
kind,index,name,value,status,lower,upper,dual_or_reduced_cost
cost,<j>,<name>,<c_j>,basic|nonbasic,<lower>,<upper>,<z_j>       c_j may move within [lower, upper]
rhs,<i>,<name>,<b_i>,upper_binding|lower_binding,<lower>,<upper>,<y_i>   the binding bound b_i may move within [lower, upper]
rhs,<i>,<name>,<b_i>,not_binding,<activity>,inf,<y_i>          upper bound b_i: down to the activity, up freely
rhs,<i>,<name>,<b_i>,not_binding,-inf,<activity>,<y_i>         (no finite upper bound) lower bound b_i
rhs,<i>,<name>,<activity>,not_binding,-inf,inf,<y_i>           free row
```
A not-binding row with both bounds finite is reported on its upper bound (its lower bound may
likewise rise up to the activity). An equality row whose slack is basic at a degenerate vertex
is `upper_binding` with the one-point range `[b_i, b_i]`.
Within a range the optimal basis stays the same: a cost change moves the objective by Δc·x_j, a
bound change by y_i·Δb. Ranges are in the model's own objective sense; under primal degeneracy a
range can be one-sided at the current value (the CLI line says how many basic variables sit at a
bound). `%.17g`; infinite ends are written `inf` / `-inf`.

## `.lpm` model format

Exact text form of the Model contract (CLAUDE.md §6), written by `tools/mps_to_lpm.py` and the
benchmark generators; full specification in the `tools/lpm.py` docstring. Sections: header
`LPM 1`, `NAME`, `SENSE MIN|MAX`, `ROWS m`, `COLS n`, `NNZ k`, `OFFSET`, then `OBJ`,
`COL_LOWER`, `COL_UPPER`, `ROW_LOWER`, `ROW_UPPER`, `COL_START`, `ROW_INDEX`, `VALUE`, optional
`INTEGER`, `ROW_NAMES`, `COL_NAMES`, and `END`. `inf` / `-inf` for infinite bounds.

## C API and Python

`include/ps26119/ps26119.h`: `ps26119_solve_lp`, `ps26119_solve_lp_ex` (adds `dual_ray` /
`primal_ray` outputs), `ps26119_solve_mps`. No exceptions cross the API; invalid arguments
(including negative or NaN limits, unknown algorithm or precision, non-finite model data or warm
start) return `PS26119_INVALID_ARGUMENT` / `PS26119_NOT_SOLVED` with a message. Python
(`python/ps26119`): `solve_lp` returns a `Result` with the same fields plus `x, y, z, dual_ray,
primal_ray`; `solve_mps` returns the scalar fields and writes the vectors to the solution file
(`out=`); unknown names raise `ValueError`. `solve()` may be called from
several threads at once.
