# Working on this repo (team guide)

Read `CLAUDE.md` first (rules), then `docs/ARCHITECTURE.md`.

## Ownership
| Area | Owner | Notes |
|---|---|---|
| `src/io/mps_reader.*` | MPS-reader teammate | Implement `ps26119::io::read_mps` (declared in `src/io/mps_reader.h`). CMake compiles `mps_reader.cpp` automatically once it exists; the CLI then accepts `.mps`. |
| `src/gpu/` + GPU runs | GPU teammate | Run `scripts/gpu_check.sh`, commit `bench/results/`. |
| everything else | core | |

## Workflow
1. `git pull` on `main`, then branch: `git switch -c feat/<short-name>`.
2. Small steps, one feature per branch; add tests with every change; never weaken a test.
3. Before pushing: `cmake --build build -j && ctest --test-dir build` must pass.
4. `git push -u origin feat/<short-name>` and open a pull request; CI (Ubuntu + macOS) must be green.
5. Merge with "squash and merge".

## MPS reader acceptance test (for the reader author)
Two independent readers must produce the same Model. When `mps_reader.cpp` lands, add a
test that, for every `data/netlib_small/*.mps`, `read_mps(...).fingerprint_hex()` equals the
`# fingerprint` line in the matching `.lpm` (written by the highspy bridge). The fingerprint
ignores names and the order of entries inside a column; see `Model::fingerprint()`.

## Benchmarks and claims
- Never quote a number that is not in a committed CSV under `bench/results/` (file name
  contains the git hash). `python3 bench/make_evidence.py` regenerates `docs/EVIDENCE.md`.
- Do not rebuild `build/` while a benchmark is running from it (the CSV would mix code
  versions). Benchmark from a clean commit.
- On GPU machines: `PS26119_MACHINE=<name> scripts/gpu_check.sh`.

## Commit messages
Imperative, one line summary, blank line, details. Mention the milestone (M0/M1/M2...).
