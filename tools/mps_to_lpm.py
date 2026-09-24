#!/usr/bin/env python3
"""mps_to_lpm.py — TEMPORARY bridge: MPS -> our .lpm text format, using highspy.

Test tooling only (CLAUDE.md §2). The solver never links highspy. Later this bridge
cross-checks the teammate's C++ MPS reader: both must produce the same Model
fingerprint (the fingerprint is written into the .lpm header as a comment).

usage: mps_to_lpm.py in.mps [out.lpm]      (default out = in with .lpm suffix)
       mps_to_lpm.py --fingerprint in.mps  (print the fingerprint only)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpm import fingerprint, read_mps_highspy, write_lpm  # noqa: E402


def main(argv):
    if len(argv) >= 2 and argv[0] == "--fingerprint":
        print(fingerprint(read_mps_highspy(argv[1])))
        return 0
    if not argv or len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    src = argv[0]
    dst = argv[1] if len(argv) == 2 else os.path.splitext(src)[0] + ".lpm"
    m = read_mps_highspy(src)
    write_lpm(m, dst, source=os.path.basename(src))
    print(f"{src} -> {dst}: {m.num_rows} rows, {m.num_cols} cols, {m.nnz} nnz, fingerprint {fingerprint(m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
