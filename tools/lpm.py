"""lpm.py — shared Python helpers for the ps26119 tooling (NOT part of the solver).

* PyModel: the Model contract of CLAUDE.md §6 as a Python object.
* read_mps_highspy(): read an MPS file with highspy (an independent reader).
* write_lpm() / read_lpm(): our `.lpm` text format (spec below).
* fingerprint(): the same 64-bit FNV-1a hash as Model::fingerprint() in src/core/model.cpp.
* read_solution(): parse a solution file written by src/io/solution_writer.cpp.

.lpm format, version 1 (plain text, whitespace separated, '#' starts a comment line)
-----------------------------------------------------------------------------------
    LPM 1
    NAME <name>                      (rest of line)
    SENSE MIN|MAX
    ROWS <m>
    COLS <n>
    NNZ <nnz>
    OFFSET <obj_offset>
    OBJ         n numbers            c
    COL_LOWER   n numbers            (-inf allowed)
    COL_UPPER   n numbers            (inf allowed)
    ROW_LOWER   m numbers
    ROW_UPPER   m numbers
    COL_START   n+1 integers         CSC column pointers
    ROW_INDEX   nnz integers         CSC row indices (0-based)
    VALUE       nnz numbers          CSC values
    INTEGER     n integers (0/1)     optional section; absent = all continuous
    ROW_NAMES   m lines, one name per line (names may contain spaces)
    COL_NAMES   n lines
    END
Numbers use Python repr / C %.17g, so doubles round-trip exactly; infinities are
written `inf` / `-inf`. Numeric sections may wrap across lines freely.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

INF = math.inf


@dataclass
class PyModel:
    name: str = ""
    num_rows: int = 0
    num_cols: int = 0
    sense: int = 1
    obj_offset: float = 0.0
    obj: list = field(default_factory=list)
    col_lower: list = field(default_factory=list)
    col_upper: list = field(default_factory=list)
    row_lower: list = field(default_factory=list)
    row_upper: list = field(default_factory=list)
    col_start: list = field(default_factory=list)
    row_index: list = field(default_factory=list)
    value: list = field(default_factory=list)
    is_integer: list = field(default_factory=list)
    row_names: list = field(default_factory=list)
    col_names: list = field(default_factory=list)

    @property
    def nnz(self) -> int:
        return len(self.value)


# --------------------------------------------------------------------------- fingerprint
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = 0xFFFFFFFFFFFFFFFF


class _Fnv:
    def __init__(self) -> None:
        self.h = _FNV_OFFSET
        self.buf = bytearray()

    def u64(self, v: int) -> None:
        self.buf += struct.pack("<Q", v & _MASK)

    def i64(self, v: int) -> None:
        self.buf += struct.pack("<q", v)

    def f64(self, v: float) -> None:
        if v == 0.0:
            v = 0.0
        if math.isnan(v):
            self.u64(0x7FF8000000000000)
        else:
            self.buf += struct.pack("<d", v)

    def f64s(self, vs) -> None:
        self.u64(len(vs))
        for v in vs:
            self.f64(v)

    def digest(self) -> int:
        h = self.h
        for b in self.buf:
            h ^= b
            h = (h * _FNV_PRIME) & _MASK
        return h


def fingerprint(m: PyModel) -> str:
    """Identical to Model::fingerprint_hex() in src/core/model.cpp."""
    f = _Fnv()
    f.i64(m.num_rows)
    f.i64(m.num_cols)
    f.i64(m.sense)
    f.f64(m.obj_offset)
    f.f64s(m.obj)
    f.f64s(m.col_lower)
    f.f64s(m.col_upper)
    f.f64s(m.row_lower)
    f.f64s(m.row_upper)
    f.u64(len(m.value))
    for j in range(m.num_cols):
        col = sorted(
            ((m.row_index[k], m.value[k]) for k in range(m.col_start[j], m.col_start[j + 1])),
            key=lambda t: t[0],
        )
        f.i64(len(col))
        for i, v in col:
            f.i64(i)
            f.f64(v)
    for j in range(m.num_cols):
        f.i64(1 if (j < len(m.is_integer) and m.is_integer[j]) else 0)
    return "%016x" % f.digest()


# --------------------------------------------------------------------------- MPS via highspy
def read_mps_highspy(path: str) -> PyModel:
    """Read an MPS file with highspy and return it in our Model contract (CLAUDE.md §6)."""
    import highspy  # tooling only — never used by the solver

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    st = h.readModel(path)
    if st == highspy.HighsStatus.kError:
        raise RuntimeError(f"highspy could not read {path}")
    lp = h.getLp()
    a = lp.a_matrix_
    if a.format_ != highspy.MatrixFormat.kColwise:
        raise RuntimeError("highspy returned a row-wise matrix; expected column-wise")
    m = PyModel()
    m.name = lp.model_name_ or path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m.num_rows = lp.num_row_
    m.num_cols = lp.num_col_
    m.sense = -1 if lp.sense_ == highspy.ObjSense.kMaximize else 1
    m.obj_offset = float(lp.offset_)
    m.obj = [float(v) for v in lp.col_cost_]
    m.col_lower = [float(v) for v in lp.col_lower_]
    m.col_upper = [float(v) for v in lp.col_upper_]
    m.row_lower = [float(v) for v in lp.row_lower_]
    m.row_upper = [float(v) for v in lp.row_upper_]
    m.col_start = [int(v) for v in a.start_]
    m.row_index = [int(v) for v in a.index_]
    m.value = [float(v) for v in a.value_]
    nnz = m.col_start[-1] if m.col_start else 0
    m.row_index = m.row_index[:nnz]
    m.value = m.value[:nnz]
    if len(lp.integrality_) == m.num_cols:
        m.is_integer = [1 if v != highspy.HighsVarType.kContinuous else 0 for v in lp.integrality_]
        if not any(m.is_integer):
            m.is_integer = []
    m.row_names = list(lp.row_names_) if len(lp.row_names_) == m.num_rows else []
    m.col_names = list(lp.col_names_) if len(lp.col_names_) == m.num_cols else []
    return m


# --------------------------------------------------------------------------- .lpm I/O
def _fmt(v: float) -> str:
    if v == INF:
        return "inf"
    if v == -INF:
        return "-inf"
    return repr(float(v))


def _write_numbers(out, vals, per_line: int = 8) -> None:
    for k in range(0, len(vals), per_line):
        out.write(" ".join(_fmt(v) if isinstance(v, float) else str(v) for v in vals[k : k + per_line]))
        out.write("\n")


def write_lpm(m: PyModel, path: str, source: str = "") -> None:
    with open(path, "w") as out:
        out.write("LPM 1\n")
        if source:
            out.write(f"# source {source}\n")
        out.write(f"# fingerprint {fingerprint(m)}\n")
        out.write(f"NAME {m.name}\n")
        out.write(f"SENSE {'MAX' if m.sense == -1 else 'MIN'}\n")
        out.write(f"ROWS {m.num_rows}\nCOLS {m.num_cols}\nNNZ {m.nnz}\n")
        out.write(f"OFFSET {_fmt(m.obj_offset)}\n")
        for tag, vals in (
            ("OBJ", m.obj),
            ("COL_LOWER", m.col_lower),
            ("COL_UPPER", m.col_upper),
            ("ROW_LOWER", m.row_lower),
            ("ROW_UPPER", m.row_upper),
        ):
            out.write(tag + "\n")
            _write_numbers(out, [float(v) for v in vals])
        out.write("COL_START\n")
        _write_numbers(out, [int(v) for v in m.col_start], 16)
        out.write("ROW_INDEX\n")
        _write_numbers(out, [int(v) for v in m.row_index], 16)
        out.write("VALUE\n")
        _write_numbers(out, [float(v) for v in m.value])
        if m.is_integer:
            out.write("INTEGER\n")
            _write_numbers(out, [int(v) for v in m.is_integer], 32)
        if m.row_names:
            out.write("ROW_NAMES\n")
            for s in m.row_names:
                out.write(s + "\n")
        if m.col_names:
            out.write("COL_NAMES\n")
            for s in m.col_names:
                out.write(s + "\n")
        out.write("END\n")


_NUMERIC = {"OBJ", "COL_LOWER", "COL_UPPER", "ROW_LOWER", "ROW_UPPER", "COL_START", "ROW_INDEX", "VALUE", "INTEGER"}


def read_lpm(path: str) -> PyModel:
    m = PyModel()
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    if not lines or lines[0].split() != ["LPM", "1"]:
        raise ValueError("not an LPM 1 file")
    i = 1
    sizes = {}
    while i < len(lines):
        ln = lines[i]
        i += 1
        if not ln.strip() or ln.startswith("#"):
            continue
        tag, _, rest = ln.partition(" ")
        if tag == "END":
            break
        if tag == "NAME":
            m.name = rest
        elif tag == "SENSE":
            m.sense = -1 if rest.strip() == "MAX" else 1
        elif tag in ("ROWS", "COLS", "NNZ"):
            sizes[tag] = int(rest)
        elif tag == "OFFSET":
            m.obj_offset = float(rest)
        elif tag in _NUMERIC:
            count = {
                "OBJ": sizes["COLS"], "COL_LOWER": sizes["COLS"], "COL_UPPER": sizes["COLS"],
                "INTEGER": sizes["COLS"], "ROW_LOWER": sizes["ROWS"], "ROW_UPPER": sizes["ROWS"],
                "COL_START": sizes["COLS"] + 1, "ROW_INDEX": sizes["NNZ"], "VALUE": sizes["NNZ"],
            }[tag]
            vals = []
            while len(vals) < count:
                vals += lines[i].split()
                i += 1
            conv = int if tag in ("COL_START", "ROW_INDEX", "INTEGER") else float
            setattr(m, {"OBJ": "obj", "INTEGER": "is_integer"}.get(tag, tag.lower()), [conv(v) for v in vals])
        elif tag in ("ROW_NAMES", "COL_NAMES"):
            count = sizes["ROWS"] if tag == "ROW_NAMES" else sizes["COLS"]
            setattr(m, tag.lower(), lines[i : i + count])
            i += count
        else:
            raise ValueError(f"unknown section {tag!r}")
    m.num_rows, m.num_cols = sizes["ROWS"], sizes["COLS"]
    return m


# --------------------------------------------------------------------------- solution files
@dataclass
class PySolution:
    header: dict = field(default_factory=dict)
    x: list = field(default_factory=list)
    z: list = field(default_factory=list)
    row_activity: list = field(default_factory=list)
    y: list = field(default_factory=list)
    col_names: list = field(default_factory=list)
    row_names: list = field(default_factory=list)
    dual_ray: list = field(default_factory=list)    # Infeasible: Farkas row multipliers
    primal_ray: list = field(default_factory=list)  # Unbounded: recession direction


def read_solution(path: str) -> PySolution:
    """Parse the format written by src/io/solution_writer.cpp:

        PS26119-SOLUTION 1
        <key> <value>          (status, objective, engine, precision, model, ...)
        COLUMNS <n>
        <index> <x> <z> <name>
        ROWS <m>
        <index> <activity> <y> <name>
        DUAL_RAY <m> / PRIMAL_RAY <n>   (optional certificates: <index> <value> lines)
        END
    """
    s = PySolution()
    with open(path) as f:
        lines = f.read().splitlines()
    if not lines or not lines[0].startswith("PS26119-SOLUTION"):
        raise ValueError("not a ps26119 solution file")
    i = 1
    while i < len(lines):
        parts = lines[i].split(maxsplit=1)
        i += 1
        if not parts:
            continue
        key = parts[0]
        if key == "END":
            break
        if key in ("DUAL_RAY", "PRIMAL_RAY"):
            count = int(parts[1])
            vals = []
            for k in range(count):
                idx, v = lines[i].split()[:2]
                i += 1
                if int(idx) != k:
                    raise ValueError(f"malformed {key} block")
                vals.append(float(v))
            if key == "DUAL_RAY":
                s.dual_ray = vals
            else:
                s.primal_ray = vals
            continue
        if key in ("COLUMNS", "ROWS"):
            count = int(parts[1])
            for _ in range(count):
                idx, a, b, *name = lines[i].split(maxsplit=3)
                i += 1
                if key == "COLUMNS":
                    s.x.append(float(a)); s.z.append(float(b)); s.col_names.append(name[0] if name else "")
                else:
                    s.row_activity.append(float(a)); s.y.append(float(b)); s.row_names.append(name[0] if name else "")
        else:
            s.header[key] = parts[1] if len(parts) > 1 else ""
    return s
