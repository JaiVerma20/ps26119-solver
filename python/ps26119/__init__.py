"""ps26119 — Python interface (thin ctypes layer over the C API in include/ps26119/ps26119.h).

    import numpy as np, ps26119
    res = ps26119.solve_lp(c, A, row_lower, row_upper, col_lower, col_upper, sense=-1)
    res.status, res.objective, res.x, res.y, res.z

A may be a dense 2-D array or a (col_start, row_index, value) CSC triple (scipy.sparse
csc_matrix also works if SciPy is installed). Infinity = numpy.inf. The shared library is
looked up in $PS26119_LIB, then build/ next to this package's repo, then the system path.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["solve_lp", "Result", "version", "OPTIMAL", "INFEASIBLE", "UNBOUNDED", "ITERATION_LIMIT", "TIME_LIMIT",
           "NUMERICAL_ERROR", "NOT_SOLVED", "INVALID_ARGUMENT"]

OPTIMAL, INFEASIBLE, UNBOUNDED, ITERATION_LIMIT, TIME_LIMIT, NUMERICAL_ERROR, NOT_SOLVED, INVALID_ARGUMENT = range(8)
_STATUS = ["Optimal", "Infeasible", "Unbounded", "IterationLimit", "TimeLimit", "NumericalError", "NotSolved",
           "InvalidArgument"]
_ALG = {"auto": 0, "oracle": 1, "pdlp": 2, "r2hpdhg": 3}
_PREC = {"fp64": 0, "mixed": 1}


class _Options(ctypes.Structure):
    _fields_ = [("algorithm", ctypes.c_int), ("precision", ctypes.c_int), ("use_gpu", ctypes.c_int),
                ("tolerance", ctypes.c_double), ("time_limit", ctypes.c_double),
                ("iteration_limit", ctypes.c_longlong), ("verbosity", ctypes.c_int),
                ("warm_x", ctypes.POINTER(ctypes.c_double)), ("warm_y", ctypes.POINTER(ctypes.c_double))]


class _Result(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("objective", ctypes.c_double), ("dual_objective", ctypes.c_double),
                ("primal_residual", ctypes.c_double), ("dual_residual", ctypes.c_double), ("gap", ctypes.c_double),
                ("iterations", ctypes.c_longlong), ("seconds", ctypes.c_double), ("engine", ctypes.c_char * 16),
                ("message", ctypes.c_char * 160)]


def _load():
    names = ["libps26119.dylib", "libps26119.so", "ps26119.dll"]
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if os.environ.get("PS26119_LIB"):
        cands.append(os.environ["PS26119_LIB"])
    for d in (os.path.join(here, "..", "..", "build"), os.path.join(here, "..", "..", "build-gpu"), here):
        cands += [os.path.join(d, n) for n in names]
    cands += names
    for p in cands:
        try:
            lib = ctypes.CDLL(p)
            break
        except OSError:
            continue
    else:
        raise OSError("libps26119 not found: build with CMake (target ps26119_shared) or set PS26119_LIB")
    D = ctypes.POINTER(ctypes.c_double)
    I = ctypes.POINTER(ctypes.c_int)
    lib.ps26119_solve_lp.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, D, D, D, D, D, I, I, D,
                                     ctypes.POINTER(_Options), ctypes.POINTER(_Result), D, D, D]
    lib.ps26119_solve_lp.restype = ctypes.c_int
    lib.ps26119_default_options.argtypes = [ctypes.POINTER(_Options)]
    lib.ps26119_version.restype = ctypes.c_char_p
    return lib


_lib = None


def _get():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


def version() -> str:
    return _get().ps26119_version().decode()


@dataclass
class Result:
    status: int
    status_name: str
    objective: float
    dual_objective: float
    primal_residual: float
    dual_residual: float
    gap: float
    iterations: int
    seconds: float
    engine: str
    message: str
    x: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)
    z: np.ndarray = field(repr=False, default=None)


def _csc(A, m, n):
    if isinstance(A, tuple):
        cs, ri, v = A
        return (np.ascontiguousarray(cs, dtype=np.int32), np.ascontiguousarray(ri, dtype=np.int32),
                np.ascontiguousarray(v, dtype=np.float64))
    if hasattr(A, "tocsc"):  # scipy.sparse
        A = A.tocsc()
        A.sort_indices()
        return (A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.astype(np.float64))
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (m, n):
        raise ValueError(f"A has shape {A.shape}, expected {(m, n)}")
    cs, ri, v = [0], [], []
    for j in range(n):
        nz = np.nonzero(A[:, j])[0]
        ri += nz.tolist()
        v += A[nz, j].tolist()
        cs.append(len(ri))
    return np.asarray(cs, np.int32), np.asarray(ri, np.int32), np.asarray(v, np.float64)


def _ptr(a, ctype=ctypes.c_double):
    return a.ctypes.data_as(ctypes.POINTER(ctype)) if a is not None else None


def solve_lp(c, A, row_lower, row_upper, col_lower=None, col_upper=None, sense=1, offset=0.0, algorithm="auto",
             precision="fp64", gpu=False, tolerance=1e-8, time_limit=3600.0, iteration_limit=0, warm_x=None,
             warm_y=None, verbosity=0) -> Result:
    """min (sense=+1) or max (sense=−1) cᵀx + offset  s.t. row_lower ≤ A x ≤ row_upper, bounds."""
    c = np.ascontiguousarray(c, dtype=np.float64)
    n = c.size
    rl = np.ascontiguousarray(row_lower, dtype=np.float64)
    ru = np.ascontiguousarray(row_upper, dtype=np.float64)
    m = rl.size
    cl = np.zeros(n) if col_lower is None else np.ascontiguousarray(col_lower, dtype=np.float64)
    cu = np.full(n, np.inf) if col_upper is None else np.ascontiguousarray(col_upper, dtype=np.float64)
    cs, ri, v = _csc(A, m, n)
    lib = _get()
    opt = _Options()
    lib.ps26119_default_options(ctypes.byref(opt))
    opt.algorithm = _ALG[algorithm]
    opt.precision = _PREC[precision]
    opt.use_gpu = int(bool(gpu))
    opt.tolerance = tolerance
    opt.time_limit = time_limit
    opt.iteration_limit = iteration_limit
    opt.verbosity = verbosity
    wx = None if warm_x is None else np.ascontiguousarray(warm_x, dtype=np.float64)
    wy = None if warm_y is None else np.ascontiguousarray(warm_y, dtype=np.float64)
    opt.warm_x = _ptr(wx)
    opt.warm_y = _ptr(wy)
    res = _Result()
    x, y, z = np.empty(n), np.empty(m), np.empty(n)
    st = lib.ps26119_solve_lp(m, n, int(sense), float(offset), _ptr(c), _ptr(cl), _ptr(cu), _ptr(rl), _ptr(ru),
                              _ptr(cs, ctypes.c_int), _ptr(ri, ctypes.c_int), _ptr(v), ctypes.byref(opt),
                              ctypes.byref(res), _ptr(x), _ptr(y), _ptr(z))
    have = st in (OPTIMAL, ITERATION_LIMIT, TIME_LIMIT)
    return Result(st, _STATUS[st] if 0 <= st < len(_STATUS) else str(st), res.objective, res.dual_objective,
                  res.primal_residual, res.dual_residual, res.gap, res.iterations, res.seconds,
                  res.engine.decode(), res.message.decode(), x if have else None, y if have else None,
                  z if have else None)
