from __future__ import annotations

import typing as t
import gc as _gc
from itertools import repeat as _rp
from time import perf_counter as _pf

def _time(func: t.Callable[[], t.Any], n: int = 10000, pf: t.Callable[[], float] = _pf):
    now = pf()
    for _ in _rp(None, n):
        func()
    return pf() - now

def time(func: t.Callable[[], t.Any], n: int = 10000, timer: t.Callable[[], float] = _pf):
    _gc.disable()
    try:
        return _time(func, n, timer)
    finally:
        _gc.enable()

def time_from_factory(factory: t.Callable[[], t.Callable[[], t.Any]], n: int = 10000, timer: t.Callable[[], float] = _pf):
    return time(factory(), n, timer)

time.from_factory = time_from_factory # type: ignore[attr-defined]