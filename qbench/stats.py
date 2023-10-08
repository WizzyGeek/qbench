from __future__ import annotations

import typing as t
import gc as _gc
from itertools import repeat as _rp
from time import process_time as _pf

try:
    import numpy as np
    import numpy.typing as npt
    from scipy.stats import ttest_ind
except (ImportError, ModuleNotFoundError) as err:
    raise Exception("Need numpy and scipy installed for qbench.statistics") from err

def collect_stats(func: t.Callable[[], t.Any], n: int = 15, r: int = 1000):
    _gc.disable()
    a = np.zeros(n, dtype=np.float64)
    try:
        i = 0
        for _ in _rp(None, n):
            t = 0.0
            for _ in _rp(None, r):
                now = _pf()
                func()
                t += _pf() - now
            a[i] = t / r
            i += 1
    finally:
        _gc.enable()

    return a

if t.TYPE_CHECKING:
    Stats = t.Annotated[npt.NDArray, "One dimensional array of elapsed times"]
    class CallableWithAttr(t.Protocol):
        from_stats: t.Callable[[Stats, Stats], tuple[float, float, float]]

        def __call__(self, a: t.Callable[[], t.Any], b: t.Callable[[], t.Any], n: int = 15, r: int = 1000) -> tuple[float, float, float]:
            ...

def compare(a: t.Callable[[], t.Any], b: t.Callable[[], t.Any], n: int = 15, r: int = 1000) -> tuple[float, float, float]:
    """Returns the mean difference in seconds, probability that a is faster than b and
    the t-score

    The greater the value of n the more accurate the probability.

    Note
    ----
    You may observe inconsistent conclusions if the mean timings only differ by less than
    1 microsecond. Use a heavier computational load to reach more concrete conclusions.

    Also this will call the functions n * r times

    A result which looks like `(-0.34, 1.0, -100.322)`
    implies that a is faster than b by 340 milliseconds

    meanwhile a result like `(0.38, 0.0, 100.23)`
    implies a is slower than b by 380 milliseconds

    Warning
    -------
    Do not rely solely on this function to conclude anything statistically significant,
    the P value threshold should be taken to be very strict on the order of P<0.0001
    Also let n be small compared to r. The difference in mean should be setup to be
    theoritically be observed to be greater than 1 microsecond at the very least.
    """
    _gc.collect()
    astats = collect_stats(a, n // 2, r)
    _gc.collect()
    bstats = collect_stats(b, n // 2, r)

    ret = ttest_ind(astats, bstats, equal_var=False, alternative="less")
    # pvalue is for the Null hypothesis, 1 - pvalue is alternative, which is a less than b
    return (float(np.mean(astats) - np.mean(bstats)), 1 - ret.pvalue, ret.statistic)


def compare_from_stats(astats: Stats, bstats: Stats) -> tuple[float, float, float]:
    ret = ttest_ind(astats, bstats, equal_var=False, alternative="less")
    return (float(np.mean(astats) - np.mean(bstats)), 1 - ret.pvalue, ret.statistic)


compare.from_stats = compare_from_stats # type: ignore[attr-defined]

if t.TYPE_CHECKING:
    compare = t.cast(CallableWithAttr, compare)
