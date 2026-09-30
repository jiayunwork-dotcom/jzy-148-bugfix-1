"""分箱引擎（自行实现，不依赖 scikit-learn）。

流程：
1. 数值特征：等频（分位数）切成不超过 20 个初始箱，区间为 (lower, upper]；
   类别特征：按坏样本率升序排列每个类别。
2. 相邻合并：先消除好/坏计数为 0 的箱，再合并样本占比低于下限的箱，
   最后反复合并最严重的 WOE 单调性违例箱对，直到 WOE 严格非降/非升
   （数值特征方向由代表值与 WOE 的相关系数决定；类别特征坏率升序 => WOE 非升）。
3. 缺失值始终单独成一箱，不参与相邻合并，也不参与数值 WOE 的单调性判断。

每一步合并都记录到 merge_trace，初始箱也保留，便于监管复现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_INITIAL_BINS = 20
EPS = 1e-12


class BinningError(ValueError):
    """特征无法完成满足约束的分箱。"""


def woe(good: float, bad: float, total_good: float, total_bad: float) -> float:
    if good <= 0 or bad <= 0:
        return math.nan
    return math.log((good / total_good) / (bad / total_bad))


def bin_iv(good: int, bad: int, total_good: int, total_bad: int) -> float:
    pg = good / total_good
    pb = bad / total_bad
    if pg <= 0 or pb <= 0:
        return 0.0
    return (pg - pb) * math.log(pg / pb)


def corrected_woe_iv(
    good: int, bad: int, total_good: int, total_bad: int
) -> tuple[float, float, bool]:
    """缺失箱专用：好坏有一侧为 0 时按 +0.5 修正计数后算 WOE/IV。

    常规箱通过合并保证好坏均非零；缺失箱按业务要求始终单独保留，
    不能通过合并消除 0，因此采用标准的 0.5 校正避免无穷分值。
    """
    corrected = good == 0 or bad == 0
    g, b = (good + 0.5, bad + 0.5) if corrected else (float(good), float(bad))
    pg, pb = g / total_good, b / total_bad
    w = math.log(pg / pb)
    iv = (pg - pb) * math.log(pg / pb)
    return w, iv, corrected


@dataclass
class Group:
    good: int = 0
    bad: int = 0
    lower: float | None = None    # 数值箱 (lower, upper]
    upper: float | None = None
    rep: float = 0.0              # 箱代表值（成员均值）
    categories: list[str] = field(default_factory=list)


@dataclass
class FeatureBinResult:
    feature: str
    feature_type: str
    usable: bool
    reason: str = ""
    direction: str = ""           # increasing / decreasing（WOE 随特征值方向）
    bins: list[dict[str, Any]] = field(default_factory=list)
    missing_bin: dict[str, Any] | None = None
    initial_bins: list[dict[str, Any]] = field(default_factory=list)
    iv: float = 0.0
    merge_trace: list[dict[str, Any]] = field(default_factory=list)
    total_good: int = 0
    total_bad: int = 0


def _initial_numeric_groups(values: list[float], labels: list[int]) -> list[Group]:
    arr = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = len(arr)
    n_bins = min(MAX_INITIAL_BINS, n)

    # 先按唯一值聚合（同值不可拆箱），再贪心等频：从最小值开始累加，
    # 当前箱累计样本数达到目标 n/n_bins，且至少还剩 (剩余箱数-1) 个
    # 唯一值时封箱。决策只用到唯一值序列和整数计数，样本复制一份后
    # 计数与阈值同时翻倍，初始箱边界完全一致、好/坏计数恰好翻倍。
    uniq, inv = np.unique(arr, return_inverse=True)
    good_by_val = np.zeros(len(uniq), dtype=int)
    bad_by_val = np.zeros(len(uniq), dtype=int)
    np.add.at(good_by_val, inv[(y == 0)], 1)
    np.add.at(bad_by_val, inv[(y == 1)], 1)

    target = n / n_bins
    groups: list[Group] = []
    cur_good = cur_bad = 0
    start = 0
    for i in range(len(uniq)):
        cur_good += int(good_by_val[i])
        cur_bad += int(bad_by_val[i])
        cur_size = cur_good + cur_bad
        bins_open = len(groups)
        vals_left = len(uniq) - (i + 1)
        bins_left = n_bins - bins_open
        reached = cur_size >= target
        enough_values_left = vals_left >= bins_left - 1
        not_last_bin = bins_open < n_bins - 1
        if reached and enough_values_left and not_last_bin:
            vals = uniq[start:i + 1]
            groups.append(Group(
                good=cur_good, bad=cur_bad,
                upper=float(uniq[i]),
                rep=float(np.average(
                    vals, weights=good_by_val[start:i + 1]
                    + bad_by_val[start:i + 1]))))
            cur_good = cur_bad = 0
            start = i + 1

    vals = uniq[start:]
    groups.append(Group(
        good=cur_good, bad=cur_bad, upper=None,
        rep=float(np.average(
            vals, weights=good_by_val[start:] + bad_by_val[start:]))))

    uppers = [g.upper for g in groups[:-1]]
    for i, g in enumerate(groups):
        g.lower = None if i == 0 else uppers[i - 1]
    groups[-1].upper = None
    return groups


def _initial_category_groups(values: list[str], labels: list[int]) -> list[Group]:
    agg: dict[str, Group] = {}
    for value, label in zip(values, labels):
        grp = agg.setdefault(value, Group(categories=[value]))
        if label == 1:
            grp.bad += 1
        else:
            grp.good += 1
    # 坏样本率升序；平局按类别名保证确定性
    groups = list(agg.values())
    groups.sort(key=lambda g: (g.bad / (g.good + g.bad), g.categories[0]))
    return groups


def _merge_groups(groups: list[Group], i: int, j: int) -> Group:
    a, b = groups[i], groups[j]
    return Group(
        good=a.good + b.good,
        bad=a.bad + b.bad,
        lower=a.lower,
        upper=b.upper,
        rep=(a.rep * (a.good + a.bad) + b.rep * (b.good + b.bad))
        / (a.good + a.bad + b.good + b.bad),
        categories=a.categories + b.categories,
    )


def _woe_series(groups: list[Group], tg: int, tb: int) -> list[float]:
    return [woe(g.good, g.bad, tg, tb) for g in groups]


def _merge_zero_groups(
    groups: list[Group], trace: list[dict[str, Any]],
    tg: int, tb: int, total: int,
) -> list[Group]:
    """合并好/坏计数为 0 的箱。"""
    while True:
        zero = next((i for i, g in enumerate(groups)
                     if g.good == 0 or g.bad == 0), None)
        if zero is None:
            return groups
        if len(groups) <= 2:
            # 再合并就只剩一个箱，无法满足“每箱好坏均非零”
            raise BinningError("好坏样本完全分离，合并后只剩单一箱")
        if zero == 0:
            target = 1
        elif zero == len(groups) - 1:
            target = zero - 1
        else:
            # 确定性规则：按同方向计数最小的邻居合并，并列取左邻
            left, right = groups[zero - 1], groups[zero + 1]
            cur = groups[zero]
            if cur.bad == 0 and cur.good > 0:
                target = zero - 1 if left.bad <= right.bad else zero + 1
            elif cur.good == 0 and cur.bad > 0:
                target = zero - 1 if left.good <= right.good else zero + 1
            else:  # 空箱（不应出现，稳妥处理）
                target = zero - 1
        lo = min(zero, target)
        merged = _merge_groups(groups, lo, lo + 1)
        before = _woe_series(groups, tg, tb)
        groups = [*groups[:lo], merged, *groups[lo + 2:]]
        trace.append({
            "step": len(trace) + 1,
            "reason": "zero_good_or_bad",
            "merged_bin_indices": [lo, lo + 1],
            "woe_before": [None if math.isnan(w) else w for w in before],
            "bins_after": len(groups),
        })


def _merge_small_groups(
    groups: list[Group], min_bin_pct: float, trace: list[dict[str, Any]],
    tg: int, tb: int, total: int,
) -> list[Group]:
    """合并样本占比低于下限的箱：与有限 WOE 最接近的相邻箱合并。"""
    while len(groups) > 2:
        counts = [g.good + g.bad for g in groups]
        # 最小箱取计数最小且索引最小者（纯整数决策，复制样本后一致）
        small = min(range(len(groups)),
                    key=lambda i: (counts[i], i))
        if counts[small] / total >= min_bin_pct:
            return groups
        woe_before = _woe_series(groups, tg, tb)
        w_small = woe_before[small]

        def neighbor_key(j: int) -> tuple[float, float, int]:
            """确定性邻居键：相对 WOE 距离 -> 坏率距离 -> 索引。"""
            wj = woe_before[j]
            gj = groups[j]
            gs = groups[small]
            br_small = gs.bad / (gs.good + gs.bad)
            br_j = gj.bad / (gj.good + gj.bad)
            if math.isfinite(w_small) and math.isfinite(wj):
                scale = max(abs(w_small), abs(wj), 1.0)
                return (abs(w_small - wj) / scale,
                        abs(br_small - br_j), j)
            # 含 0 计数的箱 WOE 为 inf：按坏率距离
            return (math.inf, abs(br_small - br_j), j)

        if small == 0:
            target = 1
        elif small == len(groups) - 1:
            target = small - 1
        else:
            kl, kr = (neighbor_key(small - 1),
                      neighbor_key(small + 1))
            target = small - 1 if kl <= kr else small + 1
        lo = min(small, target)
        merged = _merge_groups(groups, lo, lo + 1)
        groups = [*groups[:lo], merged, *groups[lo + 2:]]
        trace.append({
            "step": len(trace) + 1,
            "reason": "min_bin_pct",
            "merged_bin_indices": [lo, lo + 1],
            "woe_before": [None if math.isnan(w) else w for w in woe_before],
            "min_bin_share": counts[small] / total,
            "bins_after": len(groups),
        })
    # 末轮再兜底一次零计数（理论上不会触发）
    return _merge_zero_groups(groups, trace, tg, tb, total)


def _numeric_direction(groups: list[Group], tg: int, tb: int) -> int:
    reps = np.array([g.rep for g in groups], dtype=float)
    ws = np.array([woe(g.good, g.bad, tg, tb) for g in groups], dtype=float)
    if np.std(ws) < EPS:
        return 1
    corr = float(np.corrcoef(reps, ws)[0, 1])
    if corr >= 0:
        return 1
    return -1


def _enforce_monotonic(
    groups: list[Group], direction: int, trace: list[dict[str, Any]],
    tg: int, tb: int,
) -> list[Group]:
    """反复合并“WOE 差异绝对值最小”的违例相邻箱对，直到单调。"""
    while True:
        ws = _woe_series(groups, tg, tb)
        violation: int | None = None
        best_diff = math.inf
        for i in range(len(groups) - 1):
            w1, w2 = ws[i], ws[i + 1]
            ok = (w2 >= w1 - EPS) if direction > 0 else (w2 <= w1 + EPS)
            if not ok:
                diff = abs(w2 - w1)
                if diff < best_diff:
                    best_diff, violation = diff, i
        if violation is None:
            return groups
        if len(groups) <= 2:
            raise BinningError("单调性约束无法在两个以上箱中满足")
        i = violation
        merged = _merge_groups(groups, i, i + 1)
        groups = [*groups[:i], merged, *groups[i + 2:]]
        trace.append({
            "step": len(trace) + 1,
            "reason": "monotonic_violation",
            "merged_bin_indices": [i, i + 1],
            "woe_before": [None if math.isnan(w) else w for w in ws],
            "direction": "increasing" if direction > 0 else "decreasing",
            "bins_after": len(groups),
        })


def _snapshot(groups: list[Group], feature_type: str,
              tg: int, tb: int) -> list[dict[str, Any]]:
    out = []
    for i, g in enumerate(groups):
        item: dict[str, Any] = {
            "bin_index": i,
            "total": g.good + g.bad,
            "good": g.good,
            "bad": g.bad,
            "bad_rate": g.bad / (g.good + g.bad) if g.good + g.bad else None,
            "woe": woe(g.good, g.bad, tg, tb),
            "iv_contrib": bin_iv(g.good, g.bad, tg, tb),
        }
        if feature_type == "numeric":
            item.update({"lower": g.lower, "upper": g.upper})
        else:
            item["categories"] = list(g.categories)
        out.append(item)
    return out


def bin_feature(
    feature: str,
    values: list[Any],
    labels: list[int],
    feature_type: str,
    min_bin_pct: float = 0.05,
) -> FeatureBinResult:
    """对单个特征执行完整分箱流程。"""
    total_bad = int(sum(1 for y in labels if y == 1))
    total_good = len(labels) - total_bad

    present = [(v, y) for v, y in zip(values, labels) if v is not None]
    miss_good = sum(1 for v, y in zip(values, labels)
                    if v is None and y == 0)
    miss_bad = sum(1 for v, y in zip(values, labels)
                   if v is None and y == 1)

    result = FeatureBinResult(
        feature=feature, feature_type=feature_type, usable=False,
        total_good=total_good, total_bad=total_bad,
    )

    if not present:
        result.reason = "整列缺失"
        result.missing_bin = None
        return result

    pv = [v for v, _ in present]
    py = [y for _, y in present]
    if feature_type == "numeric":
        groups = _initial_numeric_groups(pv, py)
    else:
        groups = _initial_category_groups(pv, py)

    result.initial_bins = _snapshot(groups, feature_type, total_good, total_bad)

    try:
        groups = _merge_zero_groups(
            groups, result.merge_trace, total_good, total_bad, len(labels))
        groups = _merge_small_groups(
            groups, min_bin_pct, result.merge_trace,
            total_good, total_bad, len(labels))
        # 只剩两个箱时小箱无法再合并；约束确实不可满足
        if any((g.good + g.bad) / len(labels) < min_bin_pct - EPS
               for g in groups):
            raise BinningError(
                f"无法在至少两个箱的同时满足最小箱占比 {min_bin_pct:.2%}")
        if feature_type == "numeric":
            direction = _numeric_direction(groups, total_good, total_bad)
        else:
            direction = -1  # 按坏率升序 => WOE 非升
        groups = _enforce_monotonic(
            groups, direction, result.merge_trace, total_good, total_bad)
    except BinningError as exc:
        result.reason = str(exc)
        return result

    if len(groups) < 2:
        result.reason = "合并后只剩一个箱"
        return result

    result.bins = _snapshot(groups, feature_type, total_good, total_bad)
    result.direction = "increasing" if direction > 0 else "decreasing"

    # 缺失箱（WOE/IV 以全样本好坏总数为分母，常规箱同样如此）
    if miss_good + miss_bad > 0:
        m_woe, m_iv, m_corr = corrected_woe_iv(
            miss_good, miss_bad, total_good, total_bad)
        result.missing_bin = {
            "bin_index": len(result.bins),
            "missing": True,
            "total": miss_good + miss_bad,
            "good": miss_good,
            "bad": miss_bad,
            "bad_rate": miss_bad / (miss_good + miss_bad),
            "woe": m_woe,
            "iv_contrib": m_iv,
            "counts_corrected_0p5": m_corr,
        }

    result.iv = sum(b["iv_contrib"] for b in result.bins)
    if result.missing_bin is not None:
        result.iv += result.missing_bin["iv_contrib"]
    result.usable = True
    return result
