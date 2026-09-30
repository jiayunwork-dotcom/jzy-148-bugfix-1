"""建卡管线：分箱 -> WOE -> 牛顿法逻辑回归 -> 评分换算 -> 评估。

各阶段的中间结果（初始箱、合并轨迹、各箱好坏计数/WOE/IV、回归迭代）
全部写进返回的 card 文档，供监管复现。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .binning import bin_feature
from .metrics import evaluate
from .parsing import Dataset
from .regression import fit_logistic_newton
from .scoring import build_score_table

DEFAULT_IV_THRESHOLD = 0.02


class ScorecardBuildError(RuntimeError):
    """建卡失败（分箱无可入模特征或回归不收敛等）。"""


def _woe_lookup(spec: dict[str, Any]) -> tuple[list[float | None], float | None]:
    """返回 (每个常规箱的 woe, 缺失箱 woe)。"""
    wb = [b["woe"] for b in spec["bins"]]
    mw = spec["missing_bin"]["woe"] if spec.get("missing_bin") else None
    return wb, mw


def _assign_woe(value: Any, spec: dict[str, Any]) -> float:
    if value is None:
        mb = spec.get("missing_bin")
        if mb is None:
            raise ValueError(f"打分时特征 {spec} 出现训练中未有的缺失")
        return float(mb["woe"])
    if spec["feature_type"] == "numeric":
        for b in spec["bins"]:
            lo, hi = b.get("lower"), b.get("upper")
            if (lo is None or value > lo) and (hi is None or value <= hi):
                return float(b["woe"])
        raise ValueError("数值未落入任何箱")
    key = str(value).strip()
    for b in spec["bins"]:
        if key in b.get("categories", []):
            return float(b["woe"])
    raise KeyError(key)


def build_scorecard(
    dataset: Dataset,
    *,
    name: str,
    features: list[str] | None = None,
    iv_threshold: float = DEFAULT_IV_THRESHOLD,
    min_bin_pct: float = 0.05,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 50.0,
) -> dict[str, Any]:
    started = time.time()
    y = np.array(dataset.labels, dtype=int)
    n = len(y)

    # ---- 阶段 1：逐特征分箱 ------------------------------------------------
    binning: dict[str, dict[str, Any]] = {}
    bin_traces: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for fname in dataset.feature_names:
        res = bin_feature(
            fname, dataset.columns[fname], dataset.labels,
            dataset.feature_types[fname], min_bin_pct=min_bin_pct)
        bin_traces[fname] = {
            "usable": res.usable,
            "reason": res.reason,
            "feature_type": res.feature_type,
            "direction": res.direction,
            "initial_bins": res.initial_bins,
            "merge_trace": res.merge_trace,
            "final_bins": res.bins,
            "missing_bin": res.missing_bin,
            "iv": res.iv if res.usable else None,
            "total_good": res.total_good,
            "total_bad": res.total_bad,
        }
        if not res.usable:
            skipped[fname] = res.reason
            continue
        def _bin_dict(b: dict[str, Any]) -> dict[str, Any]:
            common = ("bin_index", "total", "good", "bad", "bad_rate",
                      "woe", "iv_contrib")
            out = {k: b[k] for k in common}
            if res.feature_type == "numeric":
                out["lower"] = b["lower"]
                out["upper"] = b["upper"]
            else:
                out["categories"] = b["categories"]
            return out

        spec = {
            "feature_type": res.feature_type,
            "direction": res.direction,
            "bins": [_bin_dict(b) for b in res.bins],
            "missing_bin": res.missing_bin,
            "iv": res.iv,
        }
        binning[fname] = spec

    # ---- 阶段 2：特征筛选 --------------------------------------------------
    if features is not None:
        missing_usable = [f for f in features if f not in binning]
        if missing_usable:
            reasons = {f: skipped.get(f, "未知原因") for f in missing_usable}
            raise ScorecardBuildError(
                f"指定入模特征未能完成合格分箱: {reasons}")
        selected = list(features)
        selection_rule = "user_specified"
    else:
        selected = sorted(
            (f for f in binning if binning[f]["iv"] >= iv_threshold),
            key=lambda f: binning[f]["iv"], reverse=True)
        selection_rule = f"iv_threshold={iv_threshold}"
        if not selected:
            raise ScorecardBuildError(
                f"IV 不低于 {iv_threshold} 的特征为 0，无法自动筛选入模特征")

    # 去掉 WOE 方差为 0 的特征（对回归零贡献且会导致奇异）
    informative: list[str] = []
    for f in selected:
        ws = [b["woe"] for b in binning[f]["bins"]]
        if binning[f]["missing_bin"] is not None:
            ws.append(binning[f]["missing_bin"]["woe"])
        if max(ws) - min(ws) > 1e-12:
            informative.append(f)
        else:
            skipped[f] = "各箱 WOE 完全相同，无区分度"
    dropped_const = [f for f in selected if f not in informative]
    if not informative:
        raise ScorecardBuildError("入模特征的 WOE 均为常数，无法拟合回归")

    # ---- 阶段 3：WOE 矩阵 --------------------------------------------------
    X = np.zeros((n, len(informative)))
    for j, f in enumerate(informative):
        spec = binning[f]
        for i, v in enumerate(dataset.columns[f]):
            X[i, j] = _assign_woe(v, spec)

    # 去掉线性重复列（重复 WOE 编码会使海森奇异）
    kept_idx: list[int] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if any(np.allclose(col, X[:, k]) for k in kept_idx):
            skipped[informative[j]] = "WOE 编码与另一入模特征完全重复"
            continue
        kept_idx.append(j)
    X = X[:, kept_idx]
    final_features = [informative[j] for j in kept_idx]

    # ---- 阶段 4：牛顿法逻辑回归 -------------------------------------------
    fit = fit_logistic_newton(X, np.asarray(dataset.labels))
    intercept = fit.intercept
    coefficients = {f: float(fit.coef[j + 1])
                    for j, f in enumerate(final_features)}

    eta = fit.coef[0] + X @ fit.coef[1:]
    from .regression import sigmoid
    pd_scores = sigmoid(eta)
    mean_pd = float(pd_scores.mean())
    actual_bad_rate = float(y.mean())
    mean_pd_error = abs(mean_pd - actual_bad_rate)
    if mean_pd_error > 1e-6:
        raise ScorecardBuildError(
            f"全样本平均预测 PD {mean_pd:.10f} 与实际违约率 "
            f"{actual_bad_rate:.10f} 偏差 {mean_pd_error:.2e} 超过 1e-6")

    # ---- 阶段 5：评分换算 --------------------------------------------------
    final_binning = {f: binning[f] for f in final_features}
    score_table = build_score_table(
        final_binning, coefficients, intercept,
        base_score, base_odds, pdo)

    # ---- 阶段 6：评估 ------------------------------------------------------
    metrics = evaluate(dataset.labels, list(map(float, pd_scores)))
    feature_iv = {f: float(binning[f]["iv"]) for f in final_features}

    card: dict[str, Any] = {
        "name": name,
        "config": {
            "features_requested": features,
            "selection_rule": selection_rule,
            "iv_threshold": iv_threshold if features is None else None,
            "min_bin_pct": min_bin_pct,
            "base_score": base_score,
            "base_odds": base_odds,
            "pdo": pdo,
            "target_column": dataset.target,
        },
        "training": {
            "n_samples": n,
            "n_bad": int(y.sum()),
            "n_good": n - int(y.sum()),
            "bad_rate": actual_bad_rate,
            "newton_iterations": fit.iterations,
            "converged": True,
            "log_loss": fit.log_loss,
            "mean_predicted_pd": mean_pd,
            "mean_pd_error": mean_pd_error,
            "elapsed_seconds": time.time() - started,
        },
        "selected_features": final_features,
        "dropped_features": [
            {"feature": f, "reason": r}
            for f, r in skipped.items() if f not in final_features
        ],
        "constants_dropped": dropped_const,
        "binning": final_binning,
        "coefficients": {"intercept": intercept, **coefficients},
        "score_table": score_table,
        "metrics": {**metrics, "feature_iv": feature_iv},
        "notes": {
            "intercept_allocation": (
                f"offset={score_table['offset']:.10f} 与截距 {intercept:.10f} "
                f"按 {len(final_features)} 个入模特征平均分摊，"
                f"每特征基准分 {score_table['per_feature_offset']:.10f}、"
                f"截距份额 {score_table['intercept_share']:.10f}；"
                f"factor=PDO/ln2={score_table['factor']:.10f}；"
                "各箱分值之和即为总分，score=offset-factor*logit(PD)"),
            "unseen_category_policy": (
                "打分时训练未见类别使用专用箱 UNSEEN_CATEGORY："
                "WOE 视为 0（无证据），得该特征基准分，并在返回中标注；"
                "不静默并入任何训练箱"),
            "missing_policy": "缺失值始终单独成一箱，不参与相邻合并",
            "woe_definition": "WOE=ln((箱内好/总好)/(箱内坏/总坏))，"
                              "IV=Σ(p_good-p_bad)ln(p_good/p_bad)（含缺失箱）",
        },
        "artifacts": {
            "all_feature_binning": bin_traces,
        },
    }
    return card


def verify_monotonic(card: dict[str, Any]) -> dict[str, bool]:
    """校验数值特征最终 WOE 序列单调（与记录方向一致）。"""
    out: dict[str, bool] = {}
    for f, spec in card["binning"].items():
        if spec["feature_type"] != "numeric":
            continue
        ws = [b["woe"] for b in spec["bins"]]
        direction = spec["direction"]
        out[f] = all(ws[i + 1] >= ws[i] - 1e-12 for i in range(len(ws) - 1)) \
            if direction == "increasing" else \
            all(ws[i + 1] <= ws[i] + 1e-12 for i in range(len(ws) - 1))
    return out
