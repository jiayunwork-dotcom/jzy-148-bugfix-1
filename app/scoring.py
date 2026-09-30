"""评分换算与在线打分引擎。

换算关系（odds = 好/坏 = (1-PD)/PD，分数越高风险越低）：
    score = offset + factor * ln(odds)
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

eta = intercept + Σ coef_i * WOE_i，PD = sigmoid(eta)，
故 ln(odds) = -eta，每箱分值 = -(intercept_share + coef_i * WOE_i) * factor + offset_i。

截距分摊：offset 与截距项按入模特征数平均分摊到每个特征
（intercept_share = intercept / k），全部箱分之和恰好等于总分。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .regression import sigmoid


@dataclass
class ScoredBin:
    bin_index: int
    label: str
    woe: float
    points: float
    missing: bool = False
    unseen: bool = False
    out_of_range: bool = False


def _edge_label(lower: float | None, upper: float | None) -> str:
    if lower is None and upper is None:
        return "all"
    if lower is None:
        return f"(-inf, {upper:g}]"
    if upper is None:
        return f"({lower:g}, +inf)"
    return f"({lower:g}, {upper:g}]"


def build_score_table(
    binning: dict[str, Any],
    coefficients: dict[str, float],
    intercept: float,
    base_score: float,
    base_odds: float,
    pdo: float,
) -> dict[str, Any]:
    """为一张卡生成每箱分值表（结果同时用于存储与在线打分）。"""
    factor = pdo / math.log(2.0)
    offset = base_score - factor * math.log(base_odds)
    selected = list(coefficients.keys())
    k = len(selected)
    per_offset = offset / k
    intercept_share = intercept / k

    table: dict[str, Any] = {
        "factor": factor,
        "offset": offset,
        "per_feature_offset": per_offset,
        "intercept_share": intercept_share,
        "features": {},
    }

    for name in selected:
        coef = coefficients[name]
        spec = binning[name]
        feat_out: list[dict[str, Any]] = []
        for b in spec["bins"]:
            points = per_offset - factor * (intercept_share + coef * b["woe"])
            feat_out.append({
                **b,
                "label": _edge_label(b.get("lower"), b.get("upper"))
                if spec["feature_type"] == "numeric"
                else ",".join(b["categories"]),
                "points": points,
            })
        mb = spec.get("missing_bin")
        if mb is not None:
            points = per_offset - factor * (intercept_share + coef * mb["woe"])
            feat_out.append({
                **mb,
                "label": "missing",
                "points": points,
            })
        table["features"][name] = {
            "feature_type": spec["feature_type"],
            "bins": feat_out,
        }
    return table


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("布尔值不是合法数值")
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        f = float(value.strip())
    else:
        raise ValueError(f"无法解析为数值: {value!r}")
    if not math.isfinite(f):
        raise ValueError(f"数值不是有限数: {value!r}")
    return f


def score_one(
    score_table: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """对单个申请人打分。

    返回总分、PD 及每个特征的落箱/得分。未见类别走专用“未见类别”箱
    （WOE 视为 0、分值为该特征的基准分摊），并显式标出 unseen，
    不静默归入任何训练箱。
    """
    factor = score_table["factor"]
    offset = score_table["offset"]
    per_offset = score_table["per_feature_offset"]
    intercept_share = score_table["intercept_share"]

    detail: list[dict[str, Any]] = []
    total_points = 0.0
    unseen: list[str] = []

    for name, spec in score_table["features"].items():
        value = raw.get(name)
        is_missing = value is None or (
            isinstance(value, str) and value.strip().lower() in
            ("", "na", "n/a", "nan", "none", "null"))
        chosen: dict[str, Any] | None = None
        out_of_range = False
        is_unseen = False

        if is_missing:
            chosen = next((b for b in spec["bins"] if b.get("missing")), None)
            if chosen is None:
                raise ValueError(
                    f"特征 {name} 缺失，但训练样本中该特征无缺失箱")
        elif spec["feature_type"] == "numeric":
            x = _to_float(value)
            chosen = None
            for b in spec["bins"]:
                if b.get("missing"):
                    continue
                lo, hi = b.get("lower"), b.get("upper")
                if (lo is None or x > lo) and (hi is None or x <= hi):
                    chosen = b
                    if (lo is None and x < hi) or (hi is None and x > lo):
                        out_of_range = True
                    break
            if chosen is None:
                raise ValueError(f"数值特征 {name}={value} 无法落箱")
        else:
            key = str(value).strip()
            chosen = next(
                (b for b in spec["bins"] if not b.get("missing")
                 and key in b.get("categories", [])),
                None)
            if chosen is None:
                is_unseen = True
                unseen.append(name)
                chosen = {
                    "bin_index": -1,
                    "label": "UNSEEN_CATEGORY",
                    "woe": 0.0,
                    "points": per_offset - factor * intercept_share,
                }

        total_points += chosen["points"]

        detail.append({
            "feature": name,
            "raw_value": None if is_missing else value,
            "bin_index": chosen["bin_index"],
            "bin_label": chosen["label"],
            "woe": chosen["woe"],
            "points": chosen["points"],
            "missing": bool(chosen.get("missing", False)),
            "unseen": is_unseen,
            "out_of_range": out_of_range,
        })

    # 由总分精确反推 eta：score = offset - factor*eta
    # odds 做下界保护，避免极端分值下 exp 溢出（PD 本身始终在 (0,1)）
    log_odds = min((total_points - offset) / factor, 700.0)
    pd = float(sigmoid(np.array([-log_odds]))[0])
    return {
        "total_score": total_points,
        "pd": pd,
        "odds_good_to_bad": math.exp(log_odds),
        "unseen_features": unseen,
        "has_unseen": bool(unseen),
        "features": detail,
    }
