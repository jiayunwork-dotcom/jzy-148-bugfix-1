"""模型评估：KS 与 ROC-AUC，均自行实现。"""
from __future__ import annotations

import numpy as np


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    """ROC-AUC = 随机一坏一好，坏样本分数高于好样本的概率。

    用 Mann-Whitney U + 并列秩校正计算，完全分离时为 1。
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    # 并列值赋平均秩
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0
        i = j
    rank_pos = ranks[y[order] == 1].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def ks_statistic(y_true: list[int], scores: list[float]) -> float:
    """KS = 最大阈值处 |累计坏样本占比 - 累计好样本占比|。

    scores 为违约概率；并列分数不拆分（整组一起越过阈值）。
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    total_pos = y.sum()
    total_neg = len(y) - total_pos
    if total_pos == 0 or total_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    ks = 0.0
    cum_pos = cum_neg = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        block = y_sorted[i:j]
        cum_pos += block.sum()
        cum_neg += (1.0 - block).sum()
        diff = abs(cum_pos / total_pos - cum_neg / total_neg)
        ks = max(ks, float(diff))
        i = j
    return ks


def evaluate(y_true: list[int], pd_scores: list[float]) -> dict[str, float]:
    return {
        "ks": ks_statistic(y_true, pd_scores),
        "auc": roc_auc(y_true, pd_scores),
    }
