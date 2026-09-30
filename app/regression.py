"""逻辑回归（自己实现的牛顿迭代，不用 sklearn / statsmodels）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RegressionError(RuntimeError):
    """牛顿迭代无法收敛（含完全分离）。"""


@dataclass
class FitResult:
    coef: np.ndarray                 # 含截距，coef[0] 为截距
    iterations: int
    converged: bool
    log_loss: float
    intercept: float
    feature_coef: dict[str, float]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
    ridge: float = 1e-8,
) -> FitResult:
    """带截距的逻辑回归，牛顿-拉夫逊法求解。

    海森矩阵 H = X' W X，迭代 beta -= H^{-1} g。
    奇异时加入轻微 L2 脊（1e-8，仅作用于非截距项）以保证数值可逆；
    系数发散（|beta| > 50）视为完全/准完全分离，作业失败。
    """
    y = y.astype(float)
    n, d = X.shape
    A = np.column_stack([np.ones(n), X])
    p = d + 1
    beta = np.zeros(p)

    ridge_diag = np.zeros(p)
    ridge_diag[1:] = ridge

    for it in range(1, max_iter + 1):
        eta = A @ beta
        mu = sigmoid(eta)
        grad = A.T @ (mu - y)
        w = np.clip(mu * (1.0 - mu), 1e-15, None)
        H = (A.T * w) @ A + np.diag(ridge_diag)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            raise RegressionError("海森矩阵奇异，牛顿迭代无法求解")
        beta_new = beta - step

        if not np.all(np.isfinite(beta_new)) or np.max(np.abs(beta_new)) > 50.0:
            raise RegressionError(
                f"第 {it} 次迭代系数发散（|beta|>50），"
                "数据可能存在完全/准完全分离，模型不可用")

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        raise RegressionError(f"牛顿迭代 {max_iter} 次仍未收敛")

    eta = A @ beta
    mu = sigmoid(eta)
    eps = 1e-15
    log_loss = float(-np.mean(
        y * np.log(mu + eps) + (1.0 - y) * np.log(1.0 - mu + eps)))
    return FitResult(
        coef=beta,
        iterations=it,
        converged=True,
        log_loss=log_loss,
        intercept=float(beta[0]),
        feature_coef={},  # 由调用方按特征名填充
    )
