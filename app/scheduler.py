"""建卡作业调度。

建卡作为后台作业在线程池中执行。每个作业把 CSV 字节、参数、Dataset、
分箱中间结果、回归结果都保存在自己的工作线程局部变量中，
作业之间不共享任何可变数据，只有最终/失败状态写入 Repository。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .binning import BinningError
from .card import ScorecardBuildError, build_scorecard
from .config import settings
from .parsing import (
    SampleValidationError,
    build_dataset,
    validate_job_request,
)
from .regression import RegressionError
from .storage import Repository


@dataclass
class BuildParams:
    name: str
    csv_data: bytes
    target_col: str | None = None
    features: list[str] | None = None
    iv_threshold: float = 0.02
    min_bin_pct: float = 0.05
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 50.0


class JobScheduler:
    def __init__(self, repo: Repository, max_workers: int | None = None) -> None:
        self.repo = repo
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or settings.max_workers,
            thread_name_prefix="scorecard-build")
        self._lock = threading.Lock()

    @staticmethod
    def parse_and_validate(params: BuildParams) -> Any:
        """作业开始前的同步校验：样本/参数不合格直接拒绝（HTTP 400）。"""
        dataset = build_dataset(params.csv_data, target_col=params.target_col)
        validate_job_request(
            dataset,
            pdh=params.pdo,
            features=params.features,
            min_bin_pct=params.min_bin_pct,
            iv_threshold=params.iv_threshold)
        return dataset

    def submit(self, params: BuildParams) -> str:
        # 入队前先做完所有“硬性拒绝”校验
        dataset = self.parse_and_validate(params)
        job_id = self.repo.create_job(params.name, {
            "features": params.features,
            "iv_threshold": params.iv_threshold,
            "min_bin_pct": params.min_bin_pct,
            "base_score": params.base_score,
            "base_odds": params.base_odds,
            "pdo": params.pdo,
            "target_col": params.target_col,
            "n_samples": dataset.total,
        })
        # dataset 及 csv_data 仅作为本作业闭包局部变量传入
        self._pool.submit(self._run, job_id, params, dataset)
        return job_id

    def _run(self, job_id: str, params: BuildParams, dataset: Any) -> None:
        self.repo.mark_job_running(job_id)
        try:
            card = build_scorecard(
                dataset,
                name=params.name,
                features=params.features,
                iv_threshold=params.iv_threshold,
                min_bin_pct=params.min_bin_pct,
                base_score=params.base_score,
                base_odds=params.base_odds,
                pdo=params.pdo)
            self.repo.complete_job(job_id, card)
        except (ScorecardBuildError, RegressionError, BinningError,
                ValueError, ArithmeticError) as exc:
            self.repo.fail_job(job_id, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 后台作业必须落失败状态
            self.repo.fail_job(job_id, f"未预期错误 {type(exc).__name__}: {exc}")

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


_scheduler: JobScheduler | None = None


def get_scheduler() -> JobScheduler:
    global _scheduler
    if _scheduler is None:
        from .storage import get_repository
        _scheduler = JobScheduler(get_repository())
    return _scheduler
