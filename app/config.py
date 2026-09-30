"""运行配置：存储后端与数据库连接。

默认使用内存存储，方便本地 pytest；Docker Compose 中通过环境变量切换到
PostgreSQL。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "memory://scorecard"
    )
    max_workers: int = int(os.getenv("JOB_WORKERS", "4"))


settings = Settings()
