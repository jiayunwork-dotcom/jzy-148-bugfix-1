"""版本存储。

同一卡名可多次建卡，每次生成递增新版本，保留分箱、系数、分值表与训练指标。
提供两种实现：
* MemoryRepository：进程内字典，默认后端，pytest 使用；
* PostgresRepository：PostgreSQL 16，版本号在事务内对同名卡加锁递增。
"""
from __future__ import annotations

import json
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from .config import settings


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(obj: Any) -> Any:
    """递归把 numpy 标量/数组等转为 JSON 原生类型。"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) \
            else None
    return obj


class Repository(ABC):
    @abstractmethod
    def create_job(self, name: str, params: dict[str, Any]) -> str: ...

    @abstractmethod
    def mark_job_running(self, job_id: str) -> None: ...

    @abstractmethod
    def complete_job(self, job_id: str, card: dict[str, Any]) -> int: ...

    @abstractmethod
    def fail_job(self, job_id: str, reason: str) -> None: ...

    @abstractmethod
    def get_job(self, job_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def list_jobs(self, name: str | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def save_card(self, card: dict[str, Any]) -> int: ...

    @abstractmethod
    def get_card(self, name: str, version: int | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def list_versions(self, name: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def list_cards(self) -> list[dict[str, Any]]: ...


class MemoryRepository(Repository):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.cards: dict[str, list[dict[str, Any]]] = {}

    def create_job(self, name: str, params: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "card_name": name,
                "status": "queued",
                "params": params,
                "created_at": utcnow(),
                "started_at": None,
                "finished_at": None,
                "version": None,
                "error": None,
            }
        return job_id

    def mark_job_running(self, job_id: str) -> None:
        with self._lock:
            self.jobs[job_id].update(
                status="running", started_at=utcnow())

    def save_card(self, card: dict[str, Any]) -> int:
        name = card["name"]
        with self._lock:
            versions = self.cards.setdefault(name, [])
            version = len(versions) + 1
            stored = {"version": version, "created_at": utcnow(),
                      "card": to_jsonable(card)}
            versions.append(stored)
        return version

    def complete_job(self, job_id: str, card: dict[str, Any]) -> int:
        version = self.save_card(card)
        with self._lock:
            self.jobs[job_id].update(
                status="succeeded", finished_at=utcnow(), version=version)
        return version

    def fail_job(self, job_id: str, reason: str) -> None:
        with self._lock:
            self.jobs[job_id].update(
                status="failed", finished_at=utcnow(), error=reason)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def list_jobs(self, name: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(j) for j in self.jobs.values()
                    if name is None or j["card_name"] == name]

    def get_card(self, name: str, version: int | None = None) -> dict[str, Any]:
        with self._lock:
            versions = self.cards.get(name)
            if not versions:
                raise KeyError(f"卡 {name} 不存在")
            if version is None:
                return dict(versions[-1]["card"])
            if version < 1 or version > len(versions):
                raise KeyError(f"卡 {name} 版本 {version} 不存在")
            return dict(versions[version - 1]["card"])

    def get_card_meta(self, name: str, version: int | None = None
                      ) -> dict[str, Any]:
        with self._lock:
            versions = self.cards.get(name)
            if not versions:
                raise KeyError(f"卡 {name} 不存在")
            rec = versions[-1] if version is None else versions[version - 1]
            return {"card_name": name, "version": rec["version"],
                    "created_at": rec["created_at"]}

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            versions = self.cards.get(name, [])
            return [{
                "card_name": name,
                "version": v["version"],
                "created_at": v["created_at"],
                "metrics": v["card"]["metrics"],
                "n_features": len(v["card"]["selected_features"]),
            } for v in versions]

    def list_cards(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{
                "card_name": name,
                "latest_version": len(versions),
                "created_at": versions[-1]["created_at"],
            } for name, versions in self.cards.items()]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scorecard_jobs (
    job_id      TEXT PRIMARY KEY,
    card_name   TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      JSONB NOT NULL,
    error       TEXT,
    version     INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS scorecards (
    id         BIGSERIAL PRIMARY KEY,
    card_name  TEXT NOT NULL,
    version    INTEGER NOT NULL,
    card       JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (card_name, version)
);
"""


class PostgresRepository(Repository):
    def __init__(self, dsn: str) -> None:
        import psycopg2
        from psycopg2.pool import SimpleConnectionPool

        self._psycopg2 = psycopg2
        self._pool = SimpleConnectionPool(1, 8, dsn)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    def _conn(self):
        psycopg2 = self._psycopg2

        class _CM:
            def __init__(self, pool):
                self.pool = pool
                self.conn = None

            def __enter__(self):
                self.conn = self.pool.getconn()
                return self.conn

            def __exit__(self, exc_type, exc, tb):
                if exc_type is not None:
                    self.conn.rollback()
                self.pool.putconn(self.conn)

        return _CM(self._pool)

    def create_job(self, name: str, params: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scorecard_jobs "
                "(job_id, card_name, status, params) VALUES (%s,%s,'queued',%s)",
                (job_id, name, json.dumps(to_jsonable(params))))
            conn.commit()
        return job_id

    def mark_job_running(self, job_id: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE scorecard_jobs SET status='running', "
                "started_at=now() WHERE job_id=%s", (job_id,))
            conn.commit()

    def save_card(self, card: dict[str, Any]) -> int:
        with self._conn() as conn, conn.cursor() as cur:
            # 事务行锁保证同名卡并发建卡时版本号不冲突
            cur.execute(
                "SELECT max(version) FROM scorecards WHERE card_name=%s "
                "FOR UPDATE", (card["name"],))
            latest = cur.fetchone()[0] or 0
            version = latest + 1
            cur.execute(
                "INSERT INTO scorecards (card_name, version, card) "
                "VALUES (%s,%s,%s)",
                (card["name"], version, json.dumps(to_jsonable(card))))
            conn.commit()
        return version

    def complete_job(self, job_id: str, card: dict[str, Any]) -> int:
        version = self.save_card(card)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE scorecard_jobs SET status='succeeded', "
                "finished_at=now(), version=%s WHERE job_id=%s",
                (version, job_id))
            conn.commit()
        return version

    def fail_job(self, job_id: str, reason: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE scorecard_jobs SET status='failed', "
                "finished_at=now(), error=%s WHERE job_id=%s",
                (reason, job_id))
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, card_name, status, params, error, version, "
                "created_at, started_at, finished_at FROM scorecard_jobs "
                "WHERE job_id=%s", (job_id,))
            row = cur.fetchone()
        if row is None:
            raise KeyError(job_id)
        keys = ("job_id", "card_name", "status", "params", "error", "version",
                "created_at", "started_at", "finished_at")
        return dict(zip(keys, row))

    def list_jobs(self, name: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT job_id, card_name, status, params, error, version, "
               "created_at, started_at, finished_at FROM scorecard_jobs")
        args: tuple = ()
        if name:
            sql += " WHERE card_name=%s"
            args = (name,)
        sql += " ORDER BY created_at"
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        keys = ("job_id", "card_name", "status", "params", "error", "version",
                "created_at", "started_at", "finished_at")
        return [dict(zip(keys, r)) for r in rows]

    def _fetch_card(self, cur, name: str, version: int | None) -> dict[str, Any]:
        if version is None:
            cur.execute(
                "SELECT card FROM scorecards WHERE card_name=%s "
                "ORDER BY version DESC LIMIT 1", (name,))
        else:
            cur.execute(
                "SELECT card FROM scorecards WHERE card_name=%s AND version=%s",
                (name, version))
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"卡 {name} 版本 {version} 不存在" if version
                else f"卡 {name} 不存在")
        return row[0]

    def get_card(self, name: str, version: int | None = None) -> dict[str, Any]:
        with self._conn() as conn, conn.cursor() as cur:
            return self._fetch_card(cur, name, version)

    def get_card_meta(self, name: str, version: int | None = None
                      ) -> dict[str, Any]:
        with self._conn() as conn, conn.cursor() as cur:
            if version is None:
                cur.execute(
                    "SELECT version, created_at FROM scorecards "
                    "WHERE card_name=%s ORDER BY version DESC LIMIT 1",
                    (name,))
            else:
                cur.execute(
                    "SELECT version, created_at FROM scorecards "
                    "WHERE card_name=%s AND version=%s", (name, version))
            row = cur.fetchone()
        if row is None:
            raise KeyError(name)
        return {"card_name": name, "version": row[0], "created_at": row[1]}

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT version, created_at, card FROM scorecards "
                "WHERE card_name=%s ORDER BY version", (name,))
            rows = cur.fetchall()
        return [{
            "card_name": name, "version": v, "created_at": ts,
            "metrics": card["metrics"],
            "n_features": len(card["selected_features"]),
        } for v, ts, card in rows]

    def list_cards(self) -> list[dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (card_name) card_name, version, created_at "
                "FROM scorecards ORDER BY card_name, version DESC")
            rows = cur.fetchall()
        return [{"card_name": n, "latest_version": v, "created_at": ts}
                for n, v, ts in rows]


_repo: Repository | None = None


def get_repository() -> Repository:
    """按 DATABASE_URL 选择存储后端（单例）。"""
    global _repo
    if _repo is None:
        if settings.database_url.startswith("memory://"):
            _repo = MemoryRepository()
        else:
            _repo = PostgresRepository(settings.database_url)
    return _repo


def reset_repository() -> None:
    global _repo
    _repo = None
