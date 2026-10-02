"""版本存储。

同一卡名可多次建卡，每次生成递增新版本，保留分箱、系数、分值表与训练指标。
提供两种实现：
* MemoryRepository：进程内字典，默认后端，pytest 使用；
* PostgresRepository：PostgreSQL 16。

两种实现对外语义一致：
* 版本号在建卡完成、写库时按完成先后分配，从 1 开始稠密递增；
  版本分配、写卡、作业状态更新是不可分的一步——作业成功则版本与状态
  同时生效，建卡失败则不留任何版本痕迹（失败作业不占号）；
* 同名卡的版本分配串行进行（内存实现用同一把锁，PostgreSQL 用事务级
  咨询锁），不同卡名互不阻塞；
* get_card_with_version 在同一次读取中确定卡内容与版本号，缺省最新时
  两者必然对应，读取间隙有新版本落库也不会错位。
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
    def get_card_with_version(self, name: str, version: int | None = None
                              ) -> tuple[dict[str, Any], int]:
        """取卡并同时返回本次实际使用的版本号。

        卡内容与版本号在同一次读取中确定：version=None（缺省最新）时，
        返回的版本号就是所返回那张卡的版本；读取间隙有新版本落库也不会
        出现"卡是旧版本、版本号却是新版本"的错位。在线打分必须用它。
        """
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
        with self._lock:
            return self._save_card_locked(card)

    def _save_card_locked(self, card: dict[str, Any]) -> int:
        versions = self.cards.setdefault(card["name"], [])
        version = len(versions) + 1
        versions.append({"version": version, "created_at": utcnow(),
                         "card": to_jsonable(card)})
        return version

    def complete_job(self, job_id: str, card: dict[str, Any]) -> int:
        # 版本分配与作业状态更新在同一把锁内完成，与 PostgreSQL 实现的
        # 单事务语义一致：成功作业必有对应版本，失败作业不占号
        with self._lock:
            version = self._save_card_locked(card)
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
        return self.get_card_with_version(name, version)[0]

    def get_card_with_version(self, name: str, version: int | None = None
                              ) -> tuple[dict[str, Any], int]:
        with self._lock:
            versions = self.cards.get(name)
            if not versions:
                raise KeyError(f"卡 {name} 不存在")
            if version is None:
                rec = versions[-1]
            elif version < 1 or version > len(versions):
                raise KeyError(f"卡 {name} 版本 {version} 不存在")
            else:
                rec = versions[version - 1]
            return dict(rec["card"]), rec["version"]

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
        from psycopg2.pool import ThreadedConnectionPool

        self._psycopg2 = psycopg2
        # 调度器工作线程与 HTTP 线程会并发取连接，必须用线程安全的池；
        # 容量留够 建卡 worker + 在线打分 的并发余量
        self._pool = ThreadedConnectionPool(1, 16, dsn)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    def _conn(self):
        class _CM:
            def __init__(self, pool):
                self.pool = pool
                self.conn = None

            def __enter__(self):
                self.conn = self.pool.getconn()
                return self.conn

            def __exit__(self, exc_type, exc, tb):
                try:
                    if exc_type is not None:
                        self.conn.rollback()
                    else:
                        # 提交只读语句遗留的事务，避免连接挂着旧快照回池
                        self.conn.commit()
                finally:
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
        with self._conn() as conn:
            version = self._insert_card(conn, card)
            conn.commit()
        return version

    @staticmethod
    def _insert_card(conn, card: dict[str, Any]) -> int:
        """在调用方的事务内分配版本号并写卡，返回版本号。

        同名卡的版本分配用事务级咨询锁串行化：锁随事务提交/回滚释放，
        并发建卡不会读到相同的 max(version)，也就不会撞
        (card_name, version) 唯一约束；不同卡名锁键不同，互不阻塞。
        注意不能对 max() 聚合查询加 FOR UPDATE（PostgreSQL 直接拒绝），
        且行锁也锁不住"尚不存在的下一版本"，所以用咨询锁。
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (card["name"],))
            cur.execute(
                "SELECT max(version) FROM scorecards WHERE card_name=%s",
                (card["name"],))
            version = (cur.fetchone()[0] or 0) + 1
            cur.execute(
                "INSERT INTO scorecards (card_name, version, card) "
                "VALUES (%s,%s,%s)",
                (card["name"], version, json.dumps(to_jsonable(card))))
        return version

    def complete_job(self, job_id: str, card: dict[str, Any]) -> int:
        # 版本分配、写卡、作业状态更新在同一个事务里：要么全部生效，
        # 要么全部回滚——不会出现"作业已标成功/失败，版本却半截"的中间态
        with self._conn() as conn:
            version = self._insert_card(conn, card)
            with conn.cursor() as cur:
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

    def _fetch_card(self, cur, name: str, version: int | None
                    ) -> tuple[dict[str, Any], int]:
        if version is None:
            cur.execute(
                "SELECT card, version FROM scorecards WHERE card_name=%s "
                "ORDER BY version DESC LIMIT 1", (name,))
        else:
            cur.execute(
                "SELECT card, version FROM scorecards "
                "WHERE card_name=%s AND version=%s", (name, version))
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"卡 {name} 版本 {version} 不存在" if version is not None
                else f"卡 {name} 不存在")
        return row[0], row[1]

    def get_card(self, name: str, version: int | None = None) -> dict[str, Any]:
        with self._conn() as conn, conn.cursor() as cur:
            return self._fetch_card(cur, name, version)[0]

    def get_card_with_version(self, name: str, version: int | None = None
                              ) -> tuple[dict[str, Any], int]:
        # 卡与版本号来自同一条查询，缺省最新时两者必然对应
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
