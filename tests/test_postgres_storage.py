"""PostgreSQL 存储集成测试。

默认跳过；设置 DATABASE_URL（如 docker compose 启动的库）后运行：
    DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard \
        pytest tests/test_postgres_storage.py
"""
import os
import uuid

import pytest

from app.storage import PostgresRepository

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="未设置 DATABASE_URL，跳过 PostgreSQL 集成测试")


@pytest.fixture(scope="module")
def pg_repo():
    repo = PostgresRepository(os.environ["DATABASE_URL"])
    yield repo


def _card(version_payload: str):
    return {
        "name": version_payload,
        "selected_features": ["x"],
        "score_table": {"factor": 1.0, "offset": 2.0,
                        "per_feature_offset": 2.0,
                        "intercept_share": 0.1, "features": {}},
        "metrics": {"ks": 0.4, "auc": 0.8, "feature_iv": {"x": 0.1}},
        "binning": {}, "coefficients": {"intercept": 0.0, "x": 1.0},
        "config": {},
    }


def test_versioning_and_latest(pg_repo):
    name = "pg_" + uuid.uuid4().hex[:10]
    v1 = pg_repo.save_card(_card(name))
    v2 = pg_repo.save_card(_card(name))
    assert (v1, v2) == (1, 2)
    versions = pg_repo.list_versions(name)
    assert [v["version"] for v in versions] == [1, 2]
    got = pg_repo.get_card(name)
    assert got["metrics"]["auc"] == 0.8


def test_concurrent_version_increment(pg_repo):
    import threading

    name = "pg_conc_" + uuid.uuid4().hex[:10]
    errors = []

    def worker():
        try:
            pg_repo.save_card(_card(name))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    versions = [v["version"] for v in pg_repo.list_versions(name)]
    assert sorted(versions) == [1, 2, 3, 4, 5, 6]


def test_job_lifecycle(pg_repo):
    jid = pg_repo.create_job("pgjob_" + uuid.uuid4().hex[:8], {"p": 1})
    assert pg_repo.get_job(jid)["status"] == "queued"
    pg_repo.mark_job_running(jid)
    assert pg_repo.get_job(jid)["status"] == "running"
    pg_repo.fail_job(jid, "牛顿迭代 100 次仍未收敛")
    job = pg_repo.get_job(jid)
    assert job["status"] == "failed" and "牛顿" in job["error"]
