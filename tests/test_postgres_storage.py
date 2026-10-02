"""PostgreSQL 存储集成测试（需要 docker compose 启动的真实库）。

    docker compose up -d db
    DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard \
        pytest tests/test_postgres_storage.py

连接门槛见 tests/conftest.py：未设 DATABASE_URL 时跳过并给出原因；
REQUIRE_PG_TESTS=1 时缺连接串/连不上库直接失败，不再静默跳过。
"""
import threading
import uuid


def _card(name: str, pdo: float | None = None):
    return {
        "name": name,
        "selected_features": ["x"],
        "score_table": {"factor": 1.0, "offset": 2.0,
                        "per_feature_offset": 2.0,
                        "intercept_share": 0.1, "features": {}},
        "metrics": {"ks": 0.4, "auc": 0.8, "feature_iv": {"x": 0.1}},
        "binning": {}, "coefficients": {"intercept": 0.0, "x": 1.0},
        "config": {} if pdo is None else {"pdo": pdo},
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


def test_get_card_with_version(pg_repo):
    """版本解析与取卡一次完成：返回的版本号必然属于取回的那张卡。"""
    name = "pg_" + uuid.uuid4().hex[:10]
    pg_repo.save_card(_card(name, pdo=20))
    card, version = pg_repo.get_card_with_version(name)
    assert version == 1 and card["config"]["pdo"] == 20
    pg_repo.save_card(_card(name, pdo=40))
    card, version = pg_repo.get_card_with_version(name)
    assert version == 2 and card["config"]["pdo"] == 40
    card, version = pg_repo.get_card_with_version(name, 1)
    assert version == 1 and card["config"]["pdo"] == 20


def test_concurrent_version_increment(pg_repo):
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


def test_concurrent_complete_job_dense_versions(pg_repo):
    """并发 complete_job：全部成功、版本恰好 1..N、作业版本与卡参数对应。"""
    name = "pg_cjob_" + uuid.uuid4().hex[:10]
    pdos = (20, 30, 40, 50, 60, 70)
    job_ids = [pg_repo.create_job(name, {"pdo": p}) for p in pdos]
    barrier = threading.Barrier(len(job_ids))
    errors = []

    def worker(jid, pdo):
        try:
            barrier.wait(timeout=60)
            pg_repo.complete_job(jid, _card(name, pdo=pdo))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(jid, p))
               for jid, p in zip(job_ids, pdos)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    versions = [v["version"] for v in pg_repo.list_versions(name)]
    assert sorted(versions) == [1, 2, 3, 4, 5, 6]
    for jid, pdo in zip(job_ids, pdos):
        job = pg_repo.get_job(jid)
        assert job["status"] == "succeeded"
        # 作业记录上的版本号 ↔ 该版本卡里的建卡参数一一对应
        card = pg_repo.get_card(name, job["version"])
        assert card["config"]["pdo"] == pdo


def test_job_lifecycle(pg_repo):
    jid = pg_repo.create_job("pgjob_" + uuid.uuid4().hex[:8], {"p": 1})
    assert pg_repo.get_job(jid)["status"] == "queued"
    pg_repo.mark_job_running(jid)
    assert pg_repo.get_job(jid)["status"] == "running"
    pg_repo.fail_job(jid, "牛顿迭代 100 次仍未收敛")
    job = pg_repo.get_job(jid)
    assert job["status"] == "failed" and "牛顿" in job["error"]
    assert job["version"] is None
