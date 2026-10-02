"""PostgreSQL 集成测试：需要真实数据库（如 docker compose 起的库）。

运行方式：
    docker compose up -d db
    DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard \
        pytest tests/test_postgres_storage.py

* 未设置 DATABASE_URL：整文件跳过，pytest 摘要（-ra）里能看到原因；
* 设置了但连不上库：测试直接失败并打出连接错误，绝不静默跳过。
"""
import os
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.scheduler import JobScheduler
from app.storage import PostgresRepository
from tests.conftest import make_dev_csv

DATABASE_URL = os.getenv("DATABASE_URL")


def _new_repo() -> PostgresRepository:
    if not DATABASE_URL:
        pytest.skip("未设置 DATABASE_URL，跳过 PostgreSQL 集成测试"
                    "（docker compose up -d db 后带上 DATABASE_URL 重跑）")
    try:
        return PostgresRepository(DATABASE_URL)
    except Exception as exc:  # noqa: BLE001 - 连接失败必须明确报出来
        pytest.fail(f"DATABASE_URL 已设置但无法连接/初始化 PostgreSQL "
                    f"({DATABASE_URL}): {exc}", pytrace=False)


@pytest.fixture(scope="module")
def pg_repo():
    yield _new_repo()


@pytest.fixture()
def pg_app(pg_repo):
    sched = JobScheduler(pg_repo, max_workers=4)
    yield create_app(repo=pg_repo, scheduler=sched)
    sched.shutdown()


@pytest.fixture()
def pg_client(pg_app):
    with TestClient(pg_app) as client:
        yield client


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


def _wait_job(client, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"作业 {job_id} 超时未结束")


def _submit(client, name, data, **form):
    r = client.post("/jobs", data={"card_name": name, **form},
                    files={"file": ("dev.csv", data, "text/csv")})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


# ---------------- 存储层 ----------------

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
    name = "pgjob_" + uuid.uuid4().hex[:8]
    jid = pg_repo.create_job(name, {"p": 1})
    assert pg_repo.get_job(jid)["status"] == "queued"
    pg_repo.mark_job_running(jid)
    assert pg_repo.get_job(jid)["status"] == "running"
    pg_repo.fail_job(jid, "牛顿迭代 100 次仍未收敛")
    job = pg_repo.get_job(jid)
    assert job["status"] == "failed" and "牛顿" in job["error"]
    # 失败作业不占版本号，也不留半截版本
    assert job["version"] is None
    assert pg_repo.list_versions(name) == []


def test_complete_job_atomic_version_and_params(pg_repo):
    """complete_job 是一步：作业上的版本号与取回的卡内容一一对应。"""
    name = "pg_atomic_" + uuid.uuid4().hex[:8]
    jid = pg_repo.create_job(name, {"pdo": 33.0, "min_bin_pct": 0.05})
    pg_repo.mark_job_running(jid)
    card = _card(name)
    card["config"] = {"pdo": 33.0}
    version = pg_repo.complete_job(jid, card)
    assert version == 1
    job = pg_repo.get_job(jid)
    assert job["status"] == "succeeded" and job["version"] == 1
    got = pg_repo.get_card(name, job["version"])
    assert got["config"]["pdo"] == job["params"]["pdo"] == 33.0


def test_get_card_with_version_consistent(pg_repo):
    """缺省最新时，返回的版本号就是返回的那张卡的版本。"""
    name = "pg_gcwv_" + uuid.uuid4().hex[:8]
    pg_repo.save_card(_card(name))
    pg_repo.save_card(_card(name))
    card, version = pg_repo.get_card_with_version(name)
    assert version == 2
    assert card == pg_repo.get_card(name, 2)
    card1, version1 = pg_repo.get_card_with_version(name, 1)
    assert version1 == 1 and card1 == pg_repo.get_card(name, 1)
    with pytest.raises(KeyError):
        pg_repo.get_card_with_version(name, 99)
    with pytest.raises(KeyError):
        pg_repo.get_card_with_version("pg_no_such_" + uuid.uuid4().hex[:8])


# ---------------- 全链路（HTTP + 调度器 + PostgreSQL） ----------------

def test_concurrent_same_name_jobs_all_succeed(pg_client):
    """同一卡名并发提交 N 个合法作业：全部成功，版本号恰好 1..N 不重不缺，
    且每个作业记录上的版本号与该版本卡里的建卡参数一一对应。"""
    name = "pg_concjob_" + uuid.uuid4().hex[:8]
    data = make_dev_csv(n=1000, seed=42)
    n_jobs = 6
    pdos = [20.0 + 10.0 * i for i in range(n_jobs)]  # 各带不同 PDO 便于辨认
    barrier = threading.Barrier(n_jobs)
    responses = [None] * n_jobs
    errors = []

    def submit(i):
        try:
            barrier.wait(timeout=10)
            responses[i] = pg_client.post(
                "/jobs", data={"card_name": name, "pdo": str(pdos[i])},
                files={"file": ("dev.csv", data, "text/csv")})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(i,))
               for i in range(n_jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert all(r is not None and r.status_code == 202 for r in responses)

    jobs = [_wait_job(pg_client, r.json()["job_id"]) for r in responses]
    assert all(j["status"] == "succeeded" for j in jobs), jobs
    # 版本号恰好 1..N，不重不缺
    assert sorted(j["version"] for j in jobs) == list(range(1, n_jobs + 1))
    listed = pg_client.get(f"/cards/{name}/versions").json()
    assert [v["version"] for v in listed] == list(range(1, n_jobs + 1))
    # 作业记录上的版本号 ↔ 该版本卡里的建卡参数一一对应
    for job, pdo in zip(jobs, pdos):
        assert float(job["params"]["pdo"]) == pdo
        card = pg_client.get(f"/cards/{name}?version={job['version']}").json()
        assert float(card["config"]["pdo"]) == pdo


def test_concurrent_different_card_names(pg_client):
    """不同卡名的作业并发提交：全部成功，各自拿到版本 1。"""
    data = make_dev_csv(n=1000, seed=7)
    names = [f"pg_multi_{i}_" + uuid.uuid4().hex[:6] for i in range(4)]
    job_ids = []
    errors = []
    lock = threading.Lock()

    def submit(n):
        try:
            jid = _submit(pg_client, n, data)
            with lock:
                job_ids.append((n, jid))
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=submit, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert len(job_ids) == len(names)
    for n, jid in job_ids:
        final = _wait_job(pg_client, jid)
        assert final["status"] == "succeeded", final
        assert final["version"] == 1


def test_failed_job_leaves_no_version(pg_client, monkeypatch):
    """建卡途中失败：作业标失败，但不留任何版本痕迹。"""
    from app import scheduler as sched_mod
    from app import card as card_mod

    def boom(*a, **k):
        raise card_mod.ScorecardBuildError("模拟建卡中途失败")

    monkeypatch.setattr(sched_mod, "build_scorecard", boom)
    name = "pg_fail_" + uuid.uuid4().hex[:8]
    jid = _submit(pg_client, name, make_dev_csv(n=1000, seed=3))
    final = _wait_job(pg_client, jid)
    assert final["status"] == "failed"
    assert final["version"] is None
    assert pg_client.get(f"/cards/{name}/versions").json() == []
    assert pg_client.get(f"/cards/{name}").status_code == 404


def test_score_version_consistency_while_building(pg_app, pg_client):
    """边建卡边打分：响应里报的版本号必须就是算分实际用的那张卡。

    各版本用互不相同的 PDO，保证同一申请人在不同版本下总分互异——
    一旦"报的版本"与"用的卡"错位，总分必然对不上。
    单条与批量、缺省最新与显式指定都覆盖。
    """
    name = "pg_score_" + uuid.uuid4().hex[:8]
    data = make_dev_csv(n=1000, seed=99)
    applicant = {"income": 8200, "age": 51, "dti": 0.12,
                 "noise_num": 0.0, "city": "A", "housing": "own"}
    extra_pdos = [20.0, 30.0, 40.0, 60.0, 70.0, 80.0]

    # 先建 v1（pdo=50），保证打分线程启动时已有可打的版本
    first = _submit(pg_client, name, data, pdo="50")
    assert _wait_job(pg_client, first)["status"] == "succeeded"

    stop = threading.Event()
    observations = []  # (接口, 响应里的 card_version, total_score)
    obs_lock = threading.Lock()
    failures = []

    def scorer():
        # 每个打分线程用自己的 client，避免跨线程共享连接
        with TestClient(pg_app) as client:
            while not stop.is_set():
                try:
                    r = client.post(f"/score/{name}",
                                    json={"features": applicant})
                    assert r.status_code == 200, r.text
                    body = r.json()
                    with obs_lock:
                        observations.append(
                            ("single", body["card_version"],
                             body["total_score"]))
                    rb = client.post(f"/score/{name}/batch",
                                     json={"applicants": [applicant] * 2})
                    assert rb.status_code == 200, rb.text
                    batch = rb.json()
                    for item in batch["results"]:
                        assert item["ok"], item
                        with obs_lock:
                            observations.append(
                                ("batch", batch["card_version"],
                                 item["result"]["total_score"]))
                except Exception as exc:  # noqa: BLE001
                    failures.append(exc)
                    stop.set()

    scorers = [threading.Thread(target=scorer) for _ in range(4)]
    for t in scorers:
        t.start()
    try:
        # 边打分边持续出新版本
        for pdo in extra_pdos:
            jid = _submit(pg_client, name, data, pdo=str(pdo))
            final = _wait_job(pg_client, jid)
            assert final["status"] == "succeeded", final
            time.sleep(0.1)
    finally:
        stop.set()
        for t in scorers:
            t.join(timeout=30)
    assert not failures

    # 基准：显式指定版本重新打分；同时验证显式指定时回的版本号
    versions = [v["version"]
                for v in pg_client.get(f"/cards/{name}/versions").json()]
    assert versions == list(range(1, 2 + len(extra_pdos)))
    truth = {}
    for v in versions:
        r = pg_client.post(f"/score/{name}?version={v}",
                           json={"features": applicant})
        assert r.status_code == 200
        assert r.json()["card_version"] == v
        truth[v] = r.json()["total_score"]
    # 前提：各版本总分互不相同，版本标错一定会暴露
    assert len(set(truth.values())) == len(truth)

    # 测试确实在版本切换期间打了足够多的分
    assert len(observations) > 100
    assert len({v for _, v, _ in observations}) >= 2
    # 每一次响应：报的版本号对应的显式打分结果，必须与当时返回的总分一致
    for api_kind, v, total in observations:
        assert total == truth[v], (
            f"{api_kind} 响应标的是版本 {v}，总分却对不上（疑似用了别的版本）")
