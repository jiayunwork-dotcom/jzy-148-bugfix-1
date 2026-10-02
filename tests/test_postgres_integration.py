"""PostgreSQL 端到端集成测试：在 docker compose 启动的真实库上运行。

覆盖三层修复后的验收行为：
1. 同一卡名并发提交 N 个合法作业：全部成功，版本号恰好 1..N，
   每个作业记录上的版本与取回卡里的建卡参数一一对应；
2. 不同卡名的作业并发执行，互不阻塞；
3. 建卡中途失败的作业不留半截版本、不占号；
4. 边建卡边打分：响应中的 card_version 必须是本次实际算分用的版本
   （单条 / 批量 / 缺省最新 / 显式指定）。

运行方式见 tests/conftest.py 的 PG_SKIP_REASON；连不上库会明确失败，
不会静默跳过。
"""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.scheduler import JobScheduler
from tests.conftest import make_dev_csv, run_score_version_race


@pytest.fixture()
def pg_app(pg_repo):
    sched = JobScheduler(pg_repo, max_workers=6)
    app = create_app(repo=pg_repo, scheduler=sched)
    yield app
    sched.shutdown()


@pytest.fixture()
def pg_client(pg_app):
    with TestClient(pg_app) as client:
        yield client


def _submit(client, name, pdo, seed, n=800):
    return client.post(
        "/jobs",
        data={"card_name": name, "pdo": str(pdo)},
        files={"file": ("dev.csv", make_dev_csv(n=n, seed=seed), "text/csv")})


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


def _submit_concurrently(pg_app, tasks):
    """tasks: [(name, pdo, seed), ...]；全部线程就绪后同时提交。

    返回 {task: 响应 json}；任何线程出错都汇总到 errors 抛出。
    """
    barrier = threading.Barrier(len(tasks))
    submissions = {}
    errors = []

    def worker(task):
        name, pdo, seed = task
        try:
            # 每个提交线程用独立的 TestClient
            with TestClient(pg_app) as client:
                barrier.wait(timeout=60)
                r = _submit(client, name, pdo, seed)
                assert r.status_code == 202, r.text
                submissions[task] = r.json()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    return submissions


def test_concurrent_same_card_all_succeed_versions_dense(pg_app, pg_client):
    """同一卡名并发 6 个作业：全部成功，版本恰好 1..6，作业↔版本↔参数对应。"""
    name = "pgc_" + uuid.uuid4().hex[:8]
    pdos = [20, 30, 40, 50, 60, 70]
    tasks = [(name, pdo, 100 + i) for i, pdo in enumerate(pdos)]
    submissions = _submit_concurrently(pg_app, tasks)

    finals = {task: _wait_job(pg_client, sub["job_id"])
              for task, sub in submissions.items()}
    assert all(f["status"] == "succeeded" for f in finals.values()), finals

    versions = pg_client.get(f"/cards/{name}/versions").json()
    assert sorted(v["version"] for v in versions) == [1, 2, 3, 4, 5, 6]

    for (task, final) in finals.items():
        _, pdo, _ = task
        version = final["version"]
        # 作业记录上的版本号，和取回来的那张卡里的建卡参数一一对应
        card = pg_client.get(f"/cards/{name}?version={version}").json()
        assert card["config"]["pdo"] == pdo
        job = pg_client.get(f"/jobs/{final['job_id']}").json()
        assert job["params"]["pdo"] == pdo


def test_concurrent_different_cards_do_not_block(pg_app, pg_client):
    """不同卡名的作业并发执行：全部成功，且时间区间有重叠（不互相卡住）。"""
    names = [f"pgd{i}_" + uuid.uuid4().hex[:6] for i in range(4)]
    tasks = [(name, 50, 200 + i) for i, name in enumerate(names)]
    submissions = _submit_concurrently(pg_app, tasks)

    finals = {task: _wait_job(pg_client, sub["job_id"])
              for task, sub in submissions.items()}
    assert all(f["status"] == "succeeded" for f in finals.values()), finals
    for (name, _, _), final in finals.items():
        assert final["version"] == 1

    intervals = sorted(
        (datetime.fromisoformat(f["started_at"]),
         datetime.fromisoformat(f["finished_at"]))
        for f in finals.values())
    overlap = any(intervals[i + 1][0] < intervals[i][1]
                  for i in range(len(intervals) - 1))
    assert overlap, f"不同卡名的作业疑似互相阻塞: {intervals}"


def test_failed_job_leaves_no_version_and_no_gap(pg_app, pg_client,
                                                 monkeypatch):
    """建卡中途失败：不留半截版本、不占号；后续成功建卡版本连续。"""
    from app import scheduler as sched_mod
    from app.card import ScorecardBuildError

    name = "pgf_" + uuid.uuid4().hex[:8]
    r = _submit(pg_client, name, 50, 1)
    assert _wait_job(pg_client, r.json()["job_id"])["status"] == "succeeded"

    def boom(*a, **k):
        raise ScorecardBuildError("模拟建卡中途失败")

    monkeypatch.setattr(sched_mod, "build_scorecard", boom)
    r = _submit(pg_client, name, 60, 2)
    final = _wait_job(pg_client, r.json()["job_id"])
    assert final["status"] == "failed"
    assert final["version"] is None
    monkeypatch.undo()

    # 失败的作业没有留下版本
    versions = pg_client.get(f"/cards/{name}/versions").json()
    assert [v["version"] for v in versions] == [1]

    # 失败作业不占号：下一次成功建卡拿到紧邻的 2
    r = _submit(pg_client, name, 70, 3)
    final = _wait_job(pg_client, r.json()["job_id"])
    assert final["status"] == "succeeded"
    assert final["version"] == 2


def test_score_version_consistency_while_building(pg_app, pg_client):
    """边建卡边打分：响应版本必须是实际算分版本（单条/批量/缺省最新）。"""
    name = "pgs_" + uuid.uuid4().hex[:8]
    n_obs = run_score_version_race(
        pg_app, pg_client, name,
        extra_pdos=(40, 60, 80, 100, 120),
        n_score_threads=6)
    assert n_obs >= 100
