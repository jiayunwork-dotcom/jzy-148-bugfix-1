import csv
import io
import os
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.scheduler import JobScheduler
from app.storage import MemoryRepository, PostgresRepository


def make_dev_csv(n: int = 3000, seed: int = 7, duplicated: bool = False) -> bytes:
    """生成带数值/类别/缺失列的开发样本。"""
    rng = np.random.default_rng(seed)
    income = rng.normal(7000, 2000, n)
    age = rng.normal(40, 10, n).clip(20, 75)
    dti = rng.beta(2, 5, n)
    debt = rng.normal(0, 1, n)
    city = rng.choice(["A", "B", "C", "D"], n, p=[.3, .3, .2, .2])
    housing = rng.choice(["rent", "mortgage", "own"], n, p=[.4, .35, .25])
    city_e = {"A": -1.3, "B": -0.1, "C": 0.7, "D": 1.2}
    house_e = {"rent": 1.0, "mortgage": 0.1, "own": -1.1}
    logit = (
        -0.6
        - 0.9 * (income - 7000) / 2000
        - 0.6 * (age - 40) / 10
        + 1.7 * (dti - 0.3) / 0.2
        + np.array([city_e[c] for c in city])
        + np.array([house_e[h] for h in housing])
    )
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    income[rng.random(n) < 0.05] = np.nan
    housing = np.where(rng.random(n) < 0.05, None, housing)
    rows = []
    for i in range(n):
        rows.append([
            int(y[i]),
            "" if np.isnan(income[i]) else round(float(income[i]), 2),
            round(float(age[i]), 2),
            round(float(dti[i]), 4),
            round(float(debt[i]), 4),
            city[i],
            housing[i] if housing[i] is not None else "",
        ])
    if duplicated:
        rows = rows + rows
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["target", "income", "age", "dti", "noise_num",
                "city", "housing"])
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def dev_bytes():
    return make_dev_csv()


@pytest.fixture
def card_bytes(dev_bytes):
    from app.parsing import build_dataset
    from app.card import build_scorecard
    ds = build_dataset(dev_bytes)
    return build_scorecard(ds, name="main", min_bin_pct=0.05, pdo=50)


@pytest.fixture
def client():
    repo = MemoryRepository()
    sched = JobScheduler(repo, max_workers=4)
    app = create_app(repo=repo, scheduler=sched)
    with TestClient(app) as c:
        c.repo = repo
        c.scheduler = sched
        yield c


# ---------------- PostgreSQL 集成测试门槛 -----------------------------------
#
# 未设置 DATABASE_URL：跳过（pytest -ra 会列出原因）；
# REQUIRE_PG_TESTS=1：缺连接串直接报错、连不上库直接失败——
# 集成测试不允许再"悄悄跳过"。

PG_SKIP_REASON = (
    "未设置 DATABASE_URL，跳过 PostgreSQL 集成测试。运行方式："
    "docker compose up -d db 后 "
    "DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard "
    "pytest tests/test_postgres_storage.py tests/test_postgres_integration.py；"
    "或 docker compose --profile test run --rm tests。"
    "设 REQUIRE_PG_TESTS=1 可让缺失/连不上库时直接失败")


def _pg_dsn_or_skip() -> str:
    dsn = os.getenv("DATABASE_URL", "")
    if not dsn:
        if os.getenv("REQUIRE_PG_TESTS") == "1":
            raise RuntimeError(
                "REQUIRE_PG_TESTS=1 要求必须执行 PostgreSQL 集成测试，"
                "但未设置 DATABASE_URL")
        pytest.skip(PG_SKIP_REASON)
    return dsn


@pytest.fixture(scope="module")
def pg_repo():
    """连接真实 PostgreSQL（docker compose 起的库）；连不上时明确失败。"""
    dsn = _pg_dsn_or_skip()
    try:
        repo = PostgresRepository(dsn)
    except Exception as exc:
        pytest.fail(
            f"DATABASE_URL 已设置但无法连接 PostgreSQL：{exc}\n"
            f"DSN={dsn}\n请先 docker compose up -d db",
            pytrace=False)
    yield repo


# ---------------- 边建卡边打分：响应版本 == 实际算分版本 ----------------------

RACE_APPLICANT = {"income": 8200, "age": 51, "dti": 0.08,
                  "noise_num": 0.3, "city": "A", "housing": "own"}


def run_score_version_race(app, client, name, *,
                           base_pdo=20,
                           extra_pdos=(40, 60, 80),
                           n_score_threads=4,
                           seed0=1):
    """边建卡边打分：校验响应中的 card_version 就是实际算分用的版本。

    先建 v1，然后若干打分线程持续对同一申请人打分（单条/批量、缺省最新），
    主线程同时每隔约 150ms 提交一个新版本（不同 PDO => 总分必然不同）。
    全部落库后按显式版本重算基准总分，逐条核对观测记录。
    两种存储后端共用这一份校验逻辑，保证对外表现一致。
    """

    def submit(pdo, seed):
        r = client.post(
            "/jobs",
            data={"card_name": name, "pdo": str(pdo)},
            files={"file": ("dev.csv", make_dev_csv(n=800, seed=seed),
                            "text/csv")})
        assert r.status_code == 202, r.text
        return r.json()["job_id"]

    def wait_job(job_id, timeout=120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = client.get(f"/jobs/{job_id}").json()
            if body["status"] in ("succeeded", "failed"):
                return body
            time.sleep(0.05)
        raise AssertionError(f"作业 {job_id} 超时未结束")

    assert wait_job(submit(base_pdo, seed0))["status"] == "succeeded"

    stop = threading.Event()
    observations = []   # (card_version, total_score)
    errors = []
    lock = threading.Lock()

    def score_worker(use_batch):
        try:
            # 每个打分线程用独立的 TestClient，避免并发共用一个客户端
            with TestClient(app) as c:
                while not stop.is_set():
                    if use_batch:
                        r = c.post(f"/score/{name}/batch",
                                   json={"applicants": [RACE_APPLICANT]})
                        assert r.status_code == 200, r.text
                        body = r.json()
                        assert body["results"][0]["ok"], body
                        item = (body["card_version"],
                                body["results"][0]["result"]["total_score"])
                    else:
                        r = c.post(f"/score/{name}",
                                   json={"features": RACE_APPLICANT})
                        assert r.status_code == 200, r.text
                        body = r.json()
                        item = (body["card_version"], body["total_score"])
                    with lock:
                        observations.append(item)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=score_worker, args=(i % 2 == 1,))
               for i in range(n_score_threads)]
    for t in threads:
        t.start()
    try:
        job_ids = []
        for i, pdo in enumerate(extra_pdos):
            job_ids.append(submit(pdo, seed0 + 1 + i))
            time.sleep(0.15)  # 每隔约 150ms 落一个新版本
        for jid in job_ids:
            assert wait_job(jid)["status"] == "succeeded"
        time.sleep(0.2)  # 让打分线程覆盖最后一次提交后的窗口
    finally:
        stop.set()
        for t in threads:
            t.join()

    assert not errors
    assert len(observations) >= 100, f"打分请求压得太少: {len(observations)}"

    versions = client.get(f"/cards/{name}/versions").json()
    n_versions = len(versions)
    assert n_versions == 1 + len(extra_pdos)

    # 全部版本落库后，显式指定版本重算基准总分
    expected = {}
    for v in range(1, n_versions + 1):
        r = client.post(f"/score/{name}?version={v}",
                        json={"features": RACE_APPLICANT})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["card_version"] == v
        expected[v] = body["total_score"]
    # 不同 PDO 下各版本总分必须互不相同，否则版本对错无从分辨
    assert len(set(expected.values())) == n_versions

    for reported, total in observations:
        assert reported in expected
        assert total == expected[reported], (
            f"响应标版本 {reported}，总分却与其它版本对得上 "
            f"({total} != 该版本的 {expected[reported]})")
    return len(observations)
