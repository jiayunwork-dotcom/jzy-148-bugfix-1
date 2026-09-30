"""HTTP 接口与作业调度测试（内存存储 + FastAPI TestClient）。"""
import io
import csv
import time
import threading
import numpy as np

from tests.conftest import make_dev_csv


def _wait_job(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        status = r.json()["status"]
        if status in ("succeeded", "failed"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError("作业超时未结束")


def test_submit_build_and_score_flow(client):
    data = make_dev_csv(seed=11)
    r = client.post(
        "/jobs",
        data={"card_name": "card_a", "min_bin_pct": "0.05"},
        files={"file": ("dev.csv", data, "text/csv")})
    assert r.status_code == 202, r.text
    job = r.json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "succeeded", final
    assert final["version"] == 1

    r = client.get("/cards/card_a")
    assert r.status_code == 200
    card = r.json()
    assert {"ks", "auc", "feature_iv"} <= set(card["metrics"])

    applicant = {"income": 9000, "age": 55, "dti": 0.05,
                 "noise_num": 0.1, "city": "A", "housing": "own"}
    r = client.post("/score/card_a", json={"features": applicant})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["pd"] < 1
    assert abs(sum(f["points"] for f in body["features"])
               - body["total_score"]) < 1e-9
    assert len(body["features"]) == len(card["selected_features"])
    for f in body["features"]:
        assert "bin_index" in f and "bin_label" in f


def test_versioning_latest_default_and_explicit(client):
    for seed in (1, 2):
        r = client.post(
            "/jobs",
            data={"card_name": "ver"},
            files={"file": ("d.csv", make_dev_csv(seed=seed), "text/csv")})
        assert _wait_job(client, r.json()["job_id"])["status"] == "succeeded"
    versions = client.get("/cards/ver/versions").json()
    assert [v["version"] for v in versions] == [1, 2]
    assert client.get("/cards/ver?version=1").status_code == 200
    r = client.post("/score/ver", json={"features": {
        "income": 5000, "age": 25, "dti": 0.6,
        "noise_num": 0.0, "city": "D", "housing": "rent"}})
    assert r.status_code == 200
    assert r.json()["card_version"] == 2
    with client as c:
        pass
    missing = client.get("/cards/ver?version=99")
    assert missing.status_code == 404


def test_pre_job_rejections(client):
    # 样本少于 500 行
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["target", "x"])
    for i in range(300):
        w.writerow([i % 2, i])
    r = client.post("/jobs", data={"card_name": "small"},
                    files={"file": ("x.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 400 and r.json()["detail"]["rejected"]

    # PDO 非正
    r = client.post(
        "/jobs",
        data={"card_name": "b", "pdo": "-50"},
        files={"file": ("d.csv", make_dev_csv(), "text/csv")})
    assert r.status_code == 400

    # 入模特征不存在
    r = client.post(
        "/jobs",
        data={"card_name": "c", "features": "nope"},
        files={"file": ("d.csv", make_dev_csv(), "text/csv")})
    assert r.status_code == 400

    # 标签全同一类
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["target", "x"])
    for i in range(600):
        w.writerow([0, i])
    r = client.post("/jobs", data={"card_name": "d"},
                    files={"file": ("x.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 400
    # 被拒绝的作业不应入库
    assert client.get("/jobs").json() == []


def test_batch_single_error_isolation(client):
    r = client.post("/jobs", data={"card_name": "b"},
                    files={"file": ("d.csv", make_dev_csv(seed=5),
                                    "text/csv")})
    assert _wait_job(client, r.json()["job_id"])["status"] == "succeeded"
    good = {"income": 8000, "age": 50, "dti": 0.1,
            "noise_num": 0.0, "city": "A", "housing": "own"}
    bad = {"income": "not-a-number", "age": 50, "dti": 0.1,
           "noise_num": 0.0, "city": "A", "housing": "own"}
    unseen = {"income": 8000, "age": 50, "dti": 0.1,
              "noise_num": 0.0, "city": "MARS", "housing": "own"}
    r = client.post("/score/b/batch",
                    json={"applicants": [good, bad, unseen]})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and body["succeeded"] == 2 and body["failed"] == 1
    assert body["results"][0]["ok"]
    assert not body["results"][1]["ok"]
    unseen_result = body["results"][2]["result"]
    assert unseen_result["has_unseen"] is True


def test_concurrent_jobs_isolation(client):
    jobs = []
    for seed in range(4):
        r = client.post(
            "/jobs",
            data={"card_name": f"iso_{seed}", "min_bin_pct": "0.05"},
            files={"file": (f"d{seed}.csv", make_dev_csv(seed=100 + seed),
                            "text/csv")})
        assert r.status_code == 202
        jobs.append((f"iso_{seed}", r.json()["job_id"]))
    for name, jid in jobs:
        final = _wait_job(client, jid)
        assert final["status"] == "succeeded", final
        assert final["card_name"] == name
    cards = {n: client.get(f"/cards/{n}").json() for n, _ in jobs}
    for name, card in cards.items():
        # 每张卡的训练样本数和特征分箱各自独立
        assert card["training"]["n_samples"] == 3000
        assert set(card["binning"]).issubset(
            {"income", "age", "dti", "noise_num", "city", "housing"})


def test_user_specified_features(client):
    r = client.post(
        "/jobs",
        data={"card_name": "manual", "features": "city,dti"},
        files={"file": ("d.csv", make_dev_csv(seed=9), "text/csv")})
    jid = r.json()["job_id"]
    final = _wait_job(client, jid)
    assert final["status"] == "succeeded", final
    card = client.get("/cards/manual").json()
    assert card["selected_features"] == ["city", "dti"]
    # 中间结果可查：被排除特征也保留在 artifacts 中
    assert "income" in card["artifacts"]["all_feature_binning"]
    trace = card["artifacts"]["all_feature_binning"]["dti"]["merge_trace"]
    assert isinstance(trace, list)


def test_failed_job_records_reason(client, monkeypatch):
    """牛顿不收敛时作业失败并记下原因。"""
    from app import scheduler as sched_mod
    from app import card as card_mod

    def boom(*a, **k):
        raise card_mod.ScorecardBuildError("牛顿迭代 100 次仍未收敛")

    monkeypatch.setattr(sched_mod, "build_scorecard", boom)
    r = client.post("/jobs", data={"card_name": "fail"},
                    files={"file": ("d.csv", make_dev_csv(), "text/csv")})
    final = _wait_job(client, r.json()["job_id"])
    assert final["status"] == "failed"
    assert "牛顿迭代" in final["error"]
