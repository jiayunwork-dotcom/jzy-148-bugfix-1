import io
import csv
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.scheduler import JobScheduler
from app.storage import MemoryRepository


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
