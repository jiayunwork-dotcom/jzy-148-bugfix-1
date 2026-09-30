"""性质测试：WOE/IV 手算核对、标签反转、复制不变性、评分性质等。"""
import copy
import io
import csv
import math
import numpy as np
import pytest

from app.binning import bin_feature, woe, bin_iv
from app.card import build_scorecard, verify_monotonic
from app.metrics import roc_auc, ks_statistic
from app.parsing import build_dataset
from app.regression import fit_logistic_newton, sigmoid
from app.scoring import build_score_table, score_one


SAMPLE_PATH = "tests/data/sample_development.csv"


# ---------- 小样本手算核对（前两个特征：city / housing） --------------------

def test_city_woe_iv_hand_calc():
    ds = build_dataset(open(SAMPLE_PATH, "rb").read())
    res = bin_feature("city", ds.columns["city"], ds.labels,
                      "categorical", min_bin_pct=0.0)
    by_cat = {}
    for b in res.bins:
        for c in b["categories"]:
            by_cat[c] = b
    tg, tb = res.total_good, res.total_bad
    assert (tg, tb) == (16, 8)

    expected = {  # city: (bad, good)
        "D": (1, 6), "A": (2, 3), "B": (2, 4), "C": (3, 3),
    }
    for c, (bad, good) in expected.items():
        b = by_cat[c]
        assert b["bad"] == bad and b["good"] == good
        assert b["woe"] == pytest.approx(woe(good, bad, tg, tb), abs=1e-12)
    # 手算：ln3、ln(3/4)、0、-ln2
    assert by_cat["D"]["woe"] == pytest.approx(math.log(3), abs=1e-12)
    assert by_cat["A"]["woe"] == pytest.approx(
        math.log((3 / 16) / (2 / 8)), abs=1e-12)
    assert by_cat["B"]["woe"] == pytest.approx(0.0, abs=1e-12)
    assert by_cat["C"]["woe"] == pytest.approx(-math.log(2), abs=1e-12)
    assert res.iv == pytest.approx(
        sum(bin_iv(g, b, 16, 8) for b, g in
            ((1, 6), (2, 3), (2, 4), (3, 3))), abs=1e-12)


def test_housing_woe_iv_hand_calc():
    ds = build_dataset(open(SAMPLE_PATH, "rb").read())
    res = bin_feature("housing", ds.columns["housing"], ds.labels,
                      "categorical", min_bin_pct=0.0)
    by_cat = {c: b for b in res.bins for c in b["categories"]}
    expected = {"own": (1, 7), "mortgage": (3, 5), "rent": (4, 4)}
    for c, (bad, good) in expected.items():
        assert by_cat[c]["bad"] == bad and by_cat[c]["good"] == good
        assert by_cat[c]["woe"] == pytest.approx(
            woe(good, bad, 16, 8), abs=1e-12)
    assert by_cat["own"]["woe"] == pytest.approx(math.log(3.5), abs=1e-12)
    assert by_cat["rent"]["woe"] == pytest.approx(-math.log(2), abs=1e-12)
    assert by_cat["mortgage"]["woe"] == pytest.approx(
        math.log(5 / 6), abs=1e-12)
    assert res.iv == pytest.approx(
        sum(bin_iv(g, b, 16, 8) for b, g in
            ((1, 7), (3, 5), (4, 4))), abs=1e-12)


def test_numeric_initial_bins_at_most_20():
    ds = build_dataset(open(SAMPLE_PATH, "rb").read())
    res = bin_feature("age", [v for v in ds.columns["age"]],
                      ds.labels, "numeric", min_bin_pct=0.0)
    assert len(res.initial_bins) <= 20
    assert res.missing_bin["total"] == 1  # 样本中 age 有一个缺失
    assert res.missing_bin["missing"] is True


# ---------- 性质 1：平均预测 PD == 实际违约率 ------------------------------

def test_mean_predicted_pd_equals_bad_rate(dev_bytes):
    ds = build_dataset(dev_bytes)
    card = build_scorecard(ds, name="p1", min_bin_pct=0.05)
    assert card["training"]["mean_pd_error"] < 1e-6
    assert card["training"]["mean_predicted_pd"] == pytest.approx(
        card["training"]["bad_rate"], abs=1e-6)


# ---------- 性质 2：标签整体取反：WOE 变号、IV 不变 -----------------------

def test_label_flip_woe_sign_and_iv_invariant(dev_bytes):
    ds = build_dataset(dev_bytes)
    flipped = copy.deepcopy(ds)
    flipped.labels = [1 - y for y in ds.labels]
    flipped.bad_rate = 1 - ds.bad_rate

    card_a = build_scorecard(ds, name="a", min_bin_pct=0.05)
    card_b = build_scorecard(flipped, name="b", min_bin_pct=0.05)

    common = sorted(set(card_a["binning"]) & set(card_b["binning"]))
    assert common, "两次建卡应有共同可用特征"
    for f in common:
        ba, bb = card_a["binning"][f], card_b["binning"][f]
        # 类别箱按类别集合对齐（排序方向会反过来）
        def keyed(spec):
            if spec["feature_type"] == "categorical":
                out = {}
                for b in spec["bins"]:
                    out[tuple(sorted(b["categories"]))] = b
                return out
            return {i: b for i, b in enumerate(spec["bins"])}
        ka, kb = keyed(ba), keyed(bb)
        for k in ka:
            assert ka[k]["woe"] == pytest.approx(-kb[k]["woe"], abs=1e-10)
        # IV 完全不变
        assert ba["iv"] == pytest.approx(bb["iv"], abs=1e-10)
        if ba.get("missing_bin") and bb.get("missing_bin"):
            assert ba["missing_bin"]["woe"] == pytest.approx(
                -bb["missing_bin"]["woe"], abs=1e-10)


# ---------- 性质 3：原样复制一份，分箱/WOE/系数不变 ------------------------

def test_duplicate_sample_invariant(dev_bytes):
    from tests.conftest import make_dev_csv
    dup = make_dev_csv(duplicated=True)
    ds1, ds2 = build_dataset(dev_bytes), build_dataset(dup)
    c1 = build_scorecard(ds1, name="c1", min_bin_pct=0.05)
    c2 = build_scorecard(ds2, name="c2", min_bin_pct=0.05)
    assert c1["selected_features"] == c2["selected_features"]
    for f in c1["selected_features"]:
        b1, b2 = c1["binning"][f], c2["binning"][f]
        assert len(b1["bins"]) == len(b2["bins"])
        for x, y in zip(b1["bins"], b2["bins"]):
            assert x["woe"] == pytest.approx(y["woe"], abs=1e-10)
            assert x["good"] * 2 == y["good"]
            assert x["bad"] * 2 == y["bad"]
    assert c1["coefficients"]["intercept"] == pytest.approx(
        c2["coefficients"]["intercept"], abs=1e-9)
    for f in c1["selected_features"]:
        assert c1["coefficients"][f] == pytest.approx(
            c2["coefficients"][f], abs=1e-9)


# ---------- 性质 4：各箱分值之和 == 总分 -----------------------------------

def test_feature_points_sum_to_total(card_bytes):
    applicant = {"income": 8200, "age": 51, "dti": 0.08,
                 "noise_num": 0.3, "city": "A", "housing": "own"}
    r = score_one(card_bytes["score_table"], applicant)
    assert sum(f["points"] for f in r["features"]) == pytest.approx(
        r["total_score"], abs=1e-9)


# ---------- 性质 5：总分 +PDO => odds 翻倍 ---------------------------------

def test_pdo_doubles_odds(card_bytes):
    st = card_bytes["score_table"]
    applicant = {"income": 6000, "age": 33, "dti": 0.3,
                 "noise_num": 0.1, "city": "B", "housing": "mortgage"}
    r = score_one(st, applicant)
    pdo = card_bytes["config"]["pdo"]
    # 分数 +PDO 直接映射：odds2/odds1 = exp(PDO/factor)=2
    odds2 = r["odds_good_to_bad"] * math.exp(pdo / st["factor"])
    assert odds2 / r["odds_good_to_bad"] == pytest.approx(2.0, abs=1e-9)
    pd0 = r["pd"]
    pd_plus = 1 / (1 + (1 / pd0 - 1) * 2)  # odds 翻倍后的 PD
    assert pd_plus < pd0


# ---------- 性质 6：PD 随总分严格单调下降且始终在 (0,1) --------------------

def test_pd_strictly_decreasing_and_in_range(card_bytes):
    st = card_bytes["score_table"]
    # 找一个箱数较多的数值特征，在每个箱内部各取一个值
    target = max(
        (f for f, s in st["features"].items()
         if card_bytes["binning"][f]["feature_type"] == "numeric"),
        key=lambda f: len(st["features"][f]["bins"]))
    spec = card_bytes["binning"][target]
    bins = [b for b in spec["bins"] if not b.get("missing")]
    assert len(bins) >= 3
    probe_values = []
    for i, b in enumerate(bins):
        lo, hi = b.get("lower"), b.get("upper")
        if lo is None:
            probe_values.append(hi - 1.0)
        elif hi is None:
            probe_values.append(lo + 1.0)
        else:
            probe_values.append((lo + hi) / 2.0)

    base = {"income": 7000, "age": 40, "dti": 0.25,
            "noise_num": 0.0, "city": "B", "housing": "mortgage"}
    rows = []
    for v in probe_values:
        a = dict(base)
        a[target] = v
        r = score_one(st, a)
        assert 0.0 < r["pd"] < 1.0
        rows.append((r["total_score"], r["pd"]))
    rows.sort(key=lambda t: t[0])
    # 相邻箱 WOE 可能相等（非严格单调）：同 WOE 箱分值相同属于正常，
    # 按总分去重后，PD 必须随总分严格下降
    distinct = []
    for score, pd in rows:
        if not distinct or abs(score - distinct[-1][0]) > 1e-9:
            distinct.append((score, pd))
    assert len(distinct) >= 2
    for (_, p1), (_, p2) in zip(distinct, distinct[1:]):
        assert p2 < p1 - 1e-15
    # 总分与 PD 之间 score = offset - factor*logit(PD)
    factor = st["factor"]
    offset = st["offset"]
    for score, pd in rows:
        assert score == pytest.approx(
            offset - factor * math.log(pd / (1 - pd)), abs=1e-9)


# ---------- 性质 7：数值特征 WOE 单调 --------------------------------------

def test_numeric_woe_monotonic(card_bytes):
    assert all(verify_monotonic(card_bytes).values())
    for f, spec in card_bytes["binning"].items():
        if spec["feature_type"] != "numeric":
            continue
        ws = [b["woe"] for b in spec["bins"]]
        assert spec["direction"] in ("increasing", "decreasing")
        if spec["direction"] == "increasing":
            assert all(b >= a - 1e-12 for a, b in zip(ws, ws[1:]))
        else:
            assert all(b <= a + 1e-12 for a, b in zip(ws, ws[1:]))


# ---------- 缺失箱独立、未见类别标注、每箱占比/好坏非零 --------------------

def test_bin_constraints_and_missing_separate(dev_bytes):
    ds = build_dataset(dev_bytes)
    card = build_scorecard(ds, name="m", min_bin_pct=0.05)
    for f, spec in card["binning"].items():
        for b in spec["bins"]:
            assert b["good"] > 0 and b["bad"] > 0
            assert b["total"] / len(ds.labels) >= 0.05 - 1e-9
    inc = card["binning"]["income"]
    assert inc["missing_bin"] is not None
    assert inc["missing_bin"]["missing"] is True
    assert sum(b["total"] for b in inc["bins"]) + \
        inc["missing_bin"]["total"] == len(ds.labels)


def test_unseen_category_is_flagged(card_bytes):
    r = score_one(card_bytes["score_table"], {
        "income": 7000, "age": 40, "dti": 0.2, "noise_num": 0.0,
        "city": "NEVER_SEEN", "housing": "own"})
    assert r["has_unseen"] is True
    assert r["unseen_features"] == ["city"]
    city = next(f for f in r["features"] if f["feature"] == "city")
    assert city["unseen"] is True
    assert city["bin_label"] == "UNSEEN_CATEGORY"


def test_numeric_out_of_range_falls_into_edge_bin(card_bytes):
    spec = card_bytes["binning"]["dti"]
    upper_bounds = [b["upper"] for b in spec["bins"] if b.get("upper") is not None]
    hi = max(upper_bounds)
    r = score_one(card_bytes["score_table"], {
        "income": 7000, "age": 40, "dti": hi + 10.0,
        "noise_num": 0.0, "city": "A", "housing": "own"})
    dti = next(f for f in r["features"] if f["feature"] == "dti")
    assert dti["bin_index"] == len(spec["bins"]) - 1
    assert dti["out_of_range"] is True
    assert 0.0 < r["pd"] < 1.0


# ---------- 牛顿回归自身性质 & 完全分离失败 --------------------------------

def test_newton_separation_fails():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 1))
    y = (x[:, 0] > 0).astype(int)
    with pytest.raises(Exception):
        fit_logistic_newton(x, y)


def test_newton_recovers_intercept_balance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 0.3).astype(int)
    fit = fit_logistic_newton(x, y)
    assert fit.converged
    assert abs(float(sigmoid(np.array([fit.intercept]))[0]) - 0.3) < 0.03


# ---------- KS/AUC 基本正确性 ---------------------------------------------

def test_auc_ks_perfect_and_random():
    y = [0] * 50 + [1] * 50
    perfect = list(range(50)) + list(range(50, 100))
    assert roc_auc(y, perfect) == pytest.approx(1.0)
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    rand = rng.random(1000)
    yy = (rng.random(1000) < .5).astype(int)
    assert 0.4 < roc_auc(list(yy), list(rand)) < 0.6
    assert 0.0 <= ks_statistic(list(yy), list(rand)) < 0.15


# ---------- 作业开始前拒绝 -------------------------------------------------

def test_reject_small_sample():
    from app.parsing import SampleValidationError, validate_job_request
    small = make_small_csv(300)
    ds = build_dataset(small)
    with pytest.raises(SampleValidationError):
        validate_job_request(ds, pdh=50, features=None,
                             min_bin_pct=.05, iv_threshold=.02)


def make_small_csv(n):
    import io, csv
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["target", "x"])
    for i in range(n):
        w.writerow([i % 2, i])
    return buf.getvalue().encode()


def test_reject_bad_label_and_pdo_and_feature():
    from app.parsing import (SampleValidationError, build_dataset as bd,
                             validate_job_request as vjr)
    # 标签出现 0/1 以外的值：解析阶段拒绝
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["target", "x"])
    for i in range(600):
        w.writerow([0 if i < 599 else 2, i])
    with pytest.raises(SampleValidationError):
        bd(buf.getvalue().encode())

    # 标签全为同一类
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["target", "x"])
    for i in range(600):
        w.writerow([0, i])
    with pytest.raises(SampleValidationError):
        bd(buf.getvalue().encode())

    # 小样本 + PDO 非正 + 入模特征不存在
    ds = bd(make_small_csv(500))
    # 500 行是边界，应通过样本量校验；先验证通过
    vjr(ds, pdh=50, features=None, min_bin_pct=.05, iv_threshold=.02)
    with pytest.raises(SampleValidationError):
        vjr(ds, pdh=0, features=None, min_bin_pct=.05, iv_threshold=.02)
    with pytest.raises(SampleValidationError):
        vjr(ds, pdh=50, features=["nope"],
            min_bin_pct=.05, iv_threshold=.02)
