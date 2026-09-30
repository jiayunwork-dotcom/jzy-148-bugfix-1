"""样本解析模块。

只使用标准库 csv 读取开发样本：一列 0/1 违约标签（1=坏），其余为特征列，
数值型 / 类别型自动识别，允许缺失（空串、NA、NULL 等记为缺失）。
建卡作业开始前的硬性校验（样本量、标签、PDO、入模特征）也在这里完成。
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Any

MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "none", "null"})
DEFAULT_TARGET_NAMES = ("target", "label", "y", "bad", "default", "is_bad")

MIN_SAMPLE_ROWS = 500


class SampleValidationError(ValueError):
    """作业开始前的样本/参数校验失败。"""


def is_missing(token: str) -> bool:
    return token.strip().lower() in MISSING_TOKENS


def _parse_float(token: str) -> float | None:
    try:
        value = float(token.strip())
    except (ValueError, TypeError):
        return None
    if math.isnan(value):
        return None
    return value


@dataclass
class Dataset:
    target: str
    feature_names: list[str]
    feature_types: dict[str, str]          # "numeric" / "categorical"
    labels: list[int]
    columns: dict[str, list[Any]]          # None 表示缺失
    raw_rows: list[dict[str, str]] = field(default_factory=list)
    total: int = 0
    bad_rate: float = 0.0


def read_csv_bytes(data: bytes) -> list[dict[str, str]]:
    """解析 CSV 字节为原始字符串行。"""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SampleValidationError(f"CSV 不是 UTF-8 编码: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SampleValidationError("CSV 缺少表头")
    header = [name.strip() for name in reader.fieldnames]
    if len(header) < 2:
        raise SampleValidationError("至少需要标签列和一个特征列")
    if len(set(header)) != len(header):
        raise SampleValidationError("CSV 表头存在重复列名")
    if any(name == "" for name in header):
        raise SampleValidationError("CSV 存在空列名")

    rows: list[dict[str, str]] = []
    for line_no, raw in enumerate(reader, start=2):
        if raw is None:
            continue
        if None in raw:
            raise SampleValidationError(f"第 {line_no} 行列数多于表头")
        rows.append({name: (raw.get(orig) or "") for name, orig in
                     zip(header, reader.fieldnames or [])})
    return rows


def _pick_target(rows: list[dict[str, str]], target_col: str | None) -> str:
    header = list(rows[0].keys())
    if target_col:
        if target_col not in header:
            raise SampleValidationError(f"指定的标签列不存在: {target_col}")
        return target_col
    for name in header:
        if name.lower() in DEFAULT_TARGET_NAMES and all(
                row[name].strip() in ("0", "1") for row in rows):
            return name
    # 退而求其次：第一列取值全部为 0/1 且无缺失
    first = header[0]
    if all(row[first].strip() in ("0", "1") for row in rows):
        return first
    raise SampleValidationError(
        "未找到 0/1 标签列（可用 target_col 指定列名）")


def build_dataset(
    data: bytes,
    target_col: str | None = None,
) -> Dataset:
    """读取 CSV 并完成标签与类型识别。"""
    rows = read_csv_bytes(data)
    if not rows:
        raise SampleValidationError("CSV 没有数据行")
    target = _pick_target(rows, target_col)
    header = list(rows[0].keys())
    feature_names = [c for c in header if c != target]

    # 标签校验
    labels: list[int] = []
    for line_no, row in enumerate(rows, start=2):
        token = row[target].strip()
        if token not in ("0", "1"):
            raise SampleValidationError(
                f"标签列 {target} 第 {line_no} 行不是 0/1（缺失也不允许）")
        labels.append(int(token))
    if len(set(labels)) < 2:
        raise SampleValidationError("标签全部为同一类，无法建卡")

    # 类型推断：一列非缺失值全部可解析为 float 即视为数值型
    columns: dict[str, list[Any]] = {}
    feature_types: dict[str, str] = {}
    for name in feature_names:
        numeric = True
        col: list[Any] = []
        saw_value = False
        for row in rows:
            token = row[name]
            if is_missing(token):
                col.append(None)
                continue
            value = _parse_float(token)
            if value is None:
                numeric = False
                break
            saw_value = True
            col.append(value)
        if not numeric:
            col = [None if is_missing(row[name]) else row[name].strip()
                   for row in rows]
            saw_value = any(v is not None for v in col)
        if not saw_value:
            feature_types[name] = "numeric"  # 整列缺失，后续单箱过滤
        else:
            feature_types[name] = "numeric" if numeric else "categorical"
        columns[name] = col

    total = len(labels)
    return Dataset(
        target=target,
        feature_names=feature_names,
        feature_types=feature_types,
        labels=labels,
        columns=columns,
        raw_rows=rows,
        total=total,
        bad_rate=sum(labels) / total,
    )


def validate_job_request(
    dataset: Dataset,
    *,
    pdh: float,
    features: list[str] | None,
    min_bin_pct: float,
    iv_threshold: float,
) -> None:
    """作业开始前的硬性参数校验。"""
    if dataset.total < MIN_SAMPLE_ROWS:
        raise SampleValidationError(
            f"开发样本仅 {dataset.total} 行，少于 {MIN_SAMPLE_ROWS} 行，拒绝建卡")
    if not isinstance(pdh, (int, float)) or not math.isfinite(pdh) or pdh <= 0:
        raise SampleValidationError("PDO 必须为正数")
    if not 0.0 < min_bin_pct < 1.0:
        raise SampleValidationError("最小箱样本占比需在 (0,1) 之间")
    if not 0.0 <= iv_threshold:
        raise SampleValidationError("IV 筛选阈值不能为负")
    if features is not None:
        missing = [f for f in features if f not in dataset.feature_names]
        if missing:
            raise SampleValidationError(f"入模特征不存在: {missing}")
        if len(features) == 0:
            raise SampleValidationError("入模特征清单为空")
        if len(set(features)) != len(features):
            raise SampleValidationError("入模特征清单存在重复")
