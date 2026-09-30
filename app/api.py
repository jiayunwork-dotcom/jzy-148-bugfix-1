"""HTTP 接口层（FastAPI，仅 HTTP，无前端）。

建卡：
  POST /jobs                        上传 CSV 提交后台建卡作业
  GET  /jobs                        作业列表
  GET  /jobs/{job_id}               作业状态/失败原因
卡与版本：
  GET  /cards                       卡列表
  GET  /cards/{name}/versions       版本列表（含训练指标）
  GET  /cards/{name}                取卡（?version=N，缺省最新）
打分：
  POST /score/{name}                单条打分（?version=N）
  POST /score/{name}/batch          批量打分，单条错误只影响该条
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .parsing import SampleValidationError
from .scheduler import BuildParams, JobScheduler
from .scoring import score_one
from .storage import Repository, to_jsonable
from .storage import get_repository


class ScoreRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., description="原始特征值，键为特征名，缺失传 null")


class BatchScoreRequest(BaseModel):
    applicants: list[dict[str, Any]]


def create_app(repo: Repository | None = None,
               scheduler: JobScheduler | None = None) -> FastAPI:
    app = FastAPI(title="内部评分卡服务", version="1.0.0")
    # 默认后端延迟到首次使用时初始化，避免数据库短暂不可达时启动即失败
    app.state._repo = repo
    app.state._scheduler = scheduler

    def _sched(app: FastAPI) -> JobScheduler:
        if app.state._scheduler is None:
            app.state._scheduler = JobScheduler(_repo(app))
        return app.state._scheduler

    def _repo(app: FastAPI) -> Repository:
        if app.state._repo is None:
            app.state._repo = get_repository()
        return app.state._repo

    @app.exception_handler(SampleValidationError)
    async def _sample_error(_request, exc: SampleValidationError):
        return JSONResponse(status_code=400,
                            content={"error": str(exc), "rejected": True})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---------------- 建卡作业 ----------------

    @app.post("/jobs", status_code=202)
    async def create_job(
        card_name: str = Form(...),
        file: UploadFile = File(..., description="CSV 开发样本"),
        target_col: str | None = Form(None),
        features: str | None = Form(
            None, description="逗号分隔的入模特征清单；缺省按 IV 自动筛选"),
        iv_threshold: float = Form(0.02),
        min_bin_pct: float = Form(0.05),
        base_score: float = Form(600.0),
        base_odds: float = Form(50.0),
        pdo: float = Form(50.0),
    ):
        raw = await file.read()
        feature_list = (
            [f.strip() for f in features.split(",") if f.strip()]
            if features else None)
        params = BuildParams(
            name=card_name, csv_data=raw, target_col=target_col,
            features=feature_list, iv_threshold=iv_threshold,
            min_bin_pct=min_bin_pct, base_score=base_score,
            base_odds=base_odds, pdo=pdo)
        sched = _sched(app)
        try:
            job_id = sched.submit(params)
        except SampleValidationError as exc:
            raise HTTPException(status_code=400, detail={
                "error": str(exc), "rejected": True})
        return {"job_id": job_id, "status": "queued", "card_name": card_name}

    @app.get("/jobs")
    async def list_jobs(card_name: str | None = None):
        return to_jsonable(_repo(app).list_jobs(card_name))

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            return to_jsonable(_repo(app).get_job(job_id))
        except KeyError:
            raise HTTPException(404, "作业不存在")

    # ---------------- 卡与版本 ----------------

    @app.get("/cards")
    async def list_cards():
        return to_jsonable(_repo(app).list_cards())

    @app.get("/cards/{name}/versions")
    async def list_versions(name: str):
        try:
            return to_jsonable(_repo(app).list_versions(name))
        except KeyError:
            raise HTTPException(404, f"卡 {name} 不存在")

    @app.get("/cards/{name}")
    async def get_card(name: str, version: int | None = Query(None)):
        try:
            card = _repo(app).get_card(name, version)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        return to_jsonable(card)

    # ---------------- 打分 ----------------

    @app.post("/score/{name}")
    async def score(name: str, body: ScoreRequest,
                    version: int | None = Query(None)):
        try:
            card = _repo(app).get_card(name, version)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        try:
            result = score_one(card["score_table"], body.features)
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                422, {"error": f"该申请人无法打分: {exc}",
                      "applicant": body.features})
        result["card_name"] = name
        if version is None:
            versions = _repo(app).list_versions(name)
            result["card_version"] = versions[-1]["version"] if versions else None
        else:
            result["card_version"] = version
        return to_jsonable(result)

    @app.post("/score/{name}/batch")
    async def score_batch(name: str, body: BatchScoreRequest,
                          version: int | None = Query(None)):
        repo = _repo(app)
        try:
            card = repo.get_card(name, version)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        resolved = version
        if resolved is None:
            try:
                resolved = repo.list_versions(name)[-1]["version"]
            except IndexError:
                raise HTTPException(404, f"卡 {name} 不存在")

        # 在线打分是纯 CPU 轻量计算，放到线程池避免阻塞事件循环；
        # 每条独立 try，单条出错只影响自身。
        def _do_batch():
            results = []
            for idx, applicant in enumerate(body.applicants):
                try:
                    r = score_one(card["score_table"], applicant)
                    r["index"] = idx
                    results.append({"ok": True, "result": r})
                except Exception as exc:  # noqa: BLE001
                    results.append({
                        "ok": False,
                        "index": idx,
                        "error": f"{type(exc).__name__}: {exc}",
                        "applicant": applicant,
                    })
            return results

        items = await asyncio.get_running_loop().run_in_executor(
            None, _do_batch)
        n_ok = sum(1 for it in items if it["ok"])
        return to_jsonable({
            "card_name": name,
            "card_version": resolved,
            "total": len(items),
            "succeeded": n_ok,
            "failed": len(items) - n_ok,
            "results": items,
        })

    return app


app = create_app()
