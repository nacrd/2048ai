"""Run with: python -m uvicorn server.app:app --host 127.0.0.1 --port 8765."""
import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Literal
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
gpu = None
gpu_error = 'CUDA 尚未初始化'
compute_lock = asyncio.Lock()


def init_gpu():
    global gpu, gpu_error
    try:
        from .gpu import CudaEngine
        gpu = CudaEngine()
        gpu_error = ''
    except Exception as e:
        gpu = None
        gpu_error = str(e)


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(init_gpu)
    yield


app = FastAPI(title='2048 AI local service', lifespan=lifespan)


class SolveRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    board: list[list[int]]
    algorithm: Literal['rollout'] = 'rollout'
    backend: Literal['cuda', 'cpu'] = 'cuda'
    objective: Literal['score', 'target'] = 'score'
    target: int = Field(default=2048, ge=2, le=2 ** 30)
    budgetMs: int = Field(default=200, ge=1, le=10000)
    horizon: int = Field(default=32, ge=1, le=2 ** 53 - 1)
    trajectories: int = Field(default=128, ge=1, le=65536)
    seed: int = Field(default=12345, ge=1, le=4294967295)
    requestId: str = Field(default='', max_length=100)

    @model_validator(mode='after')
    def check_board(self):
        n = len(self.board)
        if not 2 <= n <= 6 or any(len(row) != n for row in self.board):
            raise ValueError('棋盘必须为 2×2 到 6×6 的正方形矩阵')
        for row in self.board:
            for v in row:
                if v < 0 or v > 2 ** 52 or (v and (v < 2 or v & (v - 1))):
                    raise ValueError('方块必须为 0 或安全整数范围内的 2 的幂')
        if self.target & (self.target - 1):
            raise ValueError('目标必须为 2 的幂')
        return self


@app.get('/api/health')
async def health():
    threshold = None
    if gpu:
        path = ROOT / 'benchmarks' / 'calibration.json'
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('device') == gpu.name:
                threshold = data.get('autoThreshold')
    return {'available': gpu is not None, 'name': gpu.name if gpu else None, 'reason': gpu_error, 'memoryBytes': gpu.memory if gpu else None, 'autoThreshold': threshold}


def confidence(successes, count):
    z = 1.959963984540054
    p = successes / count
    a = 1 + z * z / count
    mean = (p + z * z / (2 * count)) / a
    delta = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / a
    return [max(0, mean - delta), min(1, mean + delta)]


@app.post('/api/solve')
async def solve(payload: SolveRequest, request: Request):
    start = perf_counter()
    from .cpu import direction_value, move, rollout_batch
    n = len(payload.board)
    board = np.array([int(math.log2(v)) if v else 0 for row in payload.board for v in row], dtype=np.uint8)
    directions = np.array([d for d in range(4) if move(board, n, d)[2]], dtype=np.int32)
    if payload.backend == 'cuda' and gpu is None:
        raise HTTPException(503, detail=gpu_error or 'CUDA 不可用')
    sums = np.zeros(len(directions), dtype=np.float64)
    count, kernel_ms = 0, 0.0
    async with compute_lock:
        if await request.is_disconnected():
            raise HTTPException(499, '任务已取消')
        while count < payload.trajectories and len(directions):
            # Allow one batch, then enforce elapsed budget including queue delay.
            if count and (perf_counter() - start) * 1000 >= payload.budgetMs:
                break
            if await request.is_disconnected():
                raise HTTPException(499, '任务已取消')
            batch = min(2048 if payload.backend == 'cuda' else 256, payload.trajectories - count)
            args = (board, n, directions, count, batch, payload.seed, payload.horizon, int(payload.objective == 'target'), int(math.log2(payload.target)))
            if payload.backend == 'cuda':
                values = await asyncio.to_thread(gpu.batch, *args)
                kernel_ms += gpu.last_kernel_ms
            else:
                values = await asyncio.to_thread(rollout_batch, *args)
            sums += values.sum(axis=1)
            count += batch
    choices = [{'direction': int(d), 'value': float(sums[i] / count), 'tieBreak': float(direction_value(board, n, int(d))), 'samples': count, **({'confidence': confidence(sums[i], count)} if payload.objective == 'target' else {})} for i, d in enumerate(directions)] if count else []
    best = max(choices, key=lambda c: (c['value'], c['tieBreak'], -c['direction']))['direction'] if choices else None
    note = '固定贪心后续策略的模拟估计；不是最优策略胜率' if len(directions) else '没有合法移动'
    if payload.objective == 'target' and choices and all(c['value'] == 0 for c in choices):
        note += '；无成功样本，按预期局面评分选择，概率仍为 0'
    return {'direction': best, 'choices': choices, 'algorithm': 'rollout', 'backend': payload.backend, 'complete': not len(directions) or count == payload.trajectories, 'depth': payload.horizon, 'nodes': count * len(directions) * payload.horizon, 'elapsedMs': (perf_counter() - start) * 1000, 'kernelMs': kernel_ms, 'requestId': payload.requestId, 'note': note}


if (ROOT / 'dist').exists():
    app.mount('/assets', StaticFiles(directory=ROOT / 'dist' / 'assets'), name='assets')

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'dist' / 'index.html')

    @app.get('/favicon.ico')
    async def favicon():
        return FileResponse(ROOT / 'favicon.ico')
