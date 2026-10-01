"""Warm identical rollout benchmarks: optimized parallel CPU, CUDA, and HTTP."""
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import httpx
from numba import get_num_threads
from server.cpu import rollout_batch
from server.gpu import CudaEngine

board = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 0, 0, 0], dtype=np.uint8)
directions = np.arange(4, dtype=np.int32)
seed, horizon = 71231, 32
matrix = [[int(2 ** int(v)) if v else 0 for v in board[r*4:(r+1)*4]] for r in range(4)]
cold = perf_counter(); gpu = CudaEngine(); gpu_init_ms = (perf_counter()-cold)*1000
start = perf_counter(); rollout_batch(board, 4, directions, 0, 8, seed, horizon, 0, 11); cpu_init_ms = (perf_counter()-start)*1000
results = []
with httpx.Client(base_url='http://127.0.0.1:8765', timeout=60, trust_env=False) as client:
    # API warmup is measured separately from steady state.
    payload = {'board': matrix, 'seed': seed, 'horizon': horizon, 'trajectories': 8, 'budgetMs': 10000}
    cold_api = {}
    for backend in ['cpu', 'cuda']:
        t = perf_counter(); response = client.post('/api/solve', json={**payload, 'backend': backend}); response.raise_for_status(); cold_api[backend] = (perf_counter()-t)*1000
    for count in [32, 128, 512, 2048, 8192]:
        timings = {key: [] for key in ['cpuMs', 'cudaMs', 'kernelMs', 'cpuHttpMs', 'cudaHttpMs']}
        for repeat in range(5):
            start = perf_counter(); cpu = rollout_batch(board, 4, directions, 0, count, seed, horizon, 0, 11); timings['cpuMs'].append((perf_counter()-start)*1000)
            start = perf_counter(); accelerated = gpu.batch(board, 4, directions, 0, count, seed, horizon, 0, 11); timings['cudaMs'].append((perf_counter()-start)*1000); timings['kernelMs'].append(gpu.last_kernel_ms)
            np.testing.assert_array_equal(cpu, accelerated)
            answers = {}
            for backend in ['cpu', 'cuda']:
                start = perf_counter(); response = client.post('/api/solve', json={**payload, 'trajectories': count, 'backend': backend}); response.raise_for_status(); answers[backend] = response.json(); timings[backend+'HttpMs'].append((perf_counter()-start)*1000)
                assert answers[backend]['complete'], 'Increase budget before comparing equal work'
            assert answers['cpu']['choices'] == answers['cuda']['choices']
        row = {'trajectoriesPerDirection': count, 'horizon': horizon, 'size': 4, **{key: {'p50': float(np.median(values)), 'p95': float(np.percentile(values, 95))} for key, values in timings.items()}}
        row['httpSpeedup'] = row['cpuHttpMs']['p50']/row['cudaHttpMs']['p50']; results.append(row); print(json.dumps(row), flush=True)
report = {'date': '2026-10-01', 'device': gpu.name, 'cpu': platform.processor(), 'numbaThreads': get_num_threads(), 'gpuInitMs': gpu_init_ms, 'cpuInitMs': cpu_init_ms, 'apiWarmupMs': cold_api, 'seed': seed, 'repeats': 5, 'board': matrix, 'objective': 'score', 'results': results}
Path('benchmarks/gpu-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
# Auto selection stays conservative: only this tested size and at least this horizon/count.
candidates = [r for r in results if r['httpSpeedup'] >= 1.2 and r['cudaHttpMs']['p95'] < r['cpuHttpMs']['p50']]
threshold = {'size': 4, 'minHorizon': 32, 'minTrajectories': max(512, min(r['trajectoriesPerDirection'] for r in candidates))} if candidates else None
Path('benchmarks/calibration.json').write_text(json.dumps({'device': gpu.name, 'autoThreshold': threshold, 'source': 'gpu-results.json'}, indent=2), encoding='utf-8')
