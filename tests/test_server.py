import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from server import cpu
from server.app import app


@pytest.fixture(scope='module')
def engine():
    try:
        from server.gpu import CudaEngine
        return CudaEngine()
    except Exception as e:
        pytest.skip(f'CUDA unavailable: {e}')


def test_shared_cpu_gpu_browser_fixtures(engine):
    fixtures = json.loads((Path(__file__).parent / 'fixtures' / 'cross.json').read_text())
    directions = np.arange(4, dtype=np.int32)
    for fixture in fixtures:
        b = np.array(fixture['board']['cells'], dtype=np.uint8)
        n = fixture['board']['size']
        gpu_boards, gpu_scores, gpu_moved = engine.moves(b[None, :], n)
        for d in range(4):
            expected = fixture['moves'][d]
            board, score, moved = cpu.move(b, n, d)
            assert board.tolist() == expected['cells'] == gpu_boards[0, d].tolist()
            assert score == expected['score'] == gpu_scores[0, d]
            assert moved == expected['moved'] == bool(gpu_moved[0, d])
        reference = np.array(fixture['rollouts'])
        compiled = cpu.rollout_batch(b, n, directions, 0, 8, 71231, 16, 0, 11)
        accelerated = engine.batch(b, n, directions, 0, 8, 71231, 16, 0, 11)
        np.testing.assert_array_equal(compiled, reference)
        np.testing.assert_array_equal(accelerated, reference)
        np.testing.assert_array_equal(cpu.rollout_batch(b, n, directions, 0, 8, 71231, 16, 1, 8), engine.batch(b, n, directions, 0, 8, 71231, 16, 1, 8))


def test_gpu_move_random_all_sizes(engine):
    rng = np.random.default_rng(3718)
    for n in range(2, 7):
        boards = rng.integers(0, 31, size=(128, n*n), dtype=np.uint8)
        results, scores, changes = engine.moves(boards, n)
        for i, b in enumerate(boards):
            for d in range(4):
                result, score, changed = cpu.move(b, n, d)
                np.testing.assert_array_equal(results[i, d], result)
                assert scores[i, d] == score
                assert changes[i, d] == changed


def test_api_validation_and_cpu_gpu_results(engine):
    with TestClient(app) as client:
        base = {'board': [[2, 2], [0, 4]], 'trajectories': 8, 'horizon': 8, 'budgetMs': 10000, 'seed': 4712, 'objective': 'target', 'target': 16}
        cpu_response = client.post('/api/solve', json={**base, 'backend': 'cpu'})
        gpu_response = client.post('/api/solve', json={**base, 'backend': 'cuda'})
        assert cpu_response.status_code == gpu_response.status_code == 200
        assert cpu_response.json()['choices'] == gpu_response.json()['choices']
        assert client.get('/api/health').json()['available']
        for bad in [{'board': [[3, 2], [0, 0]]}, {'board': [[2], [0, 0]]}, {'target': 3}, {'trajectories': 65537}, {'horizon': 0}, {'board': [[True, 2], [0, 0]]}]:
            assert client.post('/api/solve', json={**base, **bad}).status_code == 422
        dead = client.post('/api/solve', json={**base, 'board': [[2, 4], [8, 16]]}).json()
        assert dead['direction'] is None and dead['complete']
