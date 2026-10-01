from pathlib import Path
import numpy as np


class CudaEngine:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        properties = cp.cuda.runtime.getDeviceProperties(0)
        self.name = properties['name'].decode()
        self.memory = properties['totalGlobalMem']
        self.capability = (properties['major'], properties['minor'])
        code = Path(__file__).with_name('cuda').joinpath('rollout.cu').read_text(encoding='utf-8')
        self.kernel = cp.RawKernel(code, 'rollout', options=('--std=c++17',))
        self.move_kernel = cp.RawKernel(code, 'moves', options=('--std=c++17',))
        self._input_key = None
        self._outputs = {}
        self._begin, self._end = cp.cuda.Event(), cp.cuda.Event()
        # Compile and launch an actual simulation, not just a device query.
        self.batch(np.array([1, 1, 0, 0], dtype=np.uint8), 2, np.array([3], dtype=np.int32), 0, 1, 123, 2, 0, 11)

    def batch(self, board, n, directions, offset, count, seed, horizon, objective, target):
        cp = self.cp
        key = (n, board.tobytes(), directions.tobytes())
        if key != self._input_key:
            self._board, self._directions = cp.asarray(board, dtype=cp.uint8), cp.asarray(directions, dtype=cp.int32)
            self._input_key = key
        b, dirs = self._board, self._directions
        shape = (len(directions), count)
        result = self._outputs.get(shape)
        if result is None:
            # Keep a bounded set of batch shapes; retain only buffers for this new shape when full.
            if len(self._outputs) >= 8:
                self._outputs.clear()
            result = cp.empty(shape, dtype=cp.float64)
            self._outputs[shape] = result
        begin, end = self._begin, self._end
        begin.record()
        self.kernel(((result.size + 127) // 128,), (128,), (b, np.int32(n), dirs, np.int32(len(directions)), np.int32(offset), np.int32(count), np.uint32(seed), np.uint64(horizon), np.int32(objective), np.int32(target), result))
        end.record(); end.synchronize()
        self.last_kernel_ms = cp.cuda.get_elapsed_time(begin, end)
        return cp.asnumpy(result)

    def moves(self, boards, n):
        cp = self.cp
        count = len(boards)
        inputs = cp.asarray(boards, dtype=cp.uint8)
        results = cp.empty((count, 4, n * n), dtype=cp.uint8)
        scores = cp.empty((count, 4), dtype=cp.float64)
        changed = cp.empty((count, 4), dtype=cp.int32)
        self.move_kernel(((count * 4 + 127) // 128,), (128,), (inputs, np.int32(n), np.int32(count), results, scores, changed))
        return cp.asnumpy(results), cp.asnumpy(scores), cp.asnumpy(changed)
