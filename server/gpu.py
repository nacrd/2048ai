from pathlib import Path
import numpy as np
from .policy import cuda_paths


class CudaEngine:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        properties = cp.cuda.runtime.getDeviceProperties(0)
        self.name = properties['name'].decode()
        self.memory = properties['totalGlobalMem']
        self.capability = (properties['major'], properties['minor'])
        self.multiprocessors = properties['multiProcessorCount']
        self.block_size = 128
        code = cuda_paths() + Path(__file__).with_name('cuda').joinpath('rollout.cu').read_text(encoding='utf-8')
        self.module = cp.RawModule(code=code, options=('--std=c++17',))
        # Compile all size-specialized paths without launching a gameplay workload.
        self.module.compile()
        self.kernels = {n: self.module.get_function(f'rollout{n}') for n in range(2, 7)}
        self.move_kernel = self.module.get_function('moves')
        self._input_key = None
        self._outputs = {}
        self._sums = {}
        self._begin, self._end = cp.cuda.Event(), cp.cuda.Event()

    def _launch(self, board, n, directions, offset, count, seed, horizon, objective, target):
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
        self.kernels[n](((result.size + self.block_size - 1) // self.block_size,), (self.block_size,), (b, np.int32(n), dirs, np.int32(len(directions)), np.int32(offset), np.int32(count), np.uint32(seed), np.uint64(horizon), np.int32(objective), np.int32(target), result))
        end.record()
        return result

    def _read(self, result):
        # A blocking copy on the same stream already waits for the recorded event.
        host = self.cp.asnumpy(result)
        self.last_kernel_ms = self.cp.cuda.get_elapsed_time(self._begin, self._end)
        return host

    def batch(self, *args):
        return self._read(self._launch(*args))

    def batch_sums(self, board, n, directions, offset, count, seed, horizon, objective, target):
        mass = sum(2 ** int(e) if e else 0 for e in board)
        # Integer rewards aggregate exactly in any order only within the safe range.
        # Preserve the original host sum for exceptional high-mass/long-horizon inputs.
        safe = objective or (offset + count) * horizon * (mass + 4 * horizon) <= 2 ** 53
        if not safe:
            return self.batch(board, n, directions, offset, count, seed, horizon, objective, target).sum(axis=1)
        values = self._launch(board, n, directions, offset, count, seed, horizon, objective, target)
        sums = self._sums.get(len(directions))
        if sums is None:
            sums = self.cp.empty(len(directions), dtype=self.cp.float64)
            self._sums[len(directions)] = sums
        self.cp.sum(values, axis=1, out=sums)
        return self._read(sums)

    def moves(self, boards, n):
        cp = self.cp
        count = len(boards)
        inputs = cp.asarray(boards, dtype=cp.uint8)
        results = cp.empty((count, 4, n * n), dtype=cp.uint8)
        scores = cp.empty((count, 4), dtype=cp.float64)
        changed = cp.empty((count, 4), dtype=cp.int32)
        self.move_kernel(((count * 4 + 127) // 128,), (128,), (inputs, np.int32(n), np.int32(count), results, scores, changed))
        return cp.asnumpy(results), cp.asnumpy(scores), cp.asnumpy(changed)
