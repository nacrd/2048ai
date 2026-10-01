"""Numba-compiled reference rollout; same policy and PRNG as the browser/CUDA."""
import numpy as np
from numba import njit, prange
from .policy import SNAKE_PATHS


@njit(cache=True, inline='always')
def index_for(n, direction, line, i):
    return i * n + line if direction == 0 else line * n + n - 1 - i if direction == 1 else (n - 1 - i) * n + line if direction == 2 else line * n + i


@njit(cache=True)
def move_into(b, n, direction, out, values):
    for i in range(n * n):
        out[i] = 0
    score = 0.0
    for line in range(n):
        count = 0
        for i in range(n):
            index = index_for(n, direction, line, i)
            if b[index]:
                values[count] = b[index]
                count += 1
        k, slot = 0, 0
        while k < count:
            v = int(values[k])
            if k + 1 < count and values[k + 1] == v:
                v += 1
                score += 2.0 ** v
                k += 1
            out[index_for(n, direction, line, slot)] = v
            k += 1
            slot += 1
    moved = False
    for i in range(n * n):
        if out[i] != b[i]:
            moved = True
            break
    return score, moved


@njit(cache=True, nogil=True)
def move(b, n, direction):
    out = np.empty(n * n, dtype=np.uint8)
    values = np.empty(n, dtype=np.uint8)
    score, moved = move_into(b, n, direction, out, values)
    return out, score, moved


@njit(cache=True)
def evaluate(b, n):
    empty, smooth, merges, monotonic, maximum = 0, 0, 0, 0, 0
    for r in range(n):
        for c in range(n):
            v = np.int64(b[r * n + c])
            if not v:
                empty += 1
            maximum = max(maximum, v)
            if c + 1 < n:
                w = np.int64(b[r * n + c + 1])
                if v and w:
                    smooth += abs(v - w)
                    if v == w:
                        merges += v
            if r + 1 < n:
                w = np.int64(b[(r + 1) * n + c])
                if v and w:
                    smooth += abs(v - w)
                    if v == w:
                        merges += v
    for axis in range(2):
        for line in range(n):
            up, down = 0, 0
            for i in range(n - 1):
                a = np.int64(b[i * n + line] if axis else b[line * n + i]) ** 2
                v = np.int64(b[(i + 1) * n + line] if axis else b[line * n + i + 1]) ** 2
                up += max(0, v - a)
                down += max(0, a - v)
            monotonic += min(up, down)
    corner = max(b[0], b[n - 1], b[n * (n - 1)], b[n * n - 1]) == maximum
    if not empty and not merges:
        return -1000000000.0
    snake = -1e100
    for orientation in range(8):
        value, previous = 0, 0
        for rank in range(n * n):
            v = np.int64(b[SNAKE_PATHS[n - 2, orientation, rank]])
            value += v * v * (n * n - rank) * 4
            if rank:
                value -= max(0, v - previous) ** 2 * 35
            previous = v
        snake = max(snake, value)
    return empty * (280 + maximum * 20) - smooth * 8 + merges * 35 - monotonic * 12 + maximum * maximum * 20 + (maximum * maximum * 35 if corner else 0) + snake


@njit(cache=True)
def direction_value(b, n, direction):
    out, score, moved = move(b, n, direction)
    empty = np.flatnonzero(out == 0)
    value = 0.0
    for index in empty:
        for exponent in range(1, 3):
            out[index] = exponent
            value += evaluate(out, n) * (0.9 if exponent == 1 else 0.1) / len(empty)
        out[index] = 0
    return (value if len(empty) else evaluate(out, n)) + np.log2(score + 1) * 12


@njit(cache=True)
def next_uint(x):
    x = np.uint32(x)
    x = np.uint32(x ^ np.uint32(x << 13))
    x = np.uint32(x ^ (x >> 17))
    return np.uint32(x ^ np.uint32(x << 5))


@njit(cache=True)
def spawn(b, n, rng):
    count = 0
    for i in range(n * n):
        if not b[i]:
            count += 1
    if count:
        rng = next_uint(rng)
        v = 1 if float(rng) / 4294967296.0 < 0.9 else 2
        rng = next_uint(rng)
        selected = int(float(rng) / 4294967296.0 * count)
        # Preserve upstream's column-first order and both RNG draws.
        for x in range(n):
            for y in range(n):
                i = y * n + x
                if not b[i]:
                    if selected == 0:
                        b[i] = v
                        return rng
                    selected -= 1
    return rng


@njit(cache=True)
def greedy_into(b, n, out, values):
    best, value = -1, -1e100
    for d in range(4):
        score, moved = move_into(b, n, d, out, values)
        if moved:
            q = evaluate(out, n) + np.log2(score + 1) * 12
            if q > value:
                best, value = d, q
    return best


@njit(cache=True)
def greedy(b, n):
    return greedy_into(b, n, np.empty(n * n, dtype=np.uint8), np.empty(n, dtype=np.uint8))


@njit(cache=True)
def trajectory_seed(seed, direction, ident):
    x = np.uint32(seed) ^ np.uint32((direction + 1) * 0x9E3779B9) ^ np.uint32((ident + 1) * 0x85EBCA6B)
    x = np.uint32(x ^ (x >> 16))
    x = np.uint32(x * np.uint32(0x7FEB352D))
    x = np.uint32(x ^ (x >> 15))
    return x if x else np.uint32(0x6D2B79F5)


@njit(cache=True)
def rollout_one(board, n, direction, ident, seed, horizon, objective, target):
    b = board.copy()
    out = np.empty(n * n, dtype=np.uint8)
    trial = np.empty(n * n, dtype=np.uint8)
    values = np.empty(n, dtype=np.uint8)
    rng = trajectory_seed(seed, direction, ident)
    score, action = 0.0, direction
    for step in range(horizon):
        if objective and np.max(b) >= target:
            return 1.0
        if action < 0:
            break
        delta, moved = move_into(b, n, action, out, values)
        if not moved:
            break
        score += delta
        b, out = out, b
        rng = spawn(b, n, rng)
        if objective and np.max(b) >= target:
            return 1.0
        if step + 1 < horizon:
            action = greedy_into(b, n, trial, values)
    return (1.0 if np.max(b) >= target else 0.0) if objective else score


@njit(cache=True, parallel=True, nogil=True)
def rollout_batch(board, n, directions, offset, count, seed, horizon, objective, target):
    results = np.empty((len(directions), count), dtype=np.float64)
    for j in prange(len(directions) * count):
        d, i = j // count, j % count
        results[d, i] = rollout_one(board, n, int(directions[d]), offset + i, seed, horizon, objective, target)
    return results
