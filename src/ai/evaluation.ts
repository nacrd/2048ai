import { applyMove, emptyCells, type Board, type Direction } from '../core/engine';

const paths = new Map<number, number[][]>();
function snakePaths(n: number) {
  let result = paths.get(n);
  if (!result) {
    result = Array.from({ length: 8 }, (_, orientation) => Array.from({ length: n * n }, (_, rank) => {
      let r = Math.floor(rank / n), c = rank % n; if (r % 2) c = n - 1 - c;
      if (orientation & 1) [r, c] = [c, r];
      if (orientation & 2) r = n - 1 - r;
      if (orientation & 4) c = n - 1 - c;
      return r * n + c;
    }));
    paths.set(n, result);
  }
  return result;
}
// Integer coefficients are shared by TypeScript, Numba and CUDA; no hard corner constraint.
export function evaluate(board: Board): number {
  const { size: n, cells } = board; let empty = 0, smooth = 0, merges = 0, monotonic = 0, max = 0;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const v = cells[r * n + c]; if (!v) empty++; max = Math.max(max, v);
    if (c + 1 < n) { const w = cells[r * n + c + 1]; if (v && w) { smooth += Math.abs(v - w); if (v === w) merges += v; } }
    if (r + 1 < n) { const w = cells[(r + 1) * n + c]; if (v && w) { smooth += Math.abs(v - w); if (v === w) merges += v; } }
  }
  for (let axis = 0; axis < 2; axis++) for (let line = 0; line < n; line++) {
    let up = 0, down = 0;
    for (let i = 0; i + 1 < n; i++) {
      const a = cells[axis ? i * n + line : line * n + i] ** 2, b = cells[axis ? (i + 1) * n + line : line * n + i + 1] ** 2;
      up += Math.max(0, b - a); down += Math.max(0, a - b);
    }
    monotonic += Math.min(up, down);
  }
  if (!empty && !merges) return -1000000000;
  let snake = -Infinity;
  for (const path of snakePaths(n)) {
    let value = 0;
    for (let i = 0; i < path.length; i++) {
      const v = cells[path[i]]; value += v * v * (path.length - i) * 4;
      if (i + 1 < path.length) value -= Math.max(0, cells[path[i + 1]] - v) ** 2 * 35;
    }
    snake = Math.max(snake, value);
  }
  const corner = Math.max(cells[0], cells[n - 1], cells[n * (n - 1)], cells[n * n - 1]) === max;
  return empty * (280 + max * 20) - smooth * 8 + merges * 35 - monotonic * 12 + max * max * 20 + (corner ? max * max * 35 : 0) + snake;
}
export function directionValue(board: Board, direction: Direction): number {
  const moved = applyMove(board, direction), empty = emptyCells(moved.board);
  let value = 0;
  for (const index of empty) for (const [exponent, probability] of [[1, 0.9], [2, 0.1]]) {
    const cells = [...moved.board.cells]; cells[index] = exponent;
    value += evaluate({ size: board.size, cells }) * probability / empty.length;
  }
  return (empty.length ? value : evaluate(moved.board)) + Math.log2(moved.scoreDelta + 1) * 12;
}
