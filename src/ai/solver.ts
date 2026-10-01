import { applyMove, DIRECTIONS, emptyCells, hashSeed, legalMoves, reached, RNG, spawnTile, trajectorySeed, type Board, type Direction } from '../core/engine';
export type Algorithm = 'expectimax' | 'rollout' | 'exact';
export type Objective = 'score' | 'target';
export interface Options { algorithm: Algorithm; objective: Objective; target: number; budgetMs: number; horizon: number; trajectories: number; seed: number; maxNodes?: number }
export interface Choice { direction: Direction; value: number; samples?: number; confidence?: [number, number] }
export interface SolveResult { direction: Direction | null; choices: Choice[]; algorithm: Algorithm; backend: 'cpu' | 'cuda'; complete: boolean; depth: number; nodes: number; elapsedMs: number; note: string }

// Same lightweight policy is implemented by the CPU and CUDA rollout engines.
export function evaluate(board: Board): number {
  const { size: n, cells } = board;
  let empty = 0, smooth = 0, merges = 0, monotonic = 0, max = 0;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const v = cells[r * n + c];
    if (!v) empty++;
    max = Math.max(max, v);
    if (c + 1 < n) { const w = cells[r * n + c + 1]; if (v && w) { smooth += Math.abs(v - w); if (v === w) merges++; } }
    if (r + 1 < n) { const w = cells[(r + 1) * n + c]; if (v && w) { smooth += Math.abs(v - w); if (v === w) merges++; } }
  }
  for (let axis = 0; axis < 2; axis++) for (let line = 0; line < n; line++) {
    let up = 0, down = 0;
    for (let i = 0; i + 1 < n; i++) {
      const a = cells[axis ? i * n + line : line * n + i];
      const b = cells[axis ? (i + 1) * n + line : line * n + i + 1];
      up += Math.max(0, b - a); down += Math.max(0, a - b);
    }
    monotonic += Math.min(up, down);
  }
  const corner = Math.max(cells[0], cells[n - 1], cells[n * (n - 1)], cells[n * n - 1]) === max;
  return empty * 280 - smooth * 8 + merges * 35 - monotonic * 65 + max * 20 + (corner ? max * 45 : 0);
}
export function greedy(board: Board): Direction | null {
  let best: Direction | null = null, value = -Infinity;
  for (const d of DIRECTIONS) {
    const m = applyMove(board, d);
    if (m.moved) { const q = evaluate(m.board) + Math.log2(m.scoreDelta + 1) * 12; if (q > value) { value = q; best = d; } }
  }
  return best;
}
class Limit extends Error {}
export function wilson(successes: number, count: number): [number, number] {
  if (!count) return [0, 1];
  const z = 1.959963984540054, p = successes / count, a = 1 + z * z / count;
  const mean = (p + z * z / (2 * count)) / a;
  const delta = z * Math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / a;
  return [Math.max(0, mean - delta), Math.min(1, mean + delta)];
}
export function rolloutOne(board: Board, direction: Direction, id: number, o: Options): number {
  const rng = new RNG(trajectorySeed(o.seed, direction, id));
  let b = board, score = 0, action: Direction | null = direction;
  for (let step = 0; step < o.horizon; step++) {
    if (o.objective === 'target' && reached(b, o.target)) return 1;
    if (action === null) break;
    const m = applyMove(b, action);
    if (!m.moved) break;
    score += m.scoreDelta; b = spawnTile(m.board, rng).board;
    action = greedy(b);
  }
  return o.objective === 'target' ? Number(reached(b, o.target)) : score;
}
export function solve(board: Board, options: Options): SolveResult {
  const start = performance.now();
  const o = { ...options, maxNodes: options.maxNodes ?? 250000 };
  let nodes = 0, depth = 0, complete = false;
  const check = () => {
    nodes++;
    if (nodes > o.maxNodes || ((nodes & 63) === 0 && performance.now() - start >= o.budgetMs)) throw new Limit();
  };
  const legal = legalMoves(board);
  const fallback = legal.map(d => ({ direction: d, value: evaluate(applyMove(board, d).board) }));
  let choices: Choice[] = o.algorithm === 'exact' ? [] : fallback;
  let note = '';
  if (o.algorithm === 'rollout') {
    const sums = new Map(legal.map(d => [d, { sum: 0, n: 0 }]));
    // Complete one round for all candidates so a tiny budget cannot favor the first direction.
    for (let id = 0; id < o.trajectories; id++) {
      if (id && performance.now() - start >= o.budgetMs) break;
      for (const d of legal) { const s = sums.get(d)!; s.sum += rolloutOne(board, d, id, o); s.n++; nodes += o.horizon; }
    }
    choices = legal.map(direction => { const s = sums.get(direction)!; return { direction, value: s.n ? s.sum / s.n : 0, samples: s.n, ...(o.objective === 'target' ? { confidence: wilson(s.sum, s.n) } : {}) }; });
    complete = choices.every(c => c.samples === o.trajectories); depth = o.horizon;
    note = '固定贪心后续策略的模拟估计；不是最优策略胜率';
  } else {
    const exact = o.algorithm === 'exact';
    const cache = new Map<string, number>();
    const key = (b: Board, h: number, kind: string) => `${kind}:${h}:${b.cells.join(',')}`;
    const player = (b: Board, h: number): number => {
      check();
      if (o.objective === 'target' && reached(b, o.target)) return exact ? 1 : 100000;
      if (!h) return exact ? 0 : evaluate(b);
      const k = key(b, h, 'p'); const known = cache.get(k); if (known !== undefined) return known;
      let best = -Infinity;
      for (const d of DIRECTIONS) { const m = applyMove(b, d); if (m.moved) best = Math.max(best, reward(m.scoreDelta) + chance(m.board, h - 1)); }
      if (best === -Infinity) best = exact ? 0 : evaluate(b) - 10000;
      cache.set(k, best); return best;
    };
    const reward = (score: number) => exact ? (o.objective === 'score' ? score : 0) : Math.log2(score + 1) * 12;
    const chance = (b: Board, h: number): number => {
      check();
      if (o.objective === 'target' && reached(b, o.target)) return exact ? 1 : 100000;
      const k = key(b, h, 'c'); const known = cache.get(k); if (known !== undefined) return known;
      const empties = emptyCells(b);
      if (!empties.length) return player(b, h);
      let total = 0;
      for (const i of empties) for (const [v, p] of [[1, 0.9], [2, 0.1]]) {
        const cells = [...b.cells]; cells[i] = v;
        total += p * player({ size: b.size, cells }, h) / empties.length;
      }
      cache.set(k, total); return total;
    };
    const root = (h: number) => legal.map(direction => { const m = applyMove(board, direction); return { direction, value: reward(m.scoreDelta) + chance(m.board, h - 1) }; });
    try {
      if (o.objective === 'target' && reached(board, o.target)) { choices = legal.map(direction => ({ direction, value: exact ? 1 : 100000 })); complete = true; depth = o.horizon; }
      else if (exact) { choices = root(o.horizon); complete = true; depth = o.horizon; }
      else for (let h = 1; h <= o.horizon; h++) { const next = root(h); choices = next; depth = h; complete = h === o.horizon; }
    } catch (e) { if (!(e instanceof Limit)) throw e; }
    note = exact ? (complete ? `H=${o.horizon} 范围内完整搜索，浮点容差 1e-9` : '预算不足，未完成精确分析，不提供最优动作') : '截断 Expectimax 启发式评分，不代表胜率';
  }
  if (!legal.length) { choices = []; complete = true; note = '没有合法移动'; }
  const best = [...choices].sort((a, b) => b.value - a.value || a.direction - b.direction)[0];
  return { direction: best?.direction ?? null, choices, algorithm: o.algorithm, backend: 'cpu', complete, depth, nodes, elapsedMs: performance.now() - start, note };
}
export const DEFAULT_OPTIONS: Options = { algorithm: 'expectimax', objective: 'score', target: 2048, budgetMs: 200, horizon: 4, trajectories: 128, seed: hashSeed('analysis'), maxNodes: 250000 };
