import { applyMove, DIRECTIONS, emptyCells, hashSeed, legalMoves, reached, RNG, spawnTile, trajectorySeed, type Board, type Direction } from '../core/engine';
import { directionValue, evaluate } from './evaluation';
import { solveStrong, type StrongStats, type StrongTuning } from './strong';
import type { AutoStats } from './auto-params';
export { evaluate } from './evaluation';
export type Algorithm = 'expectimax' | 'rollout' | 'exact' | 'strong';
export type Objective = 'score' | 'target';
export interface Options { algorithm: Algorithm; objective: Objective; target: number; budgetMs: number; horizon: number; trajectories: number; seed: number; maxNodes?: number; strong?: StrongTuning; continueAfterTarget?: boolean }
export interface Choice { direction: Direction; value: number; tieBreak?: number; samples?: number; confidence?: [number, number] }
export interface SolveResult { direction: Direction | null; choices: Choice[]; algorithm: Algorithm; backend: 'cpu' | 'cuda'; complete: boolean; depth: number; nodes: number; elapsedMs: number; note: string; workers?: number; valueKind?: 'heuristic'; policyVersion?: string; strongStats?: StrongStats; autoStats?: AutoStats }
export interface RolloutSlice { directions: Direction[]; count: number; values: Float64Array; tieBreaks?: number[] }
export function rolloutSlice(board: Board, options: Options, offset: number, count: number, includeTies = false): RolloutSlice {
  const directions = legalMoves(board), values = new Float64Array(directions.length * count);
  for (let i = 0; i < count; i++) for (let d = 0; d < directions.length; d++) values[d * count + i] = rolloutOne(board, directions[d], offset + i, options);
  return { directions, count, values, ...(includeTies ? { tieBreaks: directions.map(d => directionValue(board, d)) } : {}) };
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
    if (o.objective === 'target' && reached(b, o.target)) return 1;
    if (step + 1 < o.horizon) action = greedy(b);
  }
  return o.objective === 'target' ? Number(reached(b, o.target)) : score;
}
export function solve(board: Board, options: Options): SolveResult {
  if (!Number.isSafeInteger(options.horizon) || options.horizon < 1) throw new Error('搜索 / 模拟步数必须为正的安全整数');
  if (options.algorithm === 'strong') return solveStrong(board, options);
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
    choices = legal.map(direction => { const s = sums.get(direction)!; return { direction, value: s.n ? s.sum / s.n : 0, tieBreak: directionValue(board, direction), samples: s.n, ...(o.objective === 'target' ? { confidence: wilson(s.sum, s.n) } : {}) }; });
    complete = choices.every(c => c.samples === o.trajectories); depth = o.horizon;
    note = '固定贪心后续策略的模拟估计；不是最优策略胜率';
    if (o.objective === 'target' && choices.length && choices.every(c => c.value === 0)) note += '；无成功样本，按预期局面评分选择，概率仍为 0';
  } else {
    const exact = o.algorithm === 'exact';
    const cache = new Map<string, number>();
    const key = (b: Board, h: number, kind: string) => `${kind}:${h}:${b.cells.join(',')}`;
    const reward = (score: number) => exact ? (o.objective === 'score' ? score : 0) : Math.log2(score + 1) * 12;
    type Kind = 'p' | 'c';
    interface Edge { board: Board; h: number; kind: Kind; weight: number; reward: number }
    interface Frame extends Edge { edges?: Generator<Edge>; value: number; key?: string }
    function* edges(b: Board, h: number, kind: Kind): Generator<Edge> {
      if (kind === 'p') {
        for (const d of DIRECTIONS) {
          const m = applyMove(b, d);
          if (m.moved) yield { board: m.board, h: h - 1, kind: 'c', weight: 1, reward: reward(m.scoreDelta) };
        }
      } else {
        const empties = emptyCells(b);
        if (!empties.length) yield { board: b, h, kind: 'p', weight: 1, reward: 0 };
        for (const i of empties) for (const [v, p] of [[1, 0.9], [2, 0.1]]) {
          const cells = [...b.cells]; cells[i] = v;
          yield { board: { size: b.size, cells }, h, kind: 'p', weight: p / empties.length, reward: 0 };
        }
      }
    }
    // Explicit frames preserve player-max/chance-average semantics at any requested depth.
    const chance = (b: Board, h: number): number => {
      const frame = (e: Edge): Frame => ({ ...e, value: e.kind === 'p' ? -Infinity : 0 });
      const stack = [frame({ board: b, h, kind: 'c', weight: 1, reward: 0 })];
      let result = 0;
      const finish = (value: number) => {
        const child = stack.pop()!;
        if (child.key) cache.set(child.key, value);
        const parent = stack.at(-1);
        if (!parent) result = value;
        else if (parent.kind === 'p') parent.value = Math.max(parent.value, child.reward + value);
        else parent.value += child.weight * value;
      };
      while (stack.length) {
        const current = stack.at(-1)!;
        if (!current.edges) {
          check();
          if (o.objective === 'target' && reached(current.board, o.target)) { finish(exact ? 1 : 100000); continue; }
          if (current.kind === 'p' && !current.h) { finish(exact ? 0 : evaluate(current.board)); continue; }
          current.key = key(current.board, current.h, current.kind);
          const known = cache.get(current.key);
          if (known !== undefined) { finish(known); continue; }
          current.edges = edges(current.board, current.h, current.kind);
        }
        const next = current.edges.next();
        if (!next.done) stack.push(frame(next.value));
        else finish(current.value === -Infinity ? exact ? 0 : evaluate(current.board) - 10000 : current.value);
      }
      return result;
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
  const best = [...choices].sort((a, b) => b.value - a.value || (b.tieBreak ?? 0) - (a.tieBreak ?? 0) || a.direction - b.direction)[0];
  return { direction: best?.direction ?? null, choices, algorithm: o.algorithm, backend: 'cpu', complete, depth, nodes, elapsedMs: performance.now() - start, note };
}
export const DEFAULT_OPTIONS: Options = { algorithm: 'expectimax', objective: 'score', target: 2048, budgetMs: 200, horizon: 4, trajectories: 128, seed: hashSeed('analysis'), maxNodes: 250000 };
