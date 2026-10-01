import { applyMove, DIRECTIONS, emptyCells, maxTile, reached, type Board, type Direction } from '../core/engine';
import { evaluate, snakePaths } from './evaluation';
import type { Choice, Options, SolveResult } from './solver';

export interface StrongTuning {
  adaptive?: boolean; stage?: boolean; risk?: boolean; channels?: boolean;
  symmetry?: boolean; reuseCache?: boolean; rootParallel?: boolean;
  chanceCutoff?: number; cacheEntries?: number;
  openingDepth?: number; middleDepth?: number; endgameDepth?: number; clearGap?: number;
}
export const STRONG_VERSION = 'strong-p1-v1';
export interface StrongProfile { stage: 'opening' | 'middle' | 'endgame'; empty: number; maximum: number; suggestedDepth: number; effectiveTarget: number; objective: 'score' | 'target' }
export interface StrongStats { profile: StrongProfile; cacheHits: number; cacheEntries: number; prunedEdges: number; stopReason: string; policyVersion: string; rootRisks: { direction: Direction; deadProbability: number; mobility: number }[] }
export interface StrongRoot { choice: Choice; complete: boolean; nodes: number; cacheHits: number; prunedEdges: number; cacheEntries: number }
const defaults = { adaptive: true, stage: true, risk: true, channels: true, symmetry: true, reuseCache: true, rootParallel: true, chanceCutoff: 0, cacheEntries: 100000, openingDepth: 3, middleDepth: 5, endgameDepth: 7, clearGap: 0.025 };
export function strongSettings(o: Options) {
  const t = { ...defaults, ...o.strong };
  for (const key of ['adaptive', 'stage', 'risk', 'channels', 'symmetry', 'reuseCache', 'rootParallel'] as const) if (typeof t[key] !== 'boolean') throw new Error(`强力设置 ${key} 必须为布尔值`);
  if (!Number.isFinite(t.chanceCutoff) || t.chanceCutoff < 0 || t.chanceCutoff > 0.02 || !Number.isSafeInteger(t.cacheEntries) || t.cacheEntries < 1 || t.cacheEntries > 1000000) throw new Error('强力截断阈值须为0–0.02，缓存条数须为1–1000000');
  for (const key of ['openingDepth', 'middleDepth', 'endgameDepth'] as const) if (!Number.isSafeInteger(t[key]) || t[key] < 1) throw new Error('阶段深度必须为正安全整数');
  if (!Number.isFinite(t.clearGap) || t.clearGap < 0 || t.clearGap > 2) throw new Error('候选差距阈值必须为0–2');
  if (!Number.isSafeInteger(o.horizon) || o.horizon < 1 || !Number.isSafeInteger(o.maxNodes ?? 250000) || (o.maxNodes ?? 250000) < 1 || !Number.isFinite(o.budgetMs) || o.budgetMs <= 0 || !Number.isInteger(Math.log2(o.target)) || o.target < 2 || o.target > 2 ** 30) throw new Error('强力搜索参数不合法');
  return t;
}
export function strongProfile(board: Board, o: Options): StrongProfile {
  const tuning = strongSettings(o);
  const empty = board.cells.filter(e => !e).length, maximum = maxTile(board), ratio = empty / board.cells.length;
  const stage = ratio <= 0.2 || maximum >= 16384 ? 'endgame' : ratio >= 0.5 && maximum < 2048 ? 'opening' : 'middle';
  const objective = o.objective === 'target' && o.continueAfterTarget && maximum >= 2 ** 30 ? 'score' : o.objective;
  const effectiveTarget = o.continueAfterTarget && maximum >= o.target ? Math.min(2 ** 30, maximum * 2) : o.target;
  return { stage, empty, maximum, suggestedDepth: stage === 'opening' ? tuning.openingDepth : stage === 'middle' ? tuning.middleDepth : tuning.endgameDepth, effectiveTarget, objective };
}
// A post-spawn board is nonempty. Empty cells allow a slide; a full board needs an equal neighbor.
function hasMoves(b: Board): boolean {
  const { size: n, cells } = b;
  let occupied = false, empty = false;
  for (let i = 0; i < cells.length; i++) { if (cells[i]) occupied = true; else empty = true; }
  if (empty) return occupied;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if ((c + 1 < n && cells[r * n + c] === cells[r * n + c + 1]) || (r + 1 < n && cells[r * n + c] === cells[(r + 1) * n + c])) return true;
  return false;
}
const transforms = new Map<number, number[][]>();
function boardKey(b: Board, symmetric: boolean): string {
  if (!symmetric) return String.fromCharCode(...b.cells);
  let maps = transforms.get(b.size);
  if (!maps) {
    const n = b.size;
    maps = Array.from({ length: 8 }, (_, t) => b.cells.map((_, i) => {
      let r = Math.floor(i / n), c = i % n;
      if (t & 1) [r, c] = [c, r]; if (t & 2) r = n - 1 - r; if (t & 4) c = n - 1 - c;
      return r * n + c;
    })); transforms.set(n, maps);
  }
  let best: string | undefined;
  for (const map of maps) { const key = String.fromCharCode(...map.map(i => b.cells[i])); if (best === undefined || key < best) best = key; }
  return best!;
}
let cacheScope = '';
const cache = new Map<string, number>();
export function resetStrongCache() { cache.clear(); cacheScope = ''; }

function leaf(b: Board, profile: StrongProfile, t: ReturnType<typeof strongSettings>): number {
  if (profile.objective === 'target' && reached(b, profile.effectiveTarget)) return 1;
  if (!hasMoves(b)) return profile.objective === 'target' ? 0 : -1;
  const n = b.size, count = b.cells.length, maximum = Math.max(...b.cells), empty = b.cells.filter(e => !e).length;
  const phase = t.stage ? maximum >= 14 || empty / count <= 0.2 ? 2 : maximum >= 11 ? 1 : 0 : 0;
  let raw = evaluate(b);
  if (t.risk) {
    // Activity is a geometric heuristic, not a proof of slide reachability.
    const limit = Math.max(2, Math.min(6, maximum - 5)), seen = new Set<number>(); let region = 0;
    for (let i = 0; i < count; i++) if (!seen.has(i) && b.cells[i] <= limit) {
      const stack = [i]; seen.add(i); let size = 0;
      while (stack.length) {
        const at = stack.pop()!; size++;
        const r = Math.floor(at / n), c = at % n;
        for (const next of [r ? at - n : -1, r + 1 < n ? at + n : -1, c ? at - 1 : -1, c + 1 < n ? at + 1 : -1]) if (next >= 0 && !seen.has(next) && b.cells[next] <= limit) { seen.add(next); stack.push(next); }
      }
      region = Math.max(region, size);
    }
    raw += empty * (120 + phase * 120) + region * (25 + phase * 25);
    if (empty <= 2) raw -= (3 - empty) * (400 + phase * 350);
  }
  if (t.channels) {
    let channel = 0;
    for (let i = 0; i < count; i++) if (b.cells[i] >= Math.max(2, maximum - 3)) for (let j = i + 1; j < count; j++) if (b.cells[j] === b.cells[i]) {
      const ri = Math.floor(i / n), rj = Math.floor(j / n), ci = i % n, cj = j % n;
      if (ri !== rj && ci !== cj) continue;
      const step = ri === rj ? 1 : n; let blockers = 0;
      for (let k = i + step; k < j; k += step) if (b.cells[k]) blockers += b.cells[k] >= b.cells[i] ? 3 : 1;
      channel = Math.max(channel, b.cells[i] ** 2 / (1 + blockers));
    }
    let breaks = Infinity;
    for (const path of snakePaths(n)) { let previous = 0, cost = 0; for (const i of path) if (b.cells[i]) { if (previous && b.cells[i] > previous) cost += b.cells[i] - previous; previous = b.cells[i]; } breaks = Math.min(breaks, cost); }
    raw += channel * (8 + phase * 4) - breaks * (30 + phase * 15);
  }
  const scale = count * Math.max(11, maximum) ** 2 * 6;
  const quality = (raw / (Math.abs(raw) + scale) + 1) / 2;
  if (profile.objective === 'score') return quality * 2 - 1;
  const progress = Math.min(1, maximum / Math.log2(profile.effectiveTarget));
  const mass = b.cells.reduce((sum, e) => sum + (e ? 2 ** e : 0), 0);
  return Math.min(0.99, 0.62 * progress ** 2 + 0.26 * quality + 0.12 * Math.min(1, mass / profile.effectiveTarget));
}

export function strongFallback(board: Board, o: Options) {
  const t = strongSettings(o), profile = strongProfile(board, o), choices: Choice[] = [], risks: StrongStats['rootRisks'] = [];
  for (const direction of DIRECTIONS) {
    const moved = applyMove(board, direction); if (!moved.moved) continue;
    const empty = emptyCells(moved.board); let value = 0, deadProbability = 0, mobility = 0;
    for (const index of empty) for (const [e, p] of [[1, 0.9], [2, 0.1]]) {
      moved.board.cells[index] = e; const probability = p / empty.length;
      value += probability * leaf(moved.board, profile, t);
      if (!hasMoves(moved.board)) deadProbability += probability;
      // Expected empty cells after spawning, not a reachability proof.
      mobility += probability * moved.board.cells.filter(v => !v).length;
      moved.board.cells[index] = 0;
    }
    if (!empty.length) { value = leaf(moved.board, profile, t); deadProbability = Number(!hasMoves(moved.board)); }
    if (profile.objective === 'score') value += Math.log2(moved.scoreDelta + 1) * 0.0005;
    choices.push({ direction, value }); risks.push({ direction, deadProbability, mobility });
  }
  return { profile, choices, risks };
}
export function adaptiveStop(profile: StrongProfile, choices: Choice[], risks: StrongStats['rootRisks'], depth: number, o: Options) {
  if (!strongSettings(o).adaptive || depth < profile.suggestedDepth || profile.stage === 'endgame' || risks.some(r => r.deadProbability > 0)) return false;
  const ranked = [...choices].sort((a, b) => b.value - a.value);
  return ranked.length <= 1 || ranked[0].value - ranked[1].value >= strongSettings(o).clearGap;
}
class Limit extends Error {}
export function strongRoot(board: Board, o: Options, direction: Direction, depth: number, deadline = performance.now() + o.budgetMs): StrongRoot {
  if (!Number.isSafeInteger(depth) || depth < 1 || depth > o.horizon) throw new Error('根搜索深度超出范围');
  const t = strongSettings(o), profile = strongProfile(board, o);
  const scope = JSON.stringify([STRONG_VERSION, board.size, profile.objective, profile.effectiveTarget, t.stage, t.risk, t.channels, t.symmetry, t.chanceCutoff]);
  if (!t.reuseCache || cacheScope !== scope) { cache.clear(); cacheScope = scope; }
  while (cache.size > t.cacheEntries) cache.delete(cache.keys().next().value!);
  let nodes = 0, cacheHits = 0, prunedEdges = 0;
  const check = () => { if (nodes >= (o.maxNodes ?? 250000) || performance.now() >= deadline) throw new Limit(); nodes++; };
  const store = (key: string, value: number) => { if (cache.size >= t.cacheEntries && !cache.has(key)) cache.delete(cache.keys().next().value!); cache.set(key, value); };
  const reward = (delta: number) => profile.objective === 'score' ? Math.log2(delta + 1) * 0.0005 : 0;
  type Kind = 'p' | 'c';
  interface Edge { b: Board; h: number; kind: Kind; probability: number; reward: number; substitute?: boolean }
  interface Frame extends Edge { iterator?: Generator<Edge>; key?: string; value: number }
  function* edges(b: Board, h: number, kind: Kind): Generator<Edge> {
    if (kind === 'p') {
      for (const d of DIRECTIONS) { const moved = applyMove(b, d); if (moved.moved) yield { b: moved.board, h: h - 1, kind: 'c', probability: 1, reward: reward(moved.scoreDelta) }; }
    } else {
      const empty = emptyCells(b);
      if (!empty.length) yield { b, h, kind: 'p', probability: 1, reward: 0 };
      for (const index of empty) for (const [e, p] of [[1, 0.9], [2, 0.1]]) {
        const cells = [...b.cells]; cells[index] = e; const probability = p / empty.length;
        yield { b: { size: b.size, cells }, h, kind: 'p', probability, reward: 0, substitute: probability < t.chanceCutoff };
      }
    }
  }
  const moved = applyMove(board, direction); if (!moved.moved) throw new Error('强力根动作无效');
  const stack: Frame[] = [{ b: moved.board, h: depth - 1, kind: 'c', probability: 1, reward: 0, value: 0 }]; let result = 0;
  const finish = (value: number) => {
    const child = stack.pop()!; if (child.key) store(child.key, value);
    const parent = stack.at(-1);
    if (!parent) result = value;
    else if (parent.kind === 'p') parent.value = Math.max(parent.value, child.reward + value);
    else parent.value += child.probability * value;
  };
  try {
    while (stack.length) {
      const frame = stack.at(-1)!;
      if (!frame.iterator) {
        check();
        if (profile.objective === 'target' && reached(frame.b, profile.effectiveTarget)) { finish(1); continue; }
        if (frame.substitute || (frame.kind === 'p' && !frame.h)) { if (frame.substitute) prunedEdges++; finish(leaf(frame.b, profile, t)); continue; }
        frame.key = `${frame.kind}:${frame.h}:${boardKey(frame.b, t.symmetry)}`;
        const known = cache.get(frame.key); if (known !== undefined) { cacheHits++; finish(known); continue; }
        frame.iterator = edges(frame.b, frame.h, frame.kind);
      }
      const next = frame.iterator.next();
      if (next.done) finish(frame.value === -Infinity ? leaf(frame.b, profile, t) : frame.value);
      else stack.push({ ...next.value, value: next.value.kind === 'p' ? -Infinity : 0 });
    }
    return { choice: { direction, value: reward(moved.scoreDelta) + result }, complete: true, nodes, cacheHits, prunedEdges, cacheEntries: cache.size };
  } catch (e) { if (!(e instanceof Limit)) throw e; return { choice: { direction, value: 0 }, complete: false, nodes, cacheHits, prunedEdges, cacheEntries: cache.size }; }
}
export function strongResult(choices: Choice[], profile: StrongProfile, risks: StrongStats['rootRisks'], depth: number, nodes: number, elapsedMs: number, o: Options, stopReason: string, cacheHits = 0, prunedEdges = 0): SolveResult {
  const best = [...choices].sort((a, b) => b.value - a.value || a.direction - b.direction)[0];
  return { direction: best?.direction ?? null, choices, algorithm: 'strong', backend: 'cpu', complete: !choices.length || depth === o.horizon, depth, nodes, elapsedMs, valueKind: 'heuristic', policyVersion: STRONG_VERSION,
    note: `强力 ${profile.stage === 'opening' ? '开局' : profile.stage === 'middle' ? '中盘' : '残局'} · 完成深度 ${depth}/${o.horizon} · ${stopReason}；阶段/风险启发式估值，不是胜率${profile.effectiveTarget !== o.target ? `；继续目标 ${profile.effectiveTarget}` : ''}${prunedEdges ? `；近似替代 ${prunedEdges} 个出块分支` : ''}`,
    strongStats: { profile, rootRisks: risks, cacheHits, cacheEntries: cache.size, prunedEdges, stopReason, policyVersion: STRONG_VERSION } };
}
export function solveStrong(board: Board, o: Options): SolveResult {
  const began = performance.now(), deadline = began + o.budgetMs;
  const { profile, choices: fallback, risks } = strongFallback(board, o);
  let choices = fallback, depth = 0, nodes = 0, hits = 0, pruned = 0, stop = '搜索上限';
  for (let h = 1; h <= o.horizon && fallback.length; h++) {
    const next: Choice[] = []; let complete = true;
    for (const root of fallback) {
      const remaining = (o.maxNodes ?? 250000) - nodes;
      if (remaining < 1 || performance.now() >= deadline) { complete = false; break; }
      const result = strongRoot(board, { ...o, maxNodes: remaining }, root.direction, h, deadline);
      nodes += result.nodes; hits += result.cacheHits; pruned += result.prunedEdges;
      if (!result.complete) { complete = false; break; } next.push(result.choice);
    }
    if (!complete) { stop = '预算不足，保留完整层'; break; }
    choices = next; depth = h;
    if (adaptiveStop(profile, choices, risks, h, o)) { stop = '自适应提前停止'; break; }
  }
  return strongResult(choices, profile, risks, depth, nodes, performance.now() - began, o, stop, hits, pruned);
}
