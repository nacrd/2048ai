import { applyMove, DIRECTIONS, emptyCells, maxTile, reached, RNG, type Direction } from '../core/engine';
import { advance, cloneSnapshot, decode, encode, type Snapshot } from '../core/session';
import { evaluate } from './solver';

export type TasMode = 'fixed' | 'ideal';
export interface TasOptions { mode: TasMode; objective: 'score' | 'target'; target: number; horizon: number; budgetMs: number; maxNodes: number; strategy?: 'exact' | 'beam' | 'hybrid'; beamWidth?: number }
export interface TasAction { direction: Direction; spawn: { index: number; value: 2 | 4 } | null }
export interface TasPlan { options: TasOptions; frames: Snapshot[]; actions: TasAction[]; complete: boolean; provenDepth: number; nodes: number; elapsedMs: number; note: string }
interface Node { snapshot: Snapshot; parent: Node | null; action: TasAction | null; depth: number; rank: number }
class Limit extends Error {}

export function sameSnapshot(a: Snapshot, b: Snapshot): boolean {
  return a.score === b.score && a.moves === b.moves && a.target === b.target && a.rngState === b.rngState && a.board.size === b.board.size && a.board.cells.every((v, i) => v === b.board.cells[i]);
}
export function tasAdvance(s: Snapshot, action: TasAction, mode: TasMode) {
  const transition = applyMove(s.board, action.direction, true);
  if (!transition.moved) throw new Error('TAS 路线包含无效移动');
  if (mode === 'fixed') {
    const next = advance(s, action.direction);
    const spawn = next.spawned === null ? null : { index: next.spawned, value: 2 ** next.snapshot.board.cells[next.spawned] };
    if (JSON.stringify(spawn) !== JSON.stringify(action.spawn)) throw new Error('TAS 出块与固定随机状态不一致');
    return next;
  }
  const empty = emptyCells(transition.board), spawn = action.spawn;
  if (empty.length ? !spawn || !empty.includes(spawn.index) || ![2, 4].includes(spawn.value) : spawn !== null) throw new Error('TAS 理想出块必须为有效空位上的 2 或 4');
  const cells = [...transition.board.cells]; if (spawn) cells[spawn.index] = Math.log2(spawn.value);
  return { snapshot: { ...s, board: { size: s.board.size, cells }, score: s.score + transition.scoreDelta, moves: s.moves + 1 }, transition, spawned: spawn?.index ?? null };
}
export function planTas(start: Snapshot, options: TasOptions, progress?: (value: { depth: number; nodes: number; maxTile: number }) => void): TasPlan {
  const o = { ...options }; validateOptions(o);
  if (start.target !== o.target) throw new Error('TAS 目标与起点不一致');
  const began = performance.now(); let nodes = 0, complete = false, provenDepth = 0;
  const root: Node = { snapshot: cloneSnapshot(start), parent: null, action: null, depth: 0, rank: evaluate(start.board) };
  const strategy = o.strategy ?? 'exact', width = o.beamWidth ?? 128;
  let best = root, beamEnd: Node | null = null;
  let deadline = began + o.budgetMs, nodeLimit = o.maxNodes, lastProgress = began;
  const check = () => { if (++nodes > nodeLimit || performance.now() >= deadline) throw new Limit(); };
  const consider = (n: Node) => {
    const hit = reached(n.snapshot.board, o.target), oldHit = reached(best.snapshot.board, o.target);
    let better: boolean;
    if (o.objective === 'score') better = n.snapshot.score > best.snapshot.score || (n.snapshot.score === best.snapshot.score && n.rank > best.rank);
    else if (hit || oldHit) better = hit && (!oldHit || n.depth < best.depth);
    else better = maxTile(n.snapshot.board) > maxTile(best.snapshot.board) || (maxTile(n.snapshot.board) === maxTile(best.snapshot.board) && n.rank > best.rank);
    if (better) best = n;
  };
  const children = (n: Node): Node[] => {
    const children: Node[] = [];
    for (const direction of DIRECTIONS) {
      const add = (snapshot: Snapshot, action: TasAction) => {
        check(); const child = { snapshot, action, parent: n, depth: n.depth + 1, rank: evaluate(snapshot.board) };
        children.push(child); consider(child);
      };
      if (o.mode === 'fixed') {
        const next = advance(n.snapshot, direction, false); if (!next.transition.moved) continue;
        add(next.snapshot, { direction, spawn: next.spawned === null ? null : { index: next.spawned, value: 2 ** next.snapshot.board.cells[next.spawned] as 2 | 4 } });
      } else {
        const moved = applyMove(n.snapshot.board, direction); if (!moved.moved) continue;
        for (const index of emptyCells(moved.board)) for (const value of [2, 4] as const) {
          const cells = [...moved.board.cells]; cells[index] = Math.log2(value);
          add({ ...n.snapshot, board: { size: moved.board.size, cells }, score: n.snapshot.score + moved.scoreDelta, moves: n.snapshot.moves + 1 }, { direction, spawn: { index, value } });
        }
      }
    }
    return children.sort((a, b) => Number(reached(b.snapshot.board, o.target)) - Number(reached(a.snapshot.board, o.target)) || (o.objective === 'score' ? b.snapshot.score - a.snapshot.score : maxTile(b.snapshot.board) - maxTile(a.snapshot.board)) || b.rank - a.rank);
  };
  const report = (depth: number) => {
    if (progress && performance.now() - lastProgress >= 200) { progress({ depth, nodes, maxTile: maxTile(best.snapshot.board) }); lastProgress = performance.now(); }
  };
  const sum = (s: Snapshot) => s.board.cells.reduce((total, e) => total + (e ? 2 ** e : 0), 0);
  let lowerBound = 0;
  if (o.objective === 'target' && !reached(start.board, o.target)) {
    const mass = sum(start), deficit = Math.max(0, o.target - mass);
    lowerBound = Math.max(Math.ceil(deficit / 4), Math.log2(o.target) - Math.max(2, ...start.board.cells));
    // Fixed RNG draws two numbers per valid move. Its spawn-value prefix is action-independent.
    if (o.mode === 'fixed' && deficit > 0) {
      const rng = new RNG(start.rngState); let added = 0, steps = 0;
      while (added < deficit && steps <= o.horizon) {
        if ((steps & 255) === 0 && performance.now() >= deadline) break;
        added += rng.next() < 0.9 ? 2 : 4; rng.next(); steps++;
      }
      if (added >= deficit || steps > o.horizon) lowerBound = Math.max(lowerBound, steps);
    }
    provenDepth = Math.max(0, Math.min(o.horizon, lowerBound - 1));
  }
  try {
    if (o.objective === 'target' && reached(root.snapshot.board, o.target)) complete = true;
    else {
      if (strategy !== 'exact') {
        if (strategy === 'hybrid') { deadline = began + o.budgetMs * 0.7; nodeLimit = Math.max(1, Math.floor(o.maxNodes * 0.7)); }
        try {
          let frontier = [root];
          for (let depth = 1; depth <= o.horizon && frontier.length; depth++) {
            const unique = new Map<string, Node>();
            for (const parent of frontier) for (const child of children(parent)) {
              const key = `${child.snapshot.rngState}:${child.snapshot.board.cells.join(',')}`, prior = unique.get(key);
              if (!prior || child.snapshot.score > prior.snapshot.score) unique.set(key, child);
            }
            frontier = [...unique.values()].sort((a, b) => Number(reached(b.snapshot.board, o.target)) - Number(reached(a.snapshot.board, o.target)) || (o.objective === 'score' ? b.snapshot.score - a.snapshot.score : 0) || b.rank - a.rank).slice(0, width);
            if (frontier.length) beamEnd = frontier[0];
            report(depth);
            if (o.objective === 'target' && reached(best.snapshot.board, o.target)) break;
          }
        } catch (e) { if (!(e instanceof Limit)) throw e; }
        deadline = began + o.budgetMs; nodeLimit = o.maxNodes;
      } else {
        // Short seed route leaves most of the exact-search budget available for proofs.
        let current = root;
        for (let h = 0; h < Math.min(o.horizon, 64) && !(o.objective === 'target' && reached(current.snapshot.board, o.target)); h++) {
          const next = children(current)[0]; if (!next) break; current = next;
        }
      }
      const search = (limit: number): boolean => {
        const seen = new Map<string, number>();
        const stack = [root];
        while (stack.length) {
          const n = stack.pop()!;
          check(); consider(n);
          if (o.objective === 'target' && reached(n.snapshot.board, o.target)) return true;
          if (n.depth === limit) continue;
          if (o.objective === 'target' && Math.max(Math.ceil(Math.max(0, o.target - sum(n.snapshot)) / 4), Math.log2(o.target) - Math.max(2, ...n.snapshot.board.cells)) > limit - n.depth) continue;
          const key = `${limit - n.depth}:${n.snapshot.rngState}:${n.snapshot.board.cells.join(',')}`;
          const score = seen.get(key);
          if (score !== undefined && (o.objective === 'target' || score >= n.snapshot.score)) continue;
          seen.set(key, n.snapshot.score);
          const next = children(n);
          for (let i = next.length - 1; i >= 0; i--) stack.push(next[i]);
          report(n.depth);
        }
        return false;
      };
      if (o.objective === 'target' && lowerBound > o.horizon) complete = true;
      else if (strategy !== 'beam') {
        if (o.objective === 'score') { search(o.horizon); provenDepth = o.horizon; complete = true; }
        else for (let h = Math.max(1, lowerBound); h <= o.horizon; h++) {
          if (reached(best.snapshot.board, o.target) && h >= best.depth) { provenDepth = best.depth; complete = true; break; }
          if (search(h)) { provenDepth = h; complete = true; break; }
          provenDepth = h; if (h === o.horizon) complete = true;
        }
      }
    }
  } catch (e) { if (!(e instanceof Limit)) throw e; }
  if (strategy === 'beam' && o.objective === 'target' && !reached(best.snapshot.board, o.target) && beamEnd && maxTile(beamEnd.snapshot.board) >= maxTile(best.snapshot.board)) best = beamEnd;
  const path: Node[] = []; for (let n: Node | null = best; n; n = n.parent) path.push(n); path.reverse();
  const hit = reached(best.snapshot.board, o.target);
  const note = complete ? o.objective === 'score' ? `已证明 ${o.horizon} 步以内最高新增分 ${best.snapshot.score - start.score}` : hit ? `已证明最短达标路线：${best.depth} 步` : `已证明 ${o.horizon} 步内无法达标；显示候选路线` : `${strategy === 'beam' ? '束搜索候选' : '预算内候选'}：${hit ? `已找到 ${best.depth} 步达标路线` : `当前最大块 ${maxTile(best.snapshot.board)}`}，未证明最优${o.objective === 'target' ? `；已排除 ${provenDepth} 步以内达标` : ''}`;
  return { options: o, frames: path.map(n => cloneSnapshot(n.snapshot)), actions: path.slice(1).map(n => n.action!), complete, provenDepth, nodes, elapsedMs: performance.now() - began, note };
}
function validateOptions(o: TasOptions) {
  if (!o || !['fixed', 'ideal'].includes(o.mode) || !['score', 'target'].includes(o.objective) || !Number.isSafeInteger(o.horizon) || o.horizon < 1 || !Number.isFinite(o.budgetMs) || o.budgetMs < 1 || o.budgetMs > 60000 || !Number.isSafeInteger(o.maxNodes) || o.maxNodes < 1 || (o.strategy !== undefined && !['exact', 'beam', 'hybrid'].includes(o.strategy)) || (o.beamWidth !== undefined && (!Number.isInteger(o.beamWidth) || o.beamWidth < 1 || o.beamWidth > 4096)) || !Number.isInteger(Math.log2(o.target)) || o.target < 2 || o.target > 2 ** 30) throw new Error('TAS 参数不合法');
}
export function exportTas(plan: TasPlan): string {
  return JSON.stringify({ version: 1, type: '2048-tas', start: JSON.parse(encode(plan.frames[0])), options: plan.options, actions: plan.actions }, null, 2);
}
export function importTas(text: string): TasPlan {
  const data = JSON.parse(text);
  if (!data || data.version !== 1 || data.type !== '2048-tas') throw new Error('不支持的 TAS 路线格式');
  validateOptions(data.options); const start = decode(JSON.stringify(data.start)).snapshot;
  if (start.target !== data.options.target || !Array.isArray(data.actions) || data.actions.length > data.options.horizon) throw new Error('TAS 起点或路线长度不合法');
  const actions: TasAction[] = [], frames = [start];
  for (const raw of data.actions) {
    if (!raw || !Number.isInteger(raw.direction) || raw.direction < 0 || raw.direction > 3 || (raw.spawn !== null && (!raw.spawn || !Number.isInteger(raw.spawn.index) || ![2, 4].includes(raw.spawn.value)))) throw new Error('TAS 动作格式错误');
    const action: TasAction = { direction: raw.direction, spawn: raw.spawn === null ? null : { index: raw.spawn.index, value: raw.spawn.value } };
    frames.push(tasAdvance(frames[frames.length - 1], action, data.options.mode).snapshot); actions.push(action);
  }
  return { options: data.options, actions, frames, complete: false, provenDepth: 0, nodes: 0, elapsedMs: 0, note: '已导入可复现路线；导入文件不携带最优性证明' };
}
