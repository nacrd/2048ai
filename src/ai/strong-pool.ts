import type { Board, Direction } from '../core/engine';
import type { Options, SolveResult } from './solver';
import { adaptiveStop, strongResult, type StrongRoot, type strongFallback } from './strong';

// Workers have independent caches. Only a complete common-depth wave is selectable.
export class StrongPool {
  private idle: Worker[] = [];
  private pending = new Map<Worker, { id: number; reject: (e: Error) => void }>();
  private serial = 0;
  private generation = 0;
  readonly size = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
  cancel() {
    this.generation++;
    for (const [worker, job] of this.pending) { worker.terminate(); job.reject(new Error('已取消')); }
    this.pending.clear();
  }
  private dispatch<T>(board: Board, options: Options, strong: { prepare: true } | { direction: Direction; depth: number; deadline: number }): Promise<T> {
    return new Promise((resolve, reject) => {
      const worker = this.idle.pop() ?? new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      const id = ++this.serial;
      this.pending.set(worker, { id, reject });
      const fail = (e: Error) => {
        if (this.pending.get(worker)?.id !== id) return;
        this.pending.delete(worker); worker.terminate(); reject(e);
      };
      worker.onerror = event => fail(new Error(event.message || '强力 Worker 运行失败'));
      worker.onmessage = event => {
        if (this.pending.get(worker)?.id !== id || event.data.id !== id) return;
        if (event.data.error) { fail(new Error(event.data.error)); return; }
        this.pending.delete(worker); this.idle.push(worker);
        worker.onerror = () => { this.idle = this.idle.filter(w => w !== worker); worker.terminate(); };
        resolve(event.data.result);
      };
      try { worker.postMessage({ id, board, options, strong }); }
      catch (e) { fail(e instanceof Error ? e : new Error(String(e))); }
    });
  }
  async solve(board: Board, o: Options): Promise<SolveResult> {
    const owner = this.generation, began = performance.now(), deadline = Date.now() + o.budgetMs;
    let nodes = 0, hits = 0, pruned = 0, depth = 0, workers = 1, stop = '搜索上限';
    const sizes = new Map<Direction, number>();
    try {
      const { profile, choices: fallback, risks } = await this.dispatch<ReturnType<typeof strongFallback>>(board, o, { prepare: true });
      if (owner !== this.generation) throw new Error('已取消');
      let choices = fallback;
      for (let h = 1; h <= o.horizon && fallback.length; h++) {
        const remaining = (o.maxNodes ?? 250000) - nodes, perRoot = Math.floor(remaining / fallback.length);
        if (perRoot < 1 || Date.now() >= deadline) { stop = '预算不足，保留完整层'; break; }
        const results: StrongRoot[] = [];
        for (let at = 0; at < fallback.length; at += this.size) {
          const roots = fallback.slice(at, at + this.size);
          workers = Math.max(workers, roots.length);
          results.push(...await Promise.all(roots.map(root => this.dispatch<StrongRoot>(board, { ...o, maxNodes: perRoot }, { direction: root.direction, depth: h, deadline }))));
          if (owner !== this.generation) throw new Error('已取消');
        }
        const account = (result: StrongRoot) => { nodes += result.nodes; hits += result.cacheHits; pruned += result.prunedEdges; sizes.set(result.choice.direction, result.cacheEntries); };
        results.forEach(account);
        // Recover unused quotas once, keeping completed roots at this same depth.
        const incomplete = results.map((r, i) => r.complete ? -1 : i).filter(i => i >= 0);
        const retryQuota = incomplete.length ? Math.floor(((o.maxNodes ?? 250000) - nodes) / incomplete.length) : 0;
        if (retryQuota > 0 && Date.now() < deadline) for (let at = 0; at < incomplete.length; at += this.size) {
          const positions = incomplete.slice(at, at + this.size);
          const retry = await Promise.all(positions.map(i => this.dispatch<StrongRoot>(board, { ...o, maxNodes: retryQuota }, { direction: fallback[i].direction, depth: h, deadline })));
          if (owner !== this.generation) throw new Error('已取消');
          retry.forEach((r, i) => { account(r); results[positions[i]] = r; });
        }
        if (results.some(r => !r.complete)) { stop = '预算不足，保留完整层'; break; }
        choices = results.map(r => r.choice); depth = h;
        if (adaptiveStop(profile, choices, risks, h, o)) { stop = '自适应提前停止'; break; }
      }
      const result = strongResult(choices, profile, risks, depth, nodes, performance.now() - began, o, stop, hits, pruned);
      result.workers = workers;
      // Per-root snapshots are diagnostics, not a shared-cache cardinality.
      result.strongStats!.cacheEntries = Math.max(0, ...sizes.values());
      return result;
    } catch (e) { if (owner === this.generation) this.cancel(); throw e; }
  }
}
