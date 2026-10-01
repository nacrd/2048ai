import type { Board, Direction } from '../core/engine';
import { wilson, type Choice, type Options, type RolloutSlice, type SolveResult } from './solver';

// Split independent trajectory IDs, never a probability tree or a TAS proof.
export class RolloutPool {
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

  private dispatch(board: Board, options: Options, offset: number, count: number, includeTies: boolean): Promise<RolloutSlice> {
    return new Promise((resolve, reject) => {
      const worker = this.idle.pop() ?? new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      const id = ++this.serial;
      this.pending.set(worker, { id, reject });
      const fail = (e: Error) => {
        if (this.pending.get(worker)?.id !== id) return;
        this.pending.delete(worker); worker.terminate(); reject(e);
      };
      worker.onerror = event => fail(new Error(event.message || '并行 Worker 运行失败'));
      worker.onmessage = event => {
        if (this.pending.get(worker)?.id !== id || event.data.id !== id) return;
        if (event.data.error) { fail(new Error(event.data.error)); return; }
        this.pending.delete(worker); this.idle.push(worker);
        worker.onerror = () => { this.idle = this.idle.filter(w => w !== worker); worker.terminate(); };
        resolve(event.data.result);
      };
      try { worker.postMessage({ id, board, options, batch: { offset, count, includeTies } }); }
      catch (e) { fail(e instanceof Error ? e : new Error(String(e))); }
    });
  }

  async solve(board: Board, o: Options): Promise<SolveResult> {
    const owner = this.generation, began = performance.now();
    let count = 0, chunk = 1, workers = 0;
    let directions: Direction[] = [], ties: number[] = [], sums: number[] = [];
    try {
      while (count < o.trajectories && (!count || performance.now() - began < o.budgetMs)) {
        const jobs: Promise<RolloutSlice>[] = [];
        let offset = count;
        for (let i = 0; i < this.size && offset < o.trajectories; i++) {
          const length = Math.min(chunk, o.trajectories - offset);
          jobs.push(this.dispatch(board, o, offset, length, offset === 0)); offset += length;
        }
        workers = Math.max(workers, jobs.length);
        const round = performance.now(), results = await Promise.all(jobs);
        if (owner !== this.generation) throw new Error('已取消');
        if (!count) { directions = results[0].directions; ties = results[0].tieBreaks!; sums = directions.map(() => 0); }
        // Promise.all preserves chunk order; sum IDs in exactly the serial order.
        for (const result of results) {
          if (result.directions.length !== directions.length || result.directions.some((d, i) => d !== directions[i])) throw new Error('并行方向状态不一致');
          for (let i = 0; i < result.count; i++) for (let d = 0; d < directions.length; d++) sums[d] += result.values[d * result.count + i];
          count += result.count;
        }
        if (!directions.length) break;
        const roundMs = Math.max(0.1, performance.now() - round), remaining = o.budgetMs - (performance.now() - began);
        const desiredMs = Math.max(0.5, Math.min(10, remaining / 2));
        chunk = Math.min(256, Math.max(1, chunk >> 2, Math.min(chunk * 4, Math.floor(chunk * desiredMs / roundMs))));
      }
      const choices: Choice[] = directions.map((direction, i) => ({ direction, value: sums[i] / count, tieBreak: ties[i], samples: count, ...(o.objective === 'target' ? { confidence: wilson(sums[i], count) } : {}) }));
      const best = [...choices].sort((a, b) => b.value - a.value || (b.tieBreak ?? 0) - (a.tieBreak ?? 0) || a.direction - b.direction)[0];
      let note = directions.length ? '固定贪心后续策略的模拟估计；不是最优策略胜率' : '没有合法移动';
      if (o.objective === 'target' && choices.length && choices.every(c => c.value === 0)) note += '；无成功样本，按预期局面评分选择，概率仍为 0';
      return { direction: best?.direction ?? null, choices, algorithm: 'rollout', backend: 'cpu', complete: !directions.length || count === o.trajectories, depth: o.horizon, nodes: count * directions.length * o.horizon, elapsedMs: performance.now() - began, note, workers };
    } catch (e) {
      if (owner === this.generation) this.cancel();
      throw e;
    }
  }
}
