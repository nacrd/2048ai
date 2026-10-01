import type { Board } from '../core/engine';
import type { Options, SolveResult } from './solver';
import { strongFallback } from './strong';

export type AutoStyle = 'fast' | 'balanced' | 'steady';
export interface AutoSettings { style: AutoStyle; maxBudgetMs: number }
export interface AutoProbe extends ReturnType<typeof strongFallback> { size: number; mass: number; roughness: number }
export interface AutoStats {
  version: 'auto-v1'; style: AutoStyle; level: '宽松' | '复杂' | '危险'; complexity: number;
  horizon: number; trajectories: number; maxNodes: number; budgetMs: number; searchBudgetMs: number;
  probeMs: number; fallbackMs?: number; effectiveTarget: number; objective: 'score' | 'target'; reason: string; estimatedBackend: 'cpu' | 'cuda';
}
interface Feedback { rate?: number; depth?: number; horizon?: number; budget?: number; pressure: number; zeroSuccess: boolean }
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const profiles = {
  fast: { time: .6, steps: .8, samples: .5 },
  balanced: { time: 1, steps: 1, samples: 1 },
  steady: { time: 1.8, steps: 1.25, samples: 2 },
};
// Probe sees only the board, never a saved RNG state. Keep expensive rule/score work off the UI.
export function probeAuto(board: Board, o: Options): AutoProbe {
  const base = strongFallback(board, { ...o, strong: o.algorithm === 'strong' ? o.strong : undefined }), n = board.size, maximum = Math.max(1, ...board.cells);
  let edges = 0, changes = 0;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const at = r * n + c;
    for (const next of [c + 1 < n ? at + 1 : -1, r + 1 < n ? at + n : -1]) if (next >= 0 && board.cells[at] && board.cells[next]) { edges++; changes += Math.abs(board.cells[at] - board.cells[next]); }
  }
  return { ...base, size: n, mass: board.cells.reduce((sum, e) => sum + (e ? 2 ** e : 0), 0), roughness: edges ? changes / (edges * maximum) : 0 };
}

export class AutoTuner {
  private feedback = new Map<string, Feedback>();
  reset() { this.feedback.clear(); }
  private key(o: Options, size: number, backend: 'cpu' | 'cuda') { return `${o.algorithm}:${o.objective}:${size}:${backend}`; }
  plan(probe: AutoProbe, base: Options, settings: AutoSettings, backend: 'cpu' | 'cuda', probeMs: number, totalBudgetMs?: number): { options: Options; stats: AutoStats } {
    if (!['strong', 'expectimax', 'rollout'].includes(base.algorithm)) throw new Error('自动参数仅用于普通近似 AI');
    if (!Object.hasOwn(profiles, settings.style) || !Number.isSafeInteger(settings.maxBudgetMs) || settings.maxBudgetMs < 30 || settings.maxBudgetMs > 5000 || !Number.isFinite(probeMs) || probeMs < 0 || (totalBudgetMs !== undefined && (!Number.isSafeInteger(totalBudgetMs) || totalBudgetMs < 30 || totalBudgetMs > settings.maxBudgetMs))) throw new Error('自动参数偏好或时间上限不合法');
    const prefs = profiles[settings.style], previous = this.feedback.get(this.key(base, probe.size, backend));
    const { profile, choices, risks } = probe, emptyRatio = profile.empty / (probe.size ** 2), roots = Math.max(1, choices.length);
    const ranked = [...choices].sort((a, b) => b.value - a.value);
    const gap = ranked.length < 2 ? 1 : Math.max(0, ranked[0].value - ranked[1].value);
    const close = clamp(1 - gap / .04, 0, 1);
    const largestRisk = Math.max(0, ...risks.map(r => r.deadProbability));
    const bestRisk = risks.find(r => r.direction === ranked[0]?.direction)?.deadProbability ?? 0;
    const high = clamp((Math.log2(Math.max(2, profile.maximum)) - 10) / 7, 0, 1);
    const congestion = clamp(1 - emptyRatio * 2, 0, 1);
    let complexity = clamp(.45 * congestion + .2 * high + .15 * probe.roughness + .12 * close + .08 * Number(roots <= 2), 0, 1);
    const danger = profile.empty <= Math.max(1, Math.floor(probe.size ** 2 * .125)) || bestRisk > 0 || (emptyRatio <= .25 && largestRisk > 0);
    if (danger) complexity = Math.max(.8, complexity);
    if (roots <= 1 && !danger) complexity *= .35;
    const level: AutoStats['level'] = danger ? '危险' : complexity >= .35 ? '复杂' : '宽松';
    const reasons = [level === '危险' ? '空位少或存在下一次出块即死风险' : level === '复杂' ? '拥挤、大块或候选接近' : '空间充足'];
    const pressure = previous?.pressure ?? 0;
    let budget = (35 + 650 * complexity ** 2) * prefs.time * (1 + .25 * pressure);
    // Raise immediately for new danger; decay more slowly to avoid alternating long/short moves.
    if (previous?.budget && budget < previous.budget) budget = Math.max(budget, previous.budget * .75);
    budget = totalBudgetMs ?? Math.round(clamp(budget, 30, settings.maxBudgetMs));
    const available = Math.max(1, budget - probeMs);
    let horizon: number, trajectories = base.trajectories, maxNodes: number;
    if (base.algorithm === 'rollout') {
      horizon = Math.round((32 + 128 * complexity) * prefs.steps / 8) * 8;
      if (previous?.zeroSuccess && base.objective === 'target') { horizon = Math.round(horizon * 1.25 / 8) * 8; reasons.push('近期无成功样本，优先延长模拟'); }
      horizon = clamp(horizon, 8, 512); // Automatic policy range only; manual H remains unrestricted.
      const rate = previous?.rate ?? (backend === 'cuda' ? 10000 : 40 * (4 / probe.size) ** 2);
      const capacity = Math.max(1, rate * available * .75 / roots), minimumSamples = settings.style === 'steady' ? 32 : 16;
      if (capacity < horizon * minimumSamples) horizon = Math.max(8, Math.floor(capacity / minimumSamples / 8) * 8);
      let desired = Math.round((backend === 'cuda' ? 512 : 128) * (1 + 7 * complexity) * prefs.samples * (1 + pressure));
      if (base.objective === 'target' && profile.effectiveTarget > probe.mass + 4 * horizon) {
        desired = minimumSamples; reasons.push('本窗口质量不足以达标，避免堆叠无效样本');
      }
      const count = clamp(Math.floor(Math.min(desired, capacity / horizon)), minimumSamples, 65536);
      trajectories = 2 ** Math.floor(Math.log2(count));
      maxNodes = 1000000; // Rollout limits samples/time; this field is not a rollout work cap.
    } else {
      horizon = clamp(Math.round((3 + 7 * complexity) * prefs.steps) - Math.max(0, probe.size - 4), 2, 12);
      if (previous?.depth !== undefined && pressure > .35) horizon = Math.min(horizon, Math.max(danger ? 4 : 3, previous.depth + 2));
      if (roots <= 1) horizon = 1;
      const rate = previous?.rate ?? 200 * (4 / probe.size) ** 2;
      maxNodes = Math.round(clamp(rate * available * 1.5, 2000, 4000000));
    }
    if (probeMs >= budget) reasons.push('探测/启动已占用预算，使用最小剩余计算');
    // Auto continue advances the search goal for all approximate modes; stored goal is unchanged.
    const objective = profile.objective, target = profile.effectiveTarget;
    if (base.objective === 'target' && objective === 'score') reasons.push('已达自动目标上限，转为提高得分');
    else if (base.objective === 'target' && target !== base.target) reasons.push(`继续目标 ${target}`);
    const options: Options = { ...base, objective, target, continueAfterTarget: false, horizon, trajectories, maxNodes, budgetMs: available };
    return { options, stats: { version: 'auto-v1', style: settings.style, level, complexity, horizon, trajectories, maxNodes, budgetMs: budget, searchBudgetMs: available, probeMs, effectiveTarget: target, objective, reason: reasons.join('；'), estimatedBackend: backend } };
  }
  observe(base: Options, probe: AutoProbe, plan: AutoStats, result: SolveResult, searchMs: number) {
    if (!Number.isFinite(searchMs) || searchMs <= 0 || !result.choices.length) return;
    const key = this.key(base, probe.size, result.backend), previous = this.feedback.get(key);
    const ranked = [...result.choices].sort((a, b) => b.value - a.value);
    let ambiguous = false;
    if (ranked.length > 1) {
      const [first, second] = ranked;
      if (first.confidence && second.confidence) ambiguous = first.confidence[0] <= second.confidence[1];
      else {
        const difference = first.value - second.value;
        ambiguous = result.algorithm === 'strong' ? difference < .015 : difference / (Math.abs(first.value) + Math.abs(second.value) + 1) < .02;
      }
    }
    const zeroSuccess = result.algorithm === 'rollout' && !!ranked[0].confidence && ranked.every(c => c.value === 0);
    const incomplete = !result.complete && result.strongStats?.stopReason !== '自适应提前停止';
    const pressure = .65 * (previous?.pressure ?? 0) + .35 * Number(ambiguous || incomplete);
    let rate = previous?.rate;
    // Startup/queue outliers must not become an aggressive throughput calibration.
    if (result.nodes > 0 && searchMs <= Math.max(100, plan.searchBudgetMs * 2)) {
      const measured = result.nodes / searchMs;
      rate = rate === undefined ? measured : .75 * rate + .25 * clamp(measured, rate / 4, rate * 4);
    }
    this.feedback.set(key, { rate, pressure, zeroSuccess, depth: result.depth, horizon: plan.horizon, budget: plan.budgetMs });
  }
}

export function autoSummary(stats: AutoStats, algorithm: Options['algorithm']): string {
  return `自动 · ${stats.level} · ${stats.budgetMs}ms软预算 · ${algorithm === 'rollout' ? `模拟${stats.horizon}步 / 每方向请求${stats.trajectories}条` : `深度上限${stats.horizon} / ${stats.maxNodes.toLocaleString()}节点`} · ${stats.reason}`;
}
