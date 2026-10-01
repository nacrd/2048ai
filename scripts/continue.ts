import { mkdir, readFile, writeFile, rename, realpath } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { decode, encode, cloneSnapshot } from '../src/core/session';
import { reached, maxTile } from '../src/core/engine';
import { exportTas, planTas, sameSnapshot, tasAdvance, type TasAction, type TasMode, type TasPlan } from '../src/ai/tas';

// An offline continuation job, not a benchmark; never overwrite the supplied start.
const args = process.argv.slice(2);
const option = (name: string, fallback: string) => { const i = args.indexOf(`--${name}`); return i < 0 ? fallback : args[i + 1]; };
const source = option('save', ''), output = resolve(option('out', 'artifacts/continuation'));
const mode = option('mode', 'fixed') as TasMode, target = Number(option('target', '131072'));
const seconds = Number(option('seconds', '600')), width = Number(option('width', '128')), window = Number(option('window', '32')), commit = Number(option('commit', '8'));
if (!source || !['fixed', 'ideal'].includes(mode) || !Number.isInteger(Math.log2(target)) || target < 2 || target > 2 ** 30 || !Number.isFinite(seconds) || seconds <= 0 || !Number.isInteger(width) || width < 1 || width > 4096 || !Number.isSafeInteger(window) || window < 1 || !Number.isSafeInteger(commit) || commit < 1) throw new Error('参数错误：--save <存档> --mode fixed|ideal --target 131072 --seconds 600 --width 128 --window 32 --commit 8');
const originalText = await readFile(resolve(source), 'utf8'), original = decode(originalText).snapshot;
const start = { ...cloneSnapshot(original), target }; let current = cloneSnapshot(start);
await mkdir(output, { recursive: true });
const sourcePath = await realpath(resolve(source)), outputPath = await realpath(output);
const normalize = (path: string) => process.platform === 'win32' ? path.toLowerCase() : path;
for (const name of ['original-save.json', 'continued-save.json', 'progress.json', 'route.json']) for (const suffix of ['', '.tmp']) {
  const destination = resolve(outputPath, name + suffix), canonical = await realpath(destination).catch(() => destination);
  if (normalize(sourcePath) === normalize(canonical)) throw new Error('输入存档与输出冲突，请使用新的 --out 目录；不会覆盖输入存档');
}
await writeFile(resolve(output, 'original-save.json'), originalText);
const actions: TasAction[] = []; let nodes = 0, rounds = 0, stopped = false, reason = '', lastWrite = 0;
const began = performance.now(), deadline = began + seconds * 1000;
process.on('SIGINT', () => { stopped = true; reason = '用户中断'; });
async function atomic(path: string, content: string) { await mkdir(dirname(path), { recursive: true }); await writeFile(`${path}.tmp`, content); await rename(`${path}.tmp`, path); }
async function checkpoint(status: string, route = false) {
  await atomic(resolve(output, 'continued-save.json'), encode(current));
  const report = { source: resolve(source), mode, target, status, maxTile: maxTile(current.board), movesAdded: actions.length, scoreAdded: current.score - start.score, rngState: current.rngState, rounds, nodes, elapsedSeconds: (performance.now() - began) / 1000, shortestProven: false };
  await atomic(resolve(output, 'progress.json'), JSON.stringify(report, null, 2));
  if (route) {
    const plan: TasPlan = { options: { mode, objective: 'target', target, horizon: Math.max(1, actions.length), budgetMs: 60000, maxNodes: Math.max(1, nodes), strategy: 'beam', beamWidth: width }, frames: [start], actions, complete: false, provenDepth: 0, nodes, elapsedMs: performance.now() - began, note: status };
    await atomic(resolve(output, 'route.json'), exportTas(plan));
  }
  lastWrite = performance.now(); console.log(JSON.stringify(report));
}
await checkpoint('开始续玩', true);
while (!stopped && performance.now() < deadline && !reached(current.board, target)) {
  const plan = planTas(current, { mode, objective: 'target', target, horizon: window, budgetMs: Math.max(1, Math.min(5000, deadline - performance.now())), maxNodes: 1000000, strategy: 'beam', beamWidth: width });
  nodes += plan.nodes; rounds++;
  if (!plan.actions.length) { reason = '本轮未找到可执行路线；不代表该存档全局不可达'; break; }
  for (let i = 0; i < Math.min(commit, plan.actions.length) && !reached(current.board, target); i++) {
    const next = tasAdvance(current, plan.actions[i], mode);
    if (!sameSnapshot(next.snapshot, plan.frames[i + 1])) throw new Error('续玩路线状态不一致');
    current = next.snapshot; actions.push(plan.actions[i]);
  }
  if (performance.now() - lastWrite > 10000) await checkpoint('续玩中', true);
  await new Promise<void>(resolve => setImmediate(resolve));
}
reason = reached(current.board, target) ? `已达到 ${target}，未证明最短路线` : reason || '本轮计算时限到达，已保存续玩检查点';
await checkpoint(reason, true);
