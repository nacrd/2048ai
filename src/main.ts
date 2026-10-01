import './style.css';
import { ARROWS, DIRECTIONS, fromMatrix, hashSeed, matrix, maxTile, newBoard, reached, RNG, type Board, type Direction, type MoveResult } from './core/engine';
import { advance, cloneSnapshot, decode, encode, terminal, type Snapshot } from './core/session';
import { DEFAULT_OPTIONS, type Options, type SolveResult } from './ai/solver';
import { RolloutPool } from './ai/rollout-pool';
import { exportTas, importTas, sameSnapshot, tasAdvance, type TasOptions, type TasPlan } from './ai/tas';

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => el<HTMLInputElement>(id);
const select = (id: string) => el<HTMLSelectElement>(id);
const STORAGE = '2048-ai-v1';
function readStorage(key: string) { try { return localStorage.getItem(key); } catch { return null; } }
let best = Number(readStorage(`${STORAGE}-best`) || 0);
let history: Snapshot[] = [], replay: number | null = null;
let state: Snapshot, version = 0, running = false, busy = false, timer: ReturnType<typeof setTimeout> | undefined;
let worker: Worker | null = null, controller: AbortController | null = null;
const idleWorkers: Partial<Record<'cpu' | 'tas', Worker>> = {};
const rolloutPool = new RolloutPool();
let pendingReject: ((error: Error) => void) | null = null;
interface Device { available: boolean; name?: string; reason?: string; autoThreshold?: { size: number; minHorizon: number; minTrajectories: number } | null }
let device: Device = { available: false };
let tas: TasPlan | null = null, tasCursor = 0, tasPreview: number | null = null;
const isTas = () => select('algorithm').value === 'tas';
function resetTas() { tas = null; tasCursor = 0; tasPreview = null; showTas(); }
function showTas() {
  el('tas-status').textContent = tas ? `${tas.note} · ${tas.nodes.toLocaleString()} 节点 · ${tas.elapsedMs.toFixed(1)} ms · 新增分 ${tas.frames.at(-1)!.score - tas.frames[0].score}` : '尚未规划。时间预算作用于整条路线；搜索步数限制路线长度。';
  el('tas-position').textContent = `${tasPreview ?? tasCursor} / ${tas?.actions.length ?? 0}${tasPreview !== null ? ' · 预览' : ' · 已执行'}`;
  input('tas-timeline').max = String(tas?.actions.length ?? 0); input('tas-timeline').value = String(tasPreview ?? tasCursor);
  el('tas-actions').replaceChildren();
  tas?.actions.forEach((action, i) => {
    const row = document.createElement('li'); const spawn = action.spawn, n = tas!.frames[0].board.size;
    row.textContent = `${ARROWS[action.direction]} · ${spawn ? `出块 ${spawn.value} 于 (${Math.floor(spawn.index / n) + 1}, ${spawn.index % n + 1})` : '无出块'}`;
    if (i === tasCursor) row.className = 'active'; el('tas-actions').append(row);
  });
}

function notify(message: string, error = false) { el('notice').textContent = message; el('notice').classList.toggle('error', error); }
function clearAnalysis() {
  el('recommendation').textContent = '—'; el('analysis-meta').textContent = '点击「只分析」查看四方向比较。';
  el('choices').replaceChildren();
  for (const d of DIRECTIONS) { const tr = document.createElement('tr'); tr.innerHTML = `<td>${ARROWS[d]}</td><td>—</td><td>—</td>`; el('choices').append(tr); }
}
function cancel() {
  running = false; busy = false; version++; clearTimeout(timer); document.body.dataset.running = 'false';
  worker?.terminate(); worker = null; rolloutPool.cancel(); controller?.abort(); controller = null;
  pendingReject?.(new Error('已取消')); pendingReject = null; updateButtons();
}
function updateButtons() {
  for (const id of ['step', 'analyze', 'tas-plan']) el<HTMLButtonElement>(id).disabled = busy || replay !== null || tasPreview !== null;
  el<HTMLButtonElement>('start').disabled = running || replay !== null || tasPreview !== null;
  el<HTMLButtonElement>('undo').disabled = !history.length;
  const ready = tas !== null && tasCursor < tas.actions.length && sameSnapshot(state, tas.frames[tasCursor]);
  for (const id of ['tas-run', 'tas-step']) el<HTMLButtonElement>(id).disabled = busy || running || replay !== null || tasPreview !== null || !ready;
  el<HTMLButtonElement>('tas-restore').disabled = !tas;
  el<HTMLButtonElement>('tas-export').disabled = !tas;
  input('tas-timeline').disabled = !tas || busy;
}
function store() { try { localStorage.setItem(STORAGE, encode(state)); localStorage.setItem(`${STORAGE}-best`, String(best)); } catch { /* Gameplay works without storage. */ } }
function position(index: number, n: number) { return { left: `calc(${index % n} * (100% + var(--gap)) / ${n})`, top: `calc(${Math.floor(index / n)} * (100% + var(--gap)) / ${n})` }; }
function render(transition?: MoveResult, spawned?: number | null) {
  const shown = tas && tasPreview !== null ? tas.frames[tasPreview] : replay === null ? state : [...history, state][replay]; const b = shown.board;
  el('board').style.setProperty('--size', String(b.size));
  el('background').replaceChildren(...b.cells.map(() => document.createElement('div')));
  const tiles = el('tiles'); tiles.replaceChildren();
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const gap = parseFloat(getComputedStyle(el('background')).gap) || 12;
  const unit = (el('background').getBoundingClientRect().width + gap) / b.size;
  b.cells.forEach((v, i) => {
    if (!v) return;
    const tile = document.createElement('div'); tile.className = `ai-tile e${v}${v > 11 ? ' super' : ''}${v >= 10 || b.size >= 5 ? ' large' : ''}`;
    tile.textContent = String(2 ** v); Object.assign(tile.style, position(i, b.size));
    tile.setAttribute('aria-label', `第${Math.floor(i / b.size) + 1}行第${i % b.size + 1}列：${2 ** v}`); tiles.append(tile);
    if (transition && replay === null && !reduced) {
      const motion = transition.motions.find(m => m.to === i);
      if (motion) {
        const from = motion.from[0], dx = (from % b.size - i % b.size) * unit, dy = (Math.floor(from / b.size) - Math.floor(i / b.size)) * unit;
        tile.animate([{ transform: `translate(${dx}px,${dy}px)`, opacity: motion.from.length > 1 ? 0.7 : 1 }, { transform: 'translate(0,0)', opacity: 1 }], { duration: 130, easing: 'ease-out' });
        if (motion.from.length > 1) tile.animate([{ scale: 1 }, { scale: 1.12 }, { scale: 1 }], { delay: 100, duration: 150 });
      } else if (i === spawned) tile.animate([{ scale: 0.3, opacity: 0 }, { scale: 1, opacity: 1 }], { duration: 180 });
    }
  });
  el('score').textContent = String(shown.score); el('best').textContent = String(best); el('max').textContent = String(maxTile(b)); el('moves').textContent = String(shown.moves);
  el('game-status').textContent = tasPreview !== null ? 'TAS 路线预览' : replay !== null ? '回放中' : terminal(b) ? '游戏结束 · 无合法移动' : reached(b, shown.target) ? `已达到 ${shown.target}！` : `目标 ${shown.target}`;
  input('timeline').max = String(history.length); input('timeline').value = String(replay ?? history.length);
  el('replay-label').textContent = tasPreview !== null ? `TAS 预览 · 第 ${shown.moves} 步` : replay === null ? `当前 · 第 ${state.moves} 步` : `回放 · 第 ${shown.moves} 步`;
  updateButtons();
}
function load(snapshot: Snapshot, past: Snapshot[] = []) {
  cancel(); resetTas(); state = cloneSnapshot(snapshot); history = past.map(cloneSnapshot); replay = null;
  select('size').value = String(state.board.size); input('target').value = String(state.target);
  el('editor-section').hidden = true; clearAnalysis(); best = Math.max(best, state.score); store(); render();
}
function readTarget() {
  const t = Number(input('target').value);
  if (t < 2 || t > 2 ** 30 || !Number.isInteger(Math.log2(t))) throw new Error('目标必须为 2 到 2³⁰ 之间的 2 的幂');
  return t;
}
function fresh() {
  try { const target = readTarget(), rng = new RNG(hashSeed(input('seed').value)); const board = newBoard(Number(select('size').value), rng); load({ board, score: 0, rngState: rng.state, target, moves: 0 }); notify('新游戏已开始。相同种子和操作可复现相同出块。'); }
  catch (e) { notify((e as Error).message, true); }
}
function options(): Options {
  const horizon = Number(input('horizon').value), trajectories = Number(input('trajectories').value), interval = Number(input('interval').value);
  const maxNodes = Number(input('nodes').value);
  if (!Number.isSafeInteger(maxNodes) || maxNodes < 1) throw new Error('节点预算必须为正的安全整数');
  if (!Number.isSafeInteger(horizon) || horizon < 1) throw new Error('搜索 / 模拟步数必须为正的安全整数');
  if (!Number.isInteger(trajectories) || trajectories < 1 || trajectories > 65536) throw new Error('每方向模拟数必须为 1–65536');
  if (!Number.isFinite(interval) || interval < 0 || interval > 5000) throw new Error('移动间隔必须为 0–5000 ms');
  return { ...DEFAULT_OPTIONS, maxNodes, algorithm: select('algorithm').value as Options['algorithm'], objective: select('objective').value as Options['objective'], target: readTarget(), budgetMs: Number(select('budget').value), horizon, trajectories, seed: hashSeed('analysis-independent') };
}
function canPlay() { return !terminal(state.board) && (input('continue').checked || !reached(state.board, state.target)); }
function move(direction: Direction, manual = false) {
  if (manual) { cancel(); replay = null; }
  resetTas();
  if (!canPlay()) { notify('已结束或已达到停止目标。勾选「达到目标后继续游玩」可以继续。'); render(); return; }
  const next = advance(state, direction);
  if (!next.transition.moved) { if (manual) render(); return; }
  history.push(cloneSnapshot(state)); state = next.snapshot; version++; best = Math.max(best, state.score);
  clearAnalysis(); store(); render(next.transition, next.spawned);
}
function workerRequest<T>(kind: 'cpu' | 'tas', payload: object, id: number, progress?: (value: { depth: number; nodes: number; maxTile: number }) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    pendingReject = reject;
    const activeWorker = idleWorkers[kind] ?? (kind === 'cpu' ? new Worker(new URL('./ai/worker.ts', import.meta.url), { type: 'module' }) : new Worker(new URL('./ai/tas-worker.ts', import.meta.url), { type: 'module' }));
    delete idleWorkers[kind]; worker = activeWorker;
    activeWorker.onmessage = event => {
      if (event.data.id !== id || worker !== activeWorker || id !== version) return;
      if (event.data.progress) { progress?.(event.data.progress); return; }
      worker = null; pendingReject = null;
      if (event.data.error) { activeWorker.terminate(); reject(new Error(event.data.error)); }
      else {
        idleWorkers[kind] = activeWorker;
        activeWorker.onerror = () => { if (idleWorkers[kind] === activeWorker) delete idleWorkers[kind]; activeWorker.terminate(); };
        resolve(event.data.result);
      }
    };
    activeWorker.onerror = event => { activeWorker.terminate(); if (worker === activeWorker) { worker = null; pendingReject = null; } reject(new Error(event.message || 'Worker 运行失败')); };
    activeWorker.postMessage({ id, ...payload });
  });
}
async function cpu(board: Board, o: Options, id: number): Promise<SolveResult> {
  if (o.algorithm === 'rollout' && rolloutPool.size > 1 && o.trajectories >= 128 && o.budgetMs >= 200) {
    const result = await rolloutPool.solve(board, o);
    if (id !== version) throw new Error('已取消');
    return result;
  }
  return workerRequest('cpu', { board, options: o }, id);
}
async function compute(board: Board, o: Options, id: number): Promise<SolveResult> {
  const backend = select('backend').value;
  const threshold = device.autoThreshold;
  const useCuda = o.algorithm === 'rollout' && (backend === 'cuda' || (backend === 'auto' && device.available && threshold && board.size === threshold.size && o.horizon >= threshold.minHorizon && o.trajectories >= threshold.minTrajectories));
  if (!useCuda) return cpu(board, o, id);
  const activeController = new AbortController(); controller = activeController;
  const watchdog = setTimeout(() => activeController.abort(), o.budgetMs + 10000);
  try {
    const response = await fetch('/api/solve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: activeController.signal, body: JSON.stringify({ board: matrix(board), ...o, requestId: String(id) }) });
    if (!response.ok) { const detail = await response.json().catch(() => ({})); throw new Error(typeof detail.detail === 'string' ? detail.detail : `HTTP ${response.status}`); }
    const result: SolveResult = await response.json(); if (id !== version) throw new Error('已取消'); return result;
  } catch (e) {
    if (id !== version) throw e;
    notify(`CUDA 不可用：${(e as Error).message}。本次回退 CPU。`, true);
    const result = await cpu(board, o, id); result.note += `；CUDA 失败后回退 CPU：${(e as Error).message}`; return result;
  } finally { clearTimeout(watchdog); if (controller === activeController) controller = null; }
}
function showResult(result: SolveResult) {
  el('backend-badge').textContent = result.backend.toUpperCase(); el('recommendation').textContent = result.direction === null ? '—' : ARROWS[result.direction];
  el('analysis-meta').textContent = `${result.backend.toUpperCase()} · ${result.elapsedMs.toFixed(1)} ms · ${result.algorithm === 'rollout' ? `模拟 ${result.depth} 步` : `完成深度 ${result.depth}`} · ${result.nodes.toLocaleString()} ${result.algorithm === 'rollout' ? '模拟步预算' : '节点'}${result.complete ? ' · 已完成' : ' · 触及预算'}`;
  el('analysis-note').textContent = result.note + (result.workers ? `；CPU ${result.workers} 路并行` : '');
  const probability = select('objective').value === 'target' && result.algorithm !== 'expectimax';
  el('value-heading').textContent = probability ? '达标概率' : result.algorithm === 'expectimax' ? '启发式估值' : '期望新增分';
  el('choices').replaceChildren();
  for (const d of DIRECTIONS) {
    const c = result.choices.find(x => x.direction === d), tr = document.createElement('tr'); if (d === result.direction) tr.className = 'best-choice';
    const value = c ? probability ? `${(c.value * 100).toFixed(2)}%` : c.value.toFixed(2) : result.algorithm === 'exact' && !result.complete ? '未完成' : '不可用';
    const stats = c?.confidence ? `${c.samples} / ${(c.confidence[0] * 100).toFixed(1)}–${(c.confidence[1] * 100).toFixed(1)}%` : String(c?.samples ?? '—');
    tr.innerHTML = `<td>${ARROWS[d]}${d === result.direction ? ' 建议' : ''}</td><td>${value}</td><td>${stats}</td>`; el('choices').append(tr);
  }
}
async function runOnce(play: boolean) {
  if (isTas()) {
    const owner = version; await planRoute();
    if (owner === version && play && tas?.actions.length) executeRoute(running);
    else if (owner === version) { running = false; document.body.dataset.running = 'false'; updateButtons(); }
    return;
  }
  if (busy || replay !== null) return;
  let owner = version; busy = true; updateButtons();
  try {
    const o = options(); notify('正在计算… 可随时暂停。');
    const started = performance.now();
    const result = await compute({ ...state.board, cells: [...state.board.cells] }, o, owner);
    result.elapsedMs = performance.now() - started;
    if (owner !== version) return;
    if (play && result.direction !== null) { move(result.direction); owner = version; }
    showResult(result);
    if (play && result.direction !== null) el('analysis-meta').textContent = `已执行 ${ARROWS[result.direction]} · 上一局面的决策 · ${el('analysis-meta').textContent}`;
    notify(result.note); if (play && result.direction === null) running = false;
  } catch (e) { if (owner === version) { notify((e as Error).message, true); running = false; } }
  finally {
    if (owner === version) {
      busy = false;
      if (running && play && canPlay()) timer = setTimeout(() => void runOnce(true), Number(input('interval').value)); else running = false;
      document.body.dataset.running = String(running); updateButtons();
    }
  }
}
async function planRoute() {
  if (busy || replay !== null || tasPreview !== null) return;
  const owner = version; resetTas(); busy = true; updateButtons();
  try {
    const common = options();
    const o: TasOptions = { mode: select('tas-mode').value as TasOptions['mode'], objective: common.objective, target: common.target, horizon: common.horizon, budgetMs: common.budgetMs, maxNodes: common.maxNodes!, strategy: select('tas-search').value as TasOptions['strategy'], beamWidth: Number(input('tas-width').value) };
    notify('TAS 正在回溯规划整条路线… 暂停可取消。');
    const result = await workerRequest<TasPlan>('tas', { snapshot: cloneSnapshot(state), options: o }, owner, value => {
      el('tas-status').textContent = `搜索中 · 深度 ${value.depth} · ${value.nodes.toLocaleString()} 节点 · 候选最大块 ${value.maxTile}`;
    });
    if (owner !== version) return;
    tas = result; tasCursor = 0; tasPreview = null; showTas();
    el('backend-badge').textContent = 'TAS · CPU'; clearAnalysis(); el('analysis-meta').textContent = '完整路线见 TAS 工作台'; el('analysis-note').textContent = result.note;
    notify(`${o.mode === 'ideal' ? '理想出块（主动控制出块）' : '固定种子'}：${result.note}`);
  } catch (e) { if (owner === version) { notify((e as Error).message, true); running = false; } }
  finally { if (owner === version) { busy = false; updateButtons(); } }
}
function executeRoute(continuous: boolean) {
  if (!tas || busy || replay !== null || tasPreview !== null) return;
  if (!sameSnapshot(state, tas.frames[tasCursor])) { cancel(); notify('当前状态与路线不一致，请回到路线起点或重新规划。', true); return; }
  if (tasCursor === tas.actions.length) { cancel(); notify('TAS 路线已执行完毕。'); return; }
  const next = tasAdvance(state, tas.actions[tasCursor], tas.options.mode);
  if (!sameSnapshot(next.snapshot, tas.frames[tasCursor + 1])) { cancel(); notify('TAS 路线状态校验失败，已停止。', true); return; }
  history.push(cloneSnapshot(state)); state = next.snapshot; version++; tasCursor++; best = Math.max(best, state.score);
  showTas(); store(); render(next.transition, next.spawned);
  running = continuous && tasCursor < tas.actions.length; document.body.dataset.running = String(running);
  if (running) timer = setTimeout(() => executeRoute(true), Number(input('interval').value));
  else notify(`TAS ${tasCursor === tas.actions.length ? '路线完成' : '单步完成'} · ${tasCursor}/${tas.actions.length} 步${tas.options.mode === 'ideal' ? ' · 理想出块' : ''}`);
  updateButtons();
}
async function detect() {
  try { const r = await fetch('/api/health', { signal: AbortSignal.timeout(3000) }); if (!r.ok) throw new Error('服务未启动'); device = await r.json(); }
  catch { device = { available: false, reason: '本机服务未启动，CPU 可独立运行' }; }
  el('device-status').textContent = device.available ? `${device.name} · CUDA 就绪` : '浏览器 CPU 就绪';
  notify(device.available ? `${device.name} 可用。CUDA 用于模拟；自动选择按已校准阈值工作。` : device.reason || 'CUDA 不可用');
}
function editor() {
  cancel(); tasPreview = null; replay = null; showTas(); const n = Number(select('size').value), b = state.board.size === n ? matrix(state.board) : Array.from({ length: n }, () => Array(n).fill(0));
  el('editor-section').hidden = false; const root = el('editor'); root.style.setProperty('--size', String(n)); root.replaceChildren();
  b.flat().forEach((v, i) => { const field = document.createElement('input'); field.type = 'number'; field.min = '0'; field.value = String(v); field.setAttribute('aria-label', `编辑第${Math.floor(i / n) + 1}行第${i % n + 1}列`); root.append(field); });
  render(); notify('编辑后点击「应用棋盘」。尺寸变更在应用或新游戏时生效。');
}
function custom(board: Board) { load({ board, score: 0, rngState: hashSeed(input('seed').value), target: readTarget(), moves: 0 }); notify('自定义棋盘已加载。可分析或开始自动游玩。'); }
function importText(text: string) {
  try {
    text = text.trim();
    if (text.startsWith('{')) { const parsed = decode(text); load(parsed.snapshot, parsed.history); notify('存档与回放已恢复。'); }
    else { const data = text.startsWith('[') ? JSON.parse(text) : text.split(/\r?\n/).filter(Boolean).map(line => line.trim().split(/[\s,]+/).map(Number)); custom(fromMatrix(data)); }
  } catch (e) { notify((e as Error).message, true); }
}
el('new-game').onclick = fresh;
el('start').onclick = () => { if (busy) cancel(); if (isTas() && tas && sameSnapshot(state, tas.frames[tasCursor])) { executeRoute(true); return; } if (!canPlay()) { notify('当前棋盘已结束或已达到停止目标。'); return; } running = true; document.body.dataset.running = 'true'; void runOnce(true); };
el('pause').onclick = () => { cancel(); notify('已暂停。'); };
el('step').onclick = () => { cancel(); if (isTas() && tas && sameSnapshot(state, tas.frames[tasCursor])) executeRoute(false); else void runOnce(true); };
el('analyze').onclick = () => { cancel(); void runOnce(false); };
el('undo').onclick = () => { cancel(); resetTas(); replay = null; const prior = history.pop(); if (prior) { state = prior; clearAnalysis(); store(); render(); notify('已撤销，随机状态同步恢复。'); } };
el('detect').onclick = () => void detect(); el('edit').onclick = editor; select('size').onchange = editor;
el('cancel-editor').onclick = () => { el('editor-section').hidden = true; select('size').value = String(state.board.size); };
el('clear-editor').onclick = () => el('editor').querySelectorAll('input').forEach(field => field.value = '0');
el('apply-editor').onclick = () => { try { const n = Number(select('size').value), values = [...el('editor').querySelectorAll('input')].map(field => Number(field.value)); custom(fromMatrix(Array.from({ length: n }, (_, r) => values.slice(r * n, (r + 1) * n)))); } catch (e) { notify((e as Error).message, true); } };
el('import').onclick = () => importText(el<HTMLTextAreaElement>('matrix').value);
el('export-matrix').onclick = () => { el<HTMLTextAreaElement>('matrix').value = matrix(state.board).map(row => row.join(' ')).join('\n'); el<HTMLTextAreaElement>('matrix').closest('details')!.open = true; };
el('export').onclick = () => {
  const text = encode(state, history); el<HTMLTextAreaElement>('matrix').value = text;
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = '2048-ai-save.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); notify('存档 JSON 已生成，包含随机状态与回放；可复制文本或使用下载文件。');
};
input('file').onchange = async () => { const file = input('file').files?.[0]; if (file) { if (file.size > 20 * 1024 * 1024) notify('存档文件不能超过 20 MB', true); else importText(await file.text()); } input('file').value = ''; };
input('timeline').oninput = () => { cancel(); tasPreview = null; replay = Number(input('timeline').value); clearAnalysis(); showTas(); render(); };
el('live').onclick = () => { cancel(); tasPreview = null; replay = null; showTas(); render(); };
for (const id of ['algorithm', 'backend', 'objective', 'budget', 'horizon', 'nodes', 'trajectories', 'interval', 'continue', 'tas-search', 'tas-width']) el(id).onchange = () => {
  cancel(); if (!['interval', 'continue'].includes(id)) resetTas(); clearAnalysis(); const mode = select('algorithm').value;
  el('tas-panel').hidden = mode !== 'tas';
  if (mode !== 'tas' && Number(select('budget').value) > 5000) select('budget').value = '5000';
  el('mode-help').textContent = mode === 'tas' ? '混合搜索先生成候选，再尝试证明；束搜索适合长路线，完整回溯用于最优性证明。固定种子读取未来 RNG，理想模式控制出块。' : mode === 'exact' ? '完整枚举 H 步内随机分支。预算不足时不提供最优结论。推荐小棋盘和浅层分析。' : mode === 'rollout' ? '增强贪心后续策略的模拟估计；CUDA 适合较大批量。无成功样本时按局面评分择优。时间预算在完整批次间检查。' : '限时搜索 + 多方向结构评分。建议动作属于近似决策。';
  select('backend').disabled = mode !== 'rollout'; input('trajectories').disabled = mode !== 'rollout';
  render();
};
input('target').onchange = () => { cancel(); resetTas(); try { state.target = readTarget(); history = []; replay = null; clearAnalysis(); store(); render(); } catch (e) { input('target').value = String(state.target); render(); notify((e as Error).message, true); } };
select('tas-mode').onchange = () => { cancel(); resetTas(); render(); el('tas-rules').textContent = select('tas-mode').value === 'ideal' ? '理想条件：每步主动选择空位及 2 / 4，不按 90% / 10% 抽样，不消耗游戏 RNG。普通游玩恢复随机出块。' : '固定种子：提前计算当前存档的 RNG，回溯动作；相同起点和动作可复现路线。'; };
el('tas-plan').onclick = () => { cancel(); void planRoute(); };
el('tas-run').onclick = () => { cancel(); executeRoute(true); };
el('tas-step').onclick = () => { cancel(); executeRoute(false); };
el('tas-restore').onclick = () => {
  if (!tas) return; cancel(); state = cloneSnapshot(tas.frames[0]); history = []; replay = null; tasPreview = null; tasCursor = 0; showTas(); clearAnalysis(); store(); render(); notify('已恢复 TAS 起点，包括随机状态。');
};
input('tas-timeline').oninput = () => { if (!tas) return; cancel(); replay = null; tasPreview = Number(input('tas-timeline').value); showTas(); render(); };
el('tas-live').onclick = () => { cancel(); tasPreview = null; replay = null; showTas(); render(); };
el('tas-export').onclick = () => {
  if (!tas) return; const text = exportTas(tas); el<HTMLTextAreaElement>('tas-json').value = text;
  el<HTMLTextAreaElement>('tas-json').closest('details')!.open = true;
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = '2048-tas-route.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); notify('TAS 路线 JSON 已生成。');
};
el('tas-import').onclick = () => {
  try {
    const result = importTas(el<HTMLTextAreaElement>('tas-json').value); load(result.frames[0]); tas = result;
    select('algorithm').value = 'tas'; select('tas-mode').value = result.options.mode; select('objective').value = result.options.objective; input('horizon').value = String(result.options.horizon);
    select('tas-search').value = result.options.strategy ?? 'exact'; input('tas-width').value = String(result.options.beamWidth ?? 128); input('nodes').value = String(result.options.maxNodes);
    el('tas-panel').hidden = false; select('backend').disabled = true; input('trajectories').disabled = true;
    el('tas-rules').textContent = result.options.mode === 'ideal' ? '已加载理想出块路线：出块受控，不消耗游戏 RNG。' : '已加载固定种子路线：提前计算出块。';
    showTas(); render(); notify(result.note);
  } catch (e) { notify((e as Error).message, true); }
};
document.addEventListener('keydown', event => {
  if (event.target instanceof HTMLElement && event.target.closest('input,textarea,select,button,summary')) return;
  const key = event.key.toLowerCase(), maps: Record<string, Direction> = { arrowup: 0, arrowright: 1, arrowdown: 2, arrowleft: 3, w: 0, d: 1, s: 2, a: 3 };
  if (key in maps) { event.preventDefault(); move(maps[key], true); }
});
let touch: [number, number] | null = null;
el('board').onpointerdown = event => { touch = [event.clientX, event.clientY]; el('board').setPointerCapture(event.pointerId); };
el('board').onpointerup = event => { if (!touch) return; const dx = event.clientX - touch[0], dy = event.clientY - touch[1]; touch = null; if (Math.max(Math.abs(dx), Math.abs(dy)) > 24) move(Math.abs(dx) > Math.abs(dy) ? dx > 0 ? 1 : 3 : dy > 0 ? 2 : 0, true); };
try { const saved = readStorage(STORAGE); if (saved) { const parsed = decode(saved); load(parsed.snapshot); } else fresh(); } catch { fresh(); }
select('backend').disabled = true; input('trajectories').disabled = true; void detect();
