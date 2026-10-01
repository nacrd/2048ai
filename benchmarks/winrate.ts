import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpus, platform, arch, totalmem } from 'node:os';
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { hashSeed, maxTile, newBoard, RNG } from '../src/core/engine';
import { advance, cloneSnapshot, decode, encode, terminal, type Snapshot } from '../src/core/session';
import { DEFAULT_OPTIONS, solve, wilson, type Algorithm, type Options, type SolveResult } from '../src/ai/solver';
import { resetStrongCache, STRONG_VERSION, strongProfile, strongSettings } from '../src/ai/strong';

const BASELINE = '3c0a2c42619e156cd0a59efaee468b4170fa608f';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const digest = (text: string | Buffer) => createHash('sha256').update(text).digest('hex');
const git = (...args: string[]) => execFileSync('git', args, { cwd: root, encoding: 'utf8', windowsHide: true }).trim();
const help = `Paired CPU evaluation (runs games only when invoked):
npm run bench:ab -- --games 100 --split tuning --budget-mode nodes --nodes 10000 --horizon 4
--split training|tuning|holdout --seed-start 0 --size 4 --max-moves 30000
--budget-mode time|nodes|samples --budget 200 --nodes 1000000 --trajectories 128
--baseline-algorithm expectimax|rollout|exact --candidate-algorithm strong|expectimax|rollout|exact
--objective score|target --target 2048 --horizon 8 --out artifacts/winrate/<new-directory>
--strong-config <JSON file> --save <one saved game> OR --manifest <JSON array of save paths>
--trace-tail 64 --stop-at-target true|false
Node CPU searches are serial; browser worker pools and CUDA need separate performance evaluation.
Target objective requires stop-at-target=true so the frozen policy does not keep selecting an absorbed goal.
Saved-game cases preserve their RNG and are not independent random-win-rate trials.`;
const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i];
  if (key === '--help') { console.log(help); process.exit(0); }
  if (!/^--[a-z-]+$/.test(key) || process.argv[i + 1] === undefined || process.argv[i + 1].startsWith('--') || args.has(key.slice(2))) throw new Error(`Invalid argument ${key}\n${help}`);
  args.set(key.slice(2), process.argv[i + 1]);
}
const allowed = new Set(['games', 'split', 'seed-start', 'size', 'max-moves', 'budget-mode', 'budget', 'nodes', 'trajectories', 'baseline-algorithm', 'candidate-algorithm', 'objective', 'target', 'horizon', 'out', 'strong-config', 'save', 'manifest', 'trace-tail', 'stop-at-target']);
for (const key of args.keys()) if (!allowed.has(key)) throw new Error(`Unknown option --${key}`);
function number(key: string, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER) {
  const value = Number(args.get(key) ?? fallback);
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`--${key} must be an integer in ${min}..${max}`);
  return value;
}
function oneOf<T extends string>(key: string, fallback: T, values: readonly T[]): T {
  const value = args.get(key) ?? fallback;
  if (!values.includes(value as T)) throw new Error(`Invalid --${key}: ${value}`);
  return value as T;
}
const split = oneOf('split', 'tuning', ['training', 'tuning', 'holdout'] as const);
const mode = oneOf('budget-mode', 'nodes', ['time', 'nodes', 'samples'] as const);
const objective = oneOf('objective', 'score', ['score', 'target'] as const);
const a = oneOf<Algorithm>('baseline-algorithm', 'expectimax', ['expectimax', 'rollout', 'exact']);
const b = oneOf<Algorithm>('candidate-algorithm', 'strong', ['strong', 'expectimax', 'rollout', 'exact']);
const stopAtTarget = oneOf('stop-at-target', 'false', ['true', 'false'] as const) === 'true';
const size = number('size', 4, 2, 6), start = number('seed-start', 0, 0), maxMoves = number('max-moves', 30000), tailLength = number('trace-tail', 64, 1, 10000);
const target = number('target', 2048, 2, 2 ** 30);
if (!Number.isInteger(Math.log2(target))) throw new Error('--target must be a power of two');
if (objective === 'target' && !stopAtTarget) throw new Error('Target comparisons require --stop-at-target true; use score objective for whole-game goal series');
if (mode === 'samples' && (a !== 'rollout' || b !== 'rollout')) throw new Error('Sample budgets require rollout on both sides');
if (mode === 'nodes' && (a === 'rollout' || b === 'rollout')) throw new Error('Rollout uses sample/time budgets; its reported nodes are not search nodes');
if (args.has('save') && args.has('manifest')) throw new Error('Choose --save OR --manifest');
const resolveInput = (path: string) => realpathSync(resolve(root, path));
let saves: string[] = [];
if (args.has('save')) saves = [resolveInput(args.get('save')!)];
if (args.has('manifest')) {
  const manifest = resolveInput(args.get('manifest')!), entries: unknown = JSON.parse(readFileSync(manifest, 'utf8'));
  if (!Array.isArray(entries) || !entries.length || entries.some(p => typeof p !== 'string')) throw new Error('Manifest must be a nonempty JSON array of save paths');
  saves = entries.map(p => realpathSync(isAbsolute(p) ? p : resolve(dirname(manifest), p)));
  if (new Set(saves.map(p => p.toLowerCase())).size !== saves.length) throw new Error('Duplicate save cases are not independent trials');
}
const games = number('games', saves.length || 100);
if (saves.length && games !== saves.length) throw new Error('Saved cases run once each; --games must equal number of saves');
if (!Number.isSafeInteger(start + games)) throw new Error('Seed range exceeds safe integers');
const snapshots = saves.map(path => decode(readFileSync(path, 'utf8')).snapshot);
const saveHashes = saves.map(path => digest(readFileSync(path)));
const seeds = Array.from({ length: games }, (_, i) => `winrate-${split}-${start + i}`);
const gameSeeds = seeds.map(hashSeed), analysisSeeds = seeds.map((_, i) => hashSeed(`analysis-${split}-${start + i}`));
if (!saves.length && new Set(gameSeeds).size !== games) throw new Error('Game seed hash collision; choose a different seed range');
const strong: Options['strong'] = args.has('strong-config') ? JSON.parse(readFileSync(resolveInput(args.get('strong-config')!), 'utf8')) : {};
if (!strong || typeof strong !== 'object' || Array.isArray(strong)) throw new Error('Strong configuration must be a JSON object');
const strongKeys = new Set(Object.keys(strongSettings({ ...DEFAULT_OPTIONS, algorithm: 'strong' })));
for (const key of Object.keys(strong)) if (!strongKeys.has(key)) throw new Error(`Unknown strong configuration key ${key}`);
const options: Options = { ...DEFAULT_OPTIONS, objective, target, horizon: number('horizon', 8), trajectories: number('trajectories', 128, 1, 65536),
  budgetMs: mode === 'time' ? number('budget', 200) : 1e12, maxNodes: mode === 'nodes' ? number('nodes', 1000000) : Number.MAX_SAFE_INTEGER,
  continueAfterTarget: false, strong: { ...strong, rootParallel: false } };
strongSettings(options);
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const out = resolve(root, args.get('out') ?? `artifacts/winrate/${stamp}`);
if (existsSync(out)) throw new Error('Output directory already exists; choose a new directory');
mkdirSync(out, { recursive: true });
const actualOut = realpathSync(out);
for (const path of saves) {
  const rel = relative(actualOut, path);
  if (!rel || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))) throw new Error('Output must not contain an input save');
}
function json(path: string, value: unknown) {
  const temp = `${path}.tmp`; writeFileSync(temp, JSON.stringify(value, null, 2)); renameSync(temp, path);
}
// Freeze executable policy plus rules/session, not a copy of today's mutable solver.
const frozenFiles = ['src/ai/solver.ts', 'src/ai/evaluation.ts', 'src/core/engine.ts', 'src/core/fast4.ts', 'src/core/session.ts'];
// Every report owns a fresh frozen directory: never overwrite a user's existing cache or link.
const frozen = join(out, 'frozen-baseline'), hashes: Record<string, string> = {};
mkdirSync(frozen); json(join(frozen, 'package.json'), { type: 'module' });
for (const path of frozenFiles) {
  const content = execFileSync('git', ['show', `${BASELINE}:${path}`], { cwd: root, windowsHide: true });
  hashes[path] = digest(content); const destination = join(frozen, path);
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(destination, content, { flag: 'wx' });
}
const baseline = await import(pathToFileURL(join(frozen, 'src/ai/solver.ts')).href) as { solve: typeof solve };
const baselineSession = await import(pathToFileURL(join(frozen, 'src/core/session.ts')).href) as { advance: typeof advance; terminal: typeof terminal };
const sourceFiles = readdirSync(join(root, 'src'), { recursive: true }).filter((p): p is string => typeof p === 'string' && p.endsWith('.ts')).sort();
const sourceHashes = Object.fromEntries(sourceFiles.map(p => [p.replaceAll('\\', '/'), digest(readFileSync(join(root, 'src', p)))]));
const sourceHash = digest(JSON.stringify(sourceHashes));
for (const path of sourceFiles) {
  const destination = join(out, 'candidate-source/src', path);
  mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, readFileSync(join(root, 'src', path)), { flag: 'wx' });
}
writeFileSync(join(out, 'candidate-source/winrate.ts'), readFileSync(fileURLToPath(import.meta.url)), { flag: 'wx' });
const metadata = { createdAt: new Date().toISOString(), schemaVersion: 1, baseline: { commit: BASELINE, hashes, algorithm: a },
  candidate: { commit: git('rev-parse', 'HEAD'), dirty: !!git('status', '--porcelain'), sourceHash, sourceHashes, harnessHash: digest(readFileSync(fileURLToPath(import.meta.url))), algorithm: b, policyVersion: b === 'strong' ? STRONG_VERSION : 'legacy-current' },
  hardware: { cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem(), os: platform(), arch: arch(), node: process.version },
  execution: 'Node serial CPU, no browser worker pool or CUDA', modelHash: null, tableHits: 0,
  config: { games, split, cohortKind: saves.length ? 'saved-cases' : 'random-games', start, size: saves.length ? undefined : size, maxMovesAdded: maxMoves, tailLength, budgetMode: mode, stopAtTarget, options, strongEffective: strongSettings(options) },
  seeds: saves.length ? undefined : seeds.map((label, i) => ({ label, seed: gameSeeds[i] })), analysisSeeds,
  saves: saves.map((path, i) => ({ path, sha256: saveHashes[i], start: snapshots[i], profile: strongProfile(snapshots[i].board, options) })),
  notes: ['Same initial state and RNG draw stream per valid move; positions follow each board.', 'Strong caches reset between cases, persist between moves. A/B order alternates per case; JIT remains warm.', 'Unknown/censored goals are not counted as failures.', 'Wilson intervals apply only to fully resolved independent random trials. Save cases are descriptive.', 'Decision latency is measured after terminal checking (which initializes movement tables), includes remaining solver cold work and excludes trace/file I/O; percentiles are histogram upper bounds.'] };
json(join(out, 'metadata.json'), metadata);

const bins = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, Infinity];
interface Metrics { decisions: number; elapsed: number; histogram: number[]; depth: Record<string, number>; nodes: number; cacheHits: number; cutoffs: number; adaptiveStops: number; prunedEdges: number }
const metrics = (): Metrics => ({ decisions: 0, elapsed: 0, histogram: bins.map(() => 0), depth: {}, nodes: 0, cacheHits: 0, cutoffs: 0, adaptiveStops: 0, prunedEdges: 0 });
function record(m: Metrics, r: SolveResult, elapsed: number) {
  m.decisions++; m.elapsed += elapsed; m.histogram[bins.findIndex(upper => elapsed <= upper)]++;
  m.depth[r.depth] = (m.depth[r.depth] ?? 0) + 1; m.nodes += r.nodes; m.cacheHits += r.strongStats?.cacheHits ?? 0;
  if (!r.complete && r.strongStats?.stopReason !== '自适应提前停止') m.cutoffs++;
  if (r.strongStats?.stopReason === '自适应提前停止') m.adaptiveStops++;
  m.prunedEdges += r.strongStats?.prunedEdges ?? 0;
}
function combine(m: Metrics, n: Metrics) {
  for (const k of ['decisions', 'elapsed', 'nodes', 'cacheHits', 'cutoffs', 'adaptiveStops', 'prunedEdges'] as const) m[k] += n[k];
  n.histogram.forEach((v, i) => m.histogram[i] += v);
  for (const [k, v] of Object.entries(n.depth)) m.depth[k] = (m.depth[k] ?? 0) + v;
}
function describe(m: Metrics) {
  const percentile = (p: number) => { let count = 0; for (let i = 0; i < bins.length; i++) { count += m.histogram[i]; if (count >= Math.ceil(m.decisions * p)) return Number.isFinite(bins[i]) ? bins[i] : '>5000'; } return null; };
  return { ...m, latencyMs: { average: m.decisions ? m.elapsed / m.decisions : null, p50Upper: m.decisions ? percentile(.5) : null, p95Upper: m.decisions ? percentile(.95) : null }, cutoffRate: m.decisions ? m.cutoffs / m.decisions : null };
}
type Status = 'terminal' | 'success' | 'censored' | 'error';
interface Run { case: number; analysisSeed: number; variant: 'baseline' | 'candidate'; status: Status; reason: string; error?: string; maximum: number; score: number; addedScore: number; movesAdded: number; final: Snapshot; metrics: Metrics; elapsedMs: number }
const records: Run[] = [], totals = { baseline: metrics(), candidate: metrics() };
const goals = [...new Set([2048, 4096, 8192, 16384, 32768, 65536, target])].sort((a, b) => a - b);
let interrupted = false;
process.on('SIGINT', () => { interrupted = true; });
async function play(index: number, variant: Run['variant'], initial: Snapshot): Promise<Run> {
  if (variant === 'candidate' && b === 'strong') resetStrongCache();
  const dir = join(out, `${index}-${variant}`); mkdirSync(dir);
  let state = cloneSnapshot(initial), status: Status = 'censored', reason = 'move-cap', error: string | undefined;
  const o = { ...options, algorithm: variant === 'baseline' ? a : b, seed: analysisSeeds[index] };
  const policy = variant === 'baseline' ? baseline.solve : solve, session = variant === 'baseline' ? baselineSession : { advance, terminal };
  const m = metrics(), tail: Snapshot[] = [cloneSnapshot(state)], began = performance.now();
  json(join(dir, 'start-save.json'), JSON.parse(encode(state)));
  while (true) {
    if (session.terminal(state.board)) { status = 'terminal'; reason = 'no-legal-move'; break; }
    if (stopAtTarget && maxTile(state.board) >= target) { status = 'success'; reason = 'target-reached'; break; }
    if (interrupted || state.moves - initial.moves >= maxMoves) { reason = interrupted ? 'interrupted' : 'move-cap'; break; }
    try {
      const started = performance.now(), result = policy(state.board, o), elapsed = performance.now() - started;
      record(m, result, elapsed);
      if (result.direction === null) {
        if (o.algorithm === 'exact' && !result.complete) { reason = 'incomplete-exact-search'; break; }
        throw new Error('Solver returned no action on a nonterminal board');
      }
      const next = session.advance(state, result.direction, false);
      if (!next.transition.moved) throw new Error('Solver returned an invalid action');
      appendFileSync(join(dir, 'trace.jsonl'), JSON.stringify({ step: state.moves, direction: result.direction, spawned: next.spawned, rngBefore: state.rngState, rngAfter: next.snapshot.rngState, scoreDelta: next.transition.scoreDelta,
        depth: result.depth, nodes: result.nodes, complete: result.complete, elapsedMs: elapsed, value: result.choices.find(c => c.direction === result.direction)?.value, strong: result.strongStats, boardAfter: next.snapshot.board }) + '\n');
      state = next.snapshot; tail.push(cloneSnapshot(state)); if (tail.length > tailLength) tail.shift();
      if (m.decisions % 64 === 0) json(join(dir, 'latest-save.json'), JSON.parse(encode(state)));
    } catch (e) { status = 'error'; reason = 'solver-error'; error = e instanceof Error ? e.message : String(e); break; }
    await new Promise<void>(resolve => setImmediate(resolve));
  }
  const run: Run = { case: index, analysisSeed: o.seed, variant, status, reason, error, maximum: maxTile(state.board), score: state.score, addedScore: state.score - initial.score, movesAdded: state.moves - initial.moves, final: state, metrics: m, elapsedMs: performance.now() - began };
  json(join(dir, 'final-save.json'), JSON.parse(encode(state))); json(join(dir, 'tail.json'), tail.map(s => JSON.parse(encode(s))));
  json(join(dir, 'result.json'), { ...run, metrics: describe(m), finalProfile: strongProfile(state.board, o) });
  combine(totals[variant], m); return run;
}
function summary() {
  const rate = (rows: Run[], goal: number) => {
    const success = rows.filter(r => r.maximum >= goal).length, failed = rows.filter(r => r.status === 'terminal' && r.maximum < goal).length, unknown = rows.length - success - failed;
    return { target: goal, observedSuccesses: success, confirmedFailures: failed, unknown, rate: rows.length && !unknown ? success / rows.length : null,
      bounds: rows.length ? [success / rows.length, (success + unknown) / rows.length] : null,
      confidence95: rows.length && !unknown && !saves.length ? wilson(success, rows.length) : null };
  };
  const variants = Object.fromEntries((['baseline', 'candidate'] as const).map(variant => {
    const rows = records.filter(r => r.variant === variant), n = rows.length;
    const scores = (subset: Run[]) => {
      const values = subset.map(r => r.score).sort((x, y) => x - y), count = values.length;
      return { count, average: count ? values.reduce((x, y) => x + y, 0) / count : null, median: count ? count % 2 ? values[(count - 1) / 2] : (values[count / 2 - 1] + values[count / 2]) / 2 : null };
    };
    return [variant, { cases: n, notStarted: games - n, statuses: Object.fromEntries(['terminal', 'success', 'censored', 'error'].map(status => [status, rows.filter(r => r.status === status).length])),
      score: { terminal: scores(rows.filter(r => r.status === 'terminal')), stoppedAtTarget: scores(rows.filter(r => r.status === 'success')), partial: scores(rows.filter(r => r.status === 'censored' || r.status === 'error')) }, goals: goals.map(goal => rate(rows, goal)), metrics: describe(totals[variant]) }];
  }));
  const paired = goals.map(goal => {
    let both = 0, neither = 0, baselineOnly = 0, candidateOnly = 0, unresolved = 0, unpaired = 0;
    for (const index of new Set(records.map(r => r.case))) {
      const ar = records.find(r => r.case === index && r.variant === 'baseline'), br = records.find(r => r.case === index && r.variant === 'candidate');
      if (!ar || !br) { unpaired++; continue; }
      if ((ar.maximum < goal && ar.status !== 'terminal') || (br.maximum < goal && br.status !== 'terminal')) { unresolved++; continue; }
      const av = ar.maximum >= goal, bv = br.maximum >= goal;
      if (av && bv) both++; else if (!av && !bv) neither++; else if (av) baselineOnly++; else candidateOnly++;
    }
    const resolved = both + neither + baselineOnly + candidateOnly;
    return { target: goal, both, neither, baselineOnly, candidateOnly, unresolved, unpaired, resolvedPairs: resolved, deltaOnResolvedPairs: resolved ? (candidateOnly - baselineOnly) / resolved : null };
  });
  json(join(out, 'summary.json'), { metadata: 'metadata.json', interrupted, requestedCases: games, variants, paired, records: records.map(r => ({ ...r, metrics: describe(r.metrics) })) });
}
console.log(`A/B output: ${out}\nBaseline ${BASELINE.slice(0, 7)} ${a}; candidate ${b}; ${mode} budget; ${games} cases`);
for (let index = 0; index < games && !interrupted; index++) {
  const rng = new RNG(gameSeeds[index]);
  const initial = snapshots[index] ? { ...cloneSnapshot(snapshots[index]), target } : { board: newBoard(size, rng), score: 0, rngState: rng.state, moves: 0, target };
  for (const variant of index % 2 ? ['candidate', 'baseline'] as const : ['baseline', 'candidate'] as const) {
    if (interrupted) break;
    const run = await play(index, variant, initial); records.push(run); summary();
    console.log(`${index + 1}/${games} ${variant}: ${run.status}, max=${run.maximum}, score=${run.score}, addedMoves=${run.movesAdded}${run.error ? `, ${run.error}` : ''}`);
  }
}
summary();
console.log(`Reports and replay traces saved to ${out}`);
