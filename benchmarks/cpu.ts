import { mkdirSync, writeFileSync } from 'node:fs';
import { hashSeed, maxTile, newBoard, RNG } from '../src/core/engine';
import { advance, type Snapshot } from '../src/core/session';
import { DEFAULT_OPTIONS, solve, wilson } from '../src/ai/solver';

const games = Number(process.argv[2] || 100);
const maxMoves = 5000;
// Fixed node budgets make the quality baseline reproducible across host speeds.
const config = { ...DEFAULT_OPTIONS, horizon: 3, maxNodes: 10000, budgetMs: 10000 };
const records: { seed: string; score: number; maxTile: number; moves: number; censored: boolean }[] = [], times: number[] = [];
for (let game = 0; game < games; game++) {
  const seed = `validation-${game}`;
  const rng = new RNG(hashSeed(seed));
  let state: Snapshot = { board: newBoard(4, rng), score: 0, rngState: rng.state, target: 2048, moves: 0 };
  for (; state.moves < maxMoves;) {
    const result = solve(state.board, config); times.push(result.elapsedMs);
    if (result.direction === null) break;
    state = advance(state, result.direction).snapshot;
  }
  records.push({ seed, score: state.score, maxTile: maxTile(state.board), moves: state.moves, censored: state.moves === maxMoves });
  if ((game + 1) % 10 === 0) console.log(`Finished ${game + 1}/${games} games`);
}
const sorted = records.map(r => r.score).sort((a, b) => a - b); times.sort((a, b) => a - b);
const successes = (target: number) => records.filter(r => r.maxTile >= target).length;
const result = { date: '2026-10-01', config, games, maxMoves, averageScore: sorted.reduce((a, b) => a + b, 0) / games, medianScore: sorted[Math.floor(sorted.length / 2)],
  targetRates: [2048, 4096, 8192].map(target => ({ target, successes: successes(target), rate: successes(target) / games, confidence95: wilson(successes(target), games) })),
  maxTileDistribution: Object.fromEntries([...new Set(records.map(r => r.maxTile))].sort((a, b) => a - b).map(t => [t, records.filter(r => r.maxTile === t).length])),
  latencyMs: { p50: times[Math.floor(times.length * .5)], p95: times[Math.floor(times.length * .95)] }, records };
mkdirSync('benchmarks', { recursive: true }); writeFileSync('benchmarks/cpu-results.json', JSON.stringify(result, null, 2));
console.log(JSON.stringify({ ...result, records: undefined }, null, 2));
