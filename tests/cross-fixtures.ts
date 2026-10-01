import { mkdirSync, writeFileSync } from 'node:fs';
import { applyMove, DIRECTIONS, RNG } from '../src/core/engine';
import { DEFAULT_OPTIONS, rolloutOne } from '../src/ai/solver';
const random = new RNG(7831);
const fixtures = [];
for (let size = 2; size <= 6; size++) for (let trial = 0; trial < 12; trial++) {
  const cells = Array.from({ length: size * size }, () => random.next() < .3 ? 0 : Math.floor(random.next() * 12) + 1);
  const board = { size, cells };
  fixtures.push({ board, moves: DIRECTIONS.map(d => { const m = applyMove(board, d); return { cells: m.board.cells, moved: m.moved, score: m.scoreDelta }; }),
    rollouts: DIRECTIONS.map(d => Array.from({ length: 8 }, (_, i) => rolloutOne(board, d, i, { ...DEFAULT_OPTIONS, horizon: 16, seed: 71231 }))) });
}
mkdirSync('tests/fixtures', { recursive: true });
writeFileSync('tests/fixtures/cross.json', JSON.stringify(fixtures));
