import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { applyMove, DIRECTIONS, emptyCells, fromMatrix, hashSeed, legalMoves, matrix, newBoard, reached, RNG, type Board } from '../src/core/engine';
import { advance, decode, encode, type Snapshot } from '../src/core/session';
import { DEFAULT_OPTIONS, rolloutOne, solve, type Objective } from '../src/ai/solver';

function rows(row: number[]) { return [row, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]; }
describe('pure engine', () => {
  it.each([
    [[2, 2, 2, 2], [4, 4, 0, 0], 8], [[2, 2, 4, 0], [4, 4, 0, 0], 4],
    [[4, 4, 4, 4], [8, 8, 0, 0], 16], [[2, 0, 2, 2], [4, 2, 0, 0], 4],
    [[0, 2, 0, 4], [2, 4, 0, 0], 0], [[2 ** 30, 2 ** 30, 0, 0], [2 ** 31, 0, 0, 0], 2 ** 31],
  ])('merges %j only once', (source, wanted, score) => {
    const b = fromMatrix(rows(source as number[])), before = JSON.stringify(b);
    const m = applyMove(b, 3, true);
    expect(matrix(m.board)[0]).toEqual(wanted); expect(m.scoreDelta).toBe(score); expect(JSON.stringify(b)).toBe(before);
  });
  it('matches the unmodified upstream on 8000 directions', () => {
    const context = vm.createContext({});
    for (const file of ['tile', 'grid', 'game_manager']) vm.runInContext(readFileSync(`js/${file}.js`, 'utf8'), context);
    const upstream = context as any;
    const rng = new RNG(7937);
    for (let trial = 0; trial < 2000; trial++) {
      const cells = Array.from({ length: 16 }, () => rng.next() < .28 ? 0 : Math.floor(rng.next() * 12) + 1);
      const b = { size: 4, cells };
      for (const d of DIRECTIONS) {
        const manager = Object.create(upstream.GameManager.prototype);
        Object.assign(manager, { size: 4, grid: new upstream.Grid(4), score: 0, won: false, over: false, keepPlaying: true, addRandomTile() {}, actuate() {} });
        cells.forEach((v, i) => { if (v) manager.grid.insertTile(new upstream.Tile({ x: i % 4, y: Math.floor(i / 4) }, 2 ** v)); });
        manager.move(d);
        const expected = cells.map((_, i) => manager.grid.cells[i % 4][Math.floor(i / 4)]?.value ?? 0);
        const actual = applyMove(b, d);
        expect(matrix(actual.board).flat()).toEqual(expected); expect(actual.scoreDelta).toBe(manager.score);
      }
    }
  });
  it('preserves tile mass for every supported size', () => {
    const rng = new RNG(9999);
    for (let n = 2; n <= 6; n++) for (let j = 0; j < 100; j++) {
      const b = { size: n, cells: Array.from({ length: n * n }, () => Math.floor(rng.next() * 10)) };
      const mass = b.cells.reduce((s, e) => s + (e ? 2 ** e : 0), 0);
      for (const d of DIRECTIONS) expect(applyMove(b, d).board.cells.reduce((s, e) => s + (e ? 2 ** e : 0), 0)).toBe(mass);
    }
  });
  it('validates inputs and detects dead/custom achieved boards', () => {
    for (const bad of [[[1, 0], [0, 0]], [[3, 0], [0, 0]], [[2, 0, 0], [0, 0]], [[Infinity, 0], [0, 0]]]) expect(() => fromMatrix(bad)).toThrow();
    expect(legalMoves(fromMatrix([[2, 4], [8, 16]]))).toEqual([]);
    expect(reached(fromMatrix([[2048, 0], [0, 0]]), 2048)).toBe(true);
  });
});

describe('reproducible sessions', () => {
  function initial(): Snapshot { const rng = new RNG(hashSeed('fixture')); return { board: newBoard(4, rng), score: 0, rngState: rng.state, target: 2048, moves: 0 }; }
  it('does not advance RNG on invalid moves, analysis, or serialization', () => {
    const s: Snapshot = { board: fromMatrix([[2, 4], [8, 16]]), score: 0, rngState: 77, target: 2048, moves: 0 };
    expect(advance(s, 3).snapshot).toBe(s);
    const live = initial(), before = encode(live);
    solve(live.board, { ...DEFAULT_OPTIONS, horizon: 1 });
    expect(encode(live)).toBe(before);
    expect(decode(before).snapshot).toEqual(live);
  });
  it('undo/restore repeats the same spawned tile', () => {
    const s = initial(), d = legalMoves(s.board)[0];
    const next = advance(s, d).snapshot;
    const restored = decode(encode(s)).snapshot;
    expect(advance(restored, d).snapshot).toEqual(next);
    expect(decode(encode(next, [s])).history).toEqual([s]);
  });
  it('restores large tiles produced by legal merges', () => {
    const s = { ...initial(), board: fromMatrix(rows([2 ** 30, 2 ** 30, 0, 0])) };
    const next = advance(s, 3).snapshot;
    expect(decode(encode(next)).snapshot).toEqual(next);
  });
});

function brute(b: Board, h: number, objective: Objective, target: number): number {
  if (objective === 'target' && reached(b, target)) return 1;
  if (!h) return 0;
  const values = legalMoves(b).map(d => {
    const m = applyMove(b, d), empties = emptyCells(m.board);
    const reward = objective === 'score' ? m.scoreDelta : 0;
    return reward + empties.reduce((sum, i) => sum + [1, 2].reduce((a, e) => {
      const cells = [...m.board.cells]; cells[i] = e;
      return a + (e === 1 ? .9 : .1) * brute({ size: b.size, cells }, h - 1, objective, target) / empties.length;
    }, 0), 0);
  });
  return values.length ? Math.max(...values) : 0;
}
describe('search contract', () => {
  it.each(['score', 'target'] as const)('agrees with independent finite-horizon enumeration (%s)', objective => {
    const b = fromMatrix([[2, 2], [4, 0]]), result = solve(b, { ...DEFAULT_OPTIONS, algorithm: 'exact', objective, target: 8, horizon: 3, budgetMs: 10000 });
    expect(result.complete).toBe(true); expect(Math.max(...result.choices.map(c => c.value))).toBeCloseTo(brute(b, 3, objective, 8), 9);
  });
  it('never calls an interrupted exact search optimal', () => {
    const result = solve(fromMatrix([[2, 2], [0, 0]]), { ...DEFAULT_OPTIONS, algorithm: 'exact', horizon: 6, maxNodes: 1 });
    expect(result.complete).toBe(false); expect(result.direction).toBe(null); expect(result.choices).toEqual([]);
  });
  it('returns a legal approximate fallback and a dead-board null action', () => {
    const b = fromMatrix([[2, 2], [0, 0]]);
    expect(legalMoves(b)).toContain(solve(b, { ...DEFAULT_OPTIONS, maxNodes: 1 }).direction);
    expect(solve(fromMatrix([[2, 4], [8, 16]]), DEFAULT_OPTIONS).direction).toBe(null);
  });
  it('uses independent reproducible trajectories and bounded intervals', () => {
    const b = fromMatrix([[2, 2], [0, 0]]), o = { ...DEFAULT_OPTIONS, algorithm: 'rollout' as const, objective: 'target' as const, target: 8, horizon: 4, trajectories: 8, budgetMs: 10000 };
    const a = solve(b, o), c = solve(b, o); expect(a.choices).toEqual(c.choices);
    for (const choice of a.choices) { expect(choice.confidence![0]).toBeGreaterThanOrEqual(0); expect(choice.confidence![1]).toBeLessThanOrEqual(1); }
    expect(rolloutOne(b, 3, 0, o)).toBe(rolloutOne(b, 3, 0, o));
  });
});
