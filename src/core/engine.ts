export type Direction = 0 | 1 | 2 | 3;
export const DIRECTIONS: Direction[] = [0, 1, 2, 3];
export const ARROWS = ['↑', '→', '↓', '←'];
export interface Board { size: number; cells: number[] }
export interface Motion { from: number[]; to: number; exponent: number }
export interface MoveResult { board: Board; moved: boolean; scoreDelta: number; motions: Motion[] }

const indexCache = new Map<string, number[][]>();
export function lineIndices(size: number, direction: Direction): number[][] {
  const key = `${size}:${direction}`;
  let lines = indexCache.get(key);
  if (!lines) {
    lines = Array.from({ length: size }, (_, line) => Array.from({ length: size }, (_, i) => {
      if (direction === 0) return i * size + line;
      if (direction === 1) return line * size + size - 1 - i;
      if (direction === 2) return (size - 1 - i) * size + line;
      return line * size + i;
    }));
    indexCache.set(key, lines);
  }
  return lines;
}

export function applyMove(board: Board, direction: Direction, animate = false): MoveResult {
  const cells = new Array<number>(board.cells.length).fill(0);
  const motions: Motion[] = [];
  let scoreDelta = 0;
  for (const indices of lineIndices(board.size, direction)) {
    const occupied = indices.filter(i => board.cells[i] !== 0);
    let out = 0;
    for (let k = 0; k < occupied.length; k++) {
      const from = occupied[k];
      let exponent = board.cells[from];
      const sources = [from];
      if (k + 1 < occupied.length && board.cells[occupied[k + 1]] === exponent) {
        sources.push(occupied[++k]);
        exponent++;
        scoreDelta += 2 ** exponent;
      }
      const to = indices[out++];
      cells[to] = exponent;
      if (animate) motions.push({ from: sources, to, exponent });
    }
  }
  return { board: { size: board.size, cells }, moved: cells.some((v, i) => v !== board.cells[i]), scoreDelta, motions };
}

export function legalMoves(board: Board): Direction[] { return DIRECTIONS.filter(d => applyMove(board, d).moved); }
export function emptyCells(board: Board): number[] { return board.cells.flatMap((v, i) => v === 0 ? [i] : []); }
export function reached(board: Board, target: number): boolean { return board.cells.some(v => v >= Math.log2(target)); }
export function maxTile(board: Board): number { const e = Math.max(...board.cells); return e ? 2 ** e : 0; }
export function matrix(board: Board): number[][] {
  return Array.from({ length: board.size }, (_, r) => board.cells.slice(r * board.size, (r + 1) * board.size).map(e => e ? 2 ** e : 0));
}
export function fromMatrix(value: unknown, maxExponent = 30): Board {
  if (!Array.isArray(value) || value.length < 2 || value.length > 6) throw new Error('棋盘尺寸必须为 2×2 到 6×6');
  const n = value.length;
  const cells: number[] = [];
  for (let r = 0; r < n; r++) {
    const row: unknown = value[r];
    if (!Array.isArray(row) || row.length !== n) throw new Error(`第 ${r + 1} 行必须有 ${n} 格`);
    row.forEach((v: unknown, c: number) => {
      if (typeof v !== 'number' || !Number.isSafeInteger(v) || v < 0 || v > 2 ** maxExponent || (v !== 0 && (v < 2 || (BigInt(v) & (BigInt(v) - 1n)) !== 0n))) {
        throw new Error(`第 ${r + 1} 行第 ${c + 1} 格：请输入 0 或 2 到 2^${maxExponent} 之间的 2 的幂`);
      }
      cells.push(v === 0 ? 0 : Math.log2(v));
    });
  }
  return { size: n, cells };
}

export function hashSeed(seed: string): number {
  let x = 2166136261;
  for (let i = 0; i < seed.length; i++) x = Math.imul(x ^ seed.charCodeAt(i), 16777619);
  return (x >>> 0) || 0x6d2b79f5;
}
export class RNG {
  state: number;
  constructor(state: number) { this.state = (state >>> 0) || 0x6d2b79f5; }
  nextUint(): number { let x = this.state; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return this.state = x >>> 0; }
  next(): number { return this.nextUint() / 4294967296; }
}
export function trajectorySeed(seed: number, direction: number, id: number): number {
  let x = (seed ^ Math.imul(direction + 1, 0x9e3779b9) ^ Math.imul(id + 1, 0x85ebca6b)) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15;
  return (x >>> 0) || 0x6d2b79f5;
}
export function spawnTile(board: Board, rng: RNG): { board: Board; index: number | null } {
  // Match upstream's column-first empty-cell order and value-then-position draws.
  const available: number[] = [];
  for (let x = 0; x < board.size; x++) for (let y = 0; y < board.size; y++) {
    const i = y * board.size + x;
    if (!board.cells[i]) available.push(i);
  }
  if (!available.length) return { board, index: null };
  const exponent = rng.next() < 0.9 ? 1 : 2;
  const index = available[Math.floor(rng.next() * available.length)];
  const cells = [...board.cells]; cells[index] = exponent;
  return { board: { size: board.size, cells }, index };
}
export function newBoard(size: number, rng: RNG): Board {
  const empty = { size, cells: new Array<number>(size * size).fill(0) };
  return spawnTile(spawnTile(empty, rng).board, rng).board;
}
