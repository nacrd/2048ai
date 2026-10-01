import type { Board, Direction, MoveResult } from './engine';

// Five bits per cell retain 32768, 65536 and 131072. Larger exponents use the generic engine.
let left: Uint32Array | undefined, right: Uint32Array, scores: Float64Array;
function reverse(row: number) { return ((row & 31) << 15) | ((row & 992) << 5) | ((row >>> 5) & 992) | ((row >>> 15) & 31); }
function init() {
  const count = 1 << 20;
  left = new Uint32Array(count); right = new Uint32Array(count); scores = new Float64Array(count);
  for (let row = 0; row < count; row++) {
    let pending = 0, slot = 0, output = 0, score = 0;
    for (let i = 0; i < 4; i++) {
      const value = (row >>> (i * 5)) & 31; if (!value) continue;
      if (!pending) pending = value;
      else if (pending === value) { output |= (value + 1) << (slot++ * 5); score += 2 ** (value + 1); pending = 0; }
      else { output |= pending << (slot++ * 5); pending = value; }
    }
    if (pending) output |= pending << (slot * 5);
    left[row] = output; scores[row] = score;
  }
  for (let row = 0; row < count; row++) right[row] = reverse(left[reverse(row)]);
}
export function fastMove4(board: Board, direction: Direction): MoveResult | null {
  if (board.size !== 4 || board.cells.some(v => v >= 31)) return null;
  if (!left) init();
  const cells = new Array<number>(16).fill(0), vertical = direction === 0 || direction === 2;
  const table = direction === 0 || direction === 3 ? left! : right;
  let scoreDelta = 0;
  for (let line = 0; line < 4; line++) {
    let row = 0;
    for (let i = 0; i < 4; i++) row |= board.cells[vertical ? i * 4 + line : line * 4 + i] << (i * 5);
    const out = table[row]; scoreDelta += scores[row];
    for (let i = 0; i < 4; i++) cells[vertical ? i * 4 + line : line * 4 + i] = (out >>> (i * 5)) & 31;
  }
  return { board: { size: 4, cells }, moved: cells.some((v, i) => v !== board.cells[i]), scoreDelta, motions: [] };
}
