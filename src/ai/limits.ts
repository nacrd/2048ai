import { RNG } from '../core/engine';
import type { Snapshot } from '../core/session';

export interface FixedLimit { requestedTarget: number; upperTile: number; barrierMass: number; movesFromStart: number; requiredCells: number; reason: string }
function popcount32(value: number) {
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return Math.imul((value + (value >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
}
function popcount(value: number) { return popcount32(value >>> 0) + popcount32(Math.floor(value / 4294967296)); }
// Minimum tile count equals the binary popcount of the conserved board mass.
// At capacity, the unique representation is a full board of distinct powers: no legal move.
export function fixedLimit(start: Snapshot, target: number, maxMs = 100): FixedLimit | null {
  const capacity = start.board.cells.length, rng = new RNG(start.rngState), began = performance.now();
  let mass = start.board.cells.reduce((sum, e) => sum + (e ? 2 ** e : 0), 0), steps = 0;
  if (!Number.isSafeInteger(mass) || mass >= target) return null;
  while (mass < target) {
    if ((steps & 255) === 0 && performance.now() - began >= maxMs) return null;
    const count = popcount(mass);
    if (count === capacity) {
      const upperTile = 2 ** Math.floor(Math.log2(mass));
      return { requestedTarget: target, upperTile, barrierMass: mass, movesFromStart: steps, requiredCells: count, reason: `固定 RNG 在 ${steps} 步后遇到总质量 ${mass}，最少 ${count} 个互不相同的块占满棋盘；无法继续到 ${target}。最大块上界 ${upperTile}，尚未证明该上界可达` };
    }
    mass += rng.next() < 0.9 ? 2 : 4; rng.next(); steps++;
  }
  return null;
}
