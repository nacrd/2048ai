import { applyMove, fromMatrix, legalMoves, matrix, RNG, spawnTile, type Board, type Direction, type MoveResult } from './engine';
export interface Snapshot { board: Board; score: number; rngState: number; target: number; moves: number }
export function cloneSnapshot(s: Snapshot): Snapshot { return { ...s, board: { size: s.board.size, cells: [...s.board.cells] } }; }
export function advance(s: Snapshot, direction: Direction): { snapshot: Snapshot; transition: MoveResult; spawned: number | null } {
  const transition = applyMove(s.board, direction, true);
  if (!transition.moved) return { snapshot: s, transition, spawned: null };
  const rng = new RNG(s.rngState);
  const spawned = spawnTile(transition.board, rng);
  return { snapshot: { ...s, board: spawned.board, score: s.score + transition.scoreDelta, rngState: rng.state, moves: s.moves + 1 }, transition, spawned: spawned.index };
}
export function encode(s: Snapshot, history?: Snapshot[]): string {
  const pack = (v: Snapshot) => ({ ...v, board: matrix(v.board) });
  return JSON.stringify({ version: 1, rules: 'classic-90-10', ...pack(s), ...(history ? { history: history.map(pack) } : {}) }, null, 2);
}
export function decode(text: string): { snapshot: Snapshot; history: Snapshot[] } {
  const value = JSON.parse(text);
  if (Array.isArray(value)) return { snapshot: { board: fromMatrix(value), score: 0, rngState: 12345, target: 2048, moves: 0 }, history: [] };
  if (!value || value.version !== 1 || value.rules !== 'classic-90-10') throw new Error('不支持的存档版本或出块规则');
  const unpack = (v: Record<string, unknown>): Snapshot => {
    if (!v || typeof v !== 'object') throw new Error('存档状态格式错误');
    const { score, rngState, target, moves } = v;
    if (!Number.isSafeInteger(score) || (score as number) < 0 || !Number.isSafeInteger(moves) || (moves as number) < 0) throw new Error('分数与步数必须为非负整数');
    if (!Number.isInteger(rngState) || (rngState as number) < 1 || (rngState as number) > 4294967295) throw new Error('随机状态不合法');
    if (typeof target !== 'number' || target < 2 || target > 2 ** 30 || !Number.isInteger(Math.log2(target))) throw new Error('目标块不合法');
    return { board: fromMatrix(v.board, 52), score: score as number, rngState: rngState as number, target, moves: moves as number };
  };
  const snapshot = unpack(value);
  if (value.history !== undefined && (!Array.isArray(value.history) || value.history.length > 100000)) throw new Error('回放历史不合法或过长');
  const history = (value.history ?? []).map(unpack);
  if (history.some((s: Snapshot) => s.board.size !== snapshot.board.size || s.target !== snapshot.target)) throw new Error('回放历史的尺寸和目标必须一致');
  return { snapshot, history };
}
export function terminal(board: Board): boolean { return legalMoves(board).length === 0; }
