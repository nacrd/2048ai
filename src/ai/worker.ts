import { solve } from './solver';
self.onmessage = (event) => {
  const { id, board, options } = event.data;
  try { self.postMessage({ id, result: solve(board, options) }); }
  catch (e) { self.postMessage({ id, error: e instanceof Error ? e.message : String(e) }); }
};
