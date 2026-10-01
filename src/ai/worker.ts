import { rolloutSlice, solve } from './solver';
self.onmessage = (event) => {
  const { id, board, options, batch } = event.data;
  try {
    if (batch) {
      const result = rolloutSlice(board, options, batch.offset, batch.count, batch.includeTies);
      self.postMessage({ id, result }, { transfer: [result.values.buffer] });
    } else self.postMessage({ id, result: solve(board, options) });
  }
  catch (e) { self.postMessage({ id, error: e instanceof Error ? e.message : String(e) }); }
};
