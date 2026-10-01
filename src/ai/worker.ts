import { rolloutSlice, solve } from './solver';
import { strongFallback, strongRoot } from './strong';
import { probeAuto } from './auto-params';
self.onmessage = (event) => {
  const { id, board, options, batch, strong, autoProbe } = event.data;
  try {
    if (autoProbe) self.postMessage({ id, result: probeAuto(board, options) });
    else if (strong) {
      const result = strong.prepare ? strongFallback(board, options) : strongRoot(board, options, strong.direction, strong.depth, performance.now() + Math.max(0, strong.deadline - Date.now()));
      self.postMessage({ id, result });
    } else if (batch) {
      const result = rolloutSlice(board, options, batch.offset, batch.count, batch.includeTies);
      self.postMessage({ id, result }, { transfer: [result.values.buffer] });
    } else self.postMessage({ id, result: solve(board, options) });
  }
  catch (e) { self.postMessage({ id, error: e instanceof Error ? e.message : String(e) }); }
};
