import { planTas } from './tas';
self.onmessage = event => {
  const { id, snapshot, options } = event.data;
  try { self.postMessage({ id, result: planTas(snapshot, options) }); }
  catch (e) { self.postMessage({ id, error: (e as Error).message }); }
};
