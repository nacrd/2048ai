import { planTas } from './tas';
self.onmessage = event => {
  const { id, snapshot, options } = event.data;
  try { self.postMessage({ id, result: planTas(snapshot, options, progress => self.postMessage({ id, progress })) }); }
  catch (e) { self.postMessage({ id, error: (e as Error).message }); }
};
