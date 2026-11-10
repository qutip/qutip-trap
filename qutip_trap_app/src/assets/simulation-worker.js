// The simulation worker of the browser build: a second Python runtime beside the one that draws the page, so that a
// solver never holds the thread that answers clicks. qutip_trap_app/workers.py starts it (WebWorkerTransport) and
// answers inside it (serve_web_worker).
//
// The first message is {pyodideUrl, appPackageUrl}, the two URLs the page's own runtime was booted from. Every later
// message is one pickled request, a Uint8Array handed to Python; requests wait for the boot and run in arrival order.
// Python posts its events back as pickles. A string posted to the page is a failure: the boot's, or an exception that
// escaped a request.

async function boot({ pyodideUrl, appPackageUrl }) {
  const { loadPyodide } = await import(pyodideUrl);
  const pyodide = await loadPyodide();
  const response = await fetch(appPackageUrl);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${appPackageUrl}`);
  }
  pyodide.unpackArchive(await response.arrayBuffer(), "zip");
  return pyodide.runPython(`
import sys

sys.path.insert(0, "__pypackages__")

from qutip_trap_app.workers import serve_web_worker

serve_web_worker()
`);
}

let handler = null;

self.onmessage = (event) => {
  handler =
    handler === null
      ? boot(event.data)
      : handler.then((handle) => {
          handle(event.data);
          return handle;
        });
  handler.catch((error) => self.postMessage(`the simulation worker failed: ${error}`));
};
