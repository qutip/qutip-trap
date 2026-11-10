// The simulation worker of the browser build: a second Python runtime beside the one that draws the page, so that a
// solver never holds the thread that answers clicks. qutip_trap_app/workers.py starts this script (WebWorkerTransport)
// and answers its requests (serve_web_worker).
//
// The first message is {pyodideUrl, appPackageUrl}, the URLs the page's own runtime booted from. Every later message is
// one pickled request, which waits for the boot and runs in arrival order; Python posts its events back as pickles. A
// string posted to the page is a failure: the boot's, or an exception that escaped a request.

async function boot({ pyodideUrl, appPackageUrl }) {
  const { loadPyodide } = await import(pyodideUrl);
  const pyodide = await loadPyodide();
  const response = await fetch(appPackageUrl);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${appPackageUrl}`);
  }
  pyodide.unpackArchive(await response.arrayBuffer(), "zip");
  // the package's dependencies sit in __pypackages__, where Flet's own worker script finds them
  return pyodide.runPython(`
import sys
sys.path.insert(0, "__pypackages__")
from qutip_trap_app.workers import serve_web_worker
serve_web_worker()
`);
}

// Python's request handler, as a promise that every request extends, so that requests run one after another
let python = null;

self.onmessage = ({ data }) => {
  if (python === null) {
    python = boot(data);
  } else {
    python = python.then((answer) => {
      answer(data);
      return answer;
    });
  }
  python.catch((error) => self.postMessage(`the simulation worker failed: ${error}`));
};
