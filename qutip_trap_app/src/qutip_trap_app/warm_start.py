"""The warm start a packaged build ships: what a worker derives for the circuit and the device the app opens on.

A fresh worker derives the calibration table and the channel library of a device on the first Run that needs them
(seconds natively, half a minute in a browser, where nothing runs in parallel), and the stability boundary of the trap
page on the first physics page. ``python -m qutip_trap_app.warm_start`` answers those two requests here, as the app
submits them before anything is edited, and writes what the worker then holds beside the package, where
``workers.initial_state`` reads it in a packaged build. A checkout never reads it.
"""

from __future__ import annotations

import pickle

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.views.state import Session, Store
from qutip_trap_app.workers import WARM_START_PATH, WarmStart


def derive() -> WarmStart:
    """Answer the start-up Run, by the engine the app opens on, and the start-up device's layer as a worker that starts
    with nothing would."""
    store = Store()
    job = Session(store, ProvenanceIndex.load()).build_job()
    return WarmStart.after((store.engine.request, {"job": job}), ("derive", {"device": store.device_ref()}))


def main() -> None:
    """Write the warm start a packaged build ships."""
    warm = derive()
    WARM_START_PATH.write_bytes(pickle.dumps(warm))
    entries = sorted(key for library in warm.libraries.values() for key in library.entries)
    print(
        f"wrote {WARM_START_PATH.name}: {WARM_START_PATH.stat().st_size} bytes, "
        f"{len(warm.presets)} preset(s), {len(warm.calibration)} calibration report(s), channels {entries}, "
        f"the stability boundary from {0 if warm.stability is None else warm.stability.monodromy_calls} monodromies"
    )


if __name__ == "__main__":
    main()
