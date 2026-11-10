"""The warm start a packaged build ships: what a worker derives for the circuit and the device the app opens on.

A worker that starts with nothing derives a device's calibration table and channel library on the first Run that needs
them (seconds natively, half a minute in a browser, where nothing runs in parallel) and the stability boundary on the
first physics page. ``python -m qutip_trap_app.warm_start`` answers those two requests here, as the app submits them
before anything is edited, and writes what the worker then holds beside the package. ``workers.initial_state`` reads it
in a packaged build and never in a checkout.
"""

from __future__ import annotations

import pickle

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.views.state import Session, Store
from qutip_trap_app.workers import WARM_START_PATH, WarmStart


def derive() -> WarmStart:
    """What a worker that starts with nothing holds after the start-up Run, by the engine the app opens on, and the
    start-up device's layer."""
    store = Store()
    job = Session(store, ProvenanceIndex.load()).build_job()
    return WarmStart.after((store.engine.request, {"job": job}), ("derive", {"device": store.device_ref()}))


def main() -> None:
    """Write the warm start a packaged build ships."""
    warm = derive()
    WARM_START_PATH.write_bytes(pickle.dumps(warm))
    channels = sorted(key for library in warm.state.libraries.values() for key in library.entries)
    print(
        f"wrote {WARM_START_PATH.name}: {WARM_START_PATH.stat().st_size} bytes, channels {', '.join(channels)}"
    )


if __name__ == "__main__":
    main()
