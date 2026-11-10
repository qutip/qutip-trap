"""The warm start a packaged build ships: a worker that starts from it answers the start-up Run with the record a worker
that starts with nothing computes, and derives neither its calibration, its channels nor the stability boundary."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from qutip_trap_app import core, device_layer, provenance, workers
from qutip_trap_app.views.state import Session, Store
from qutip_trap_app.warm_start import derive
from qutip_trap_app.workers import HANDLERS, initial_state


def test_a_checkout_worker_starts_with_nothing() -> None:
    state = initial_state()
    assert not (state.presets or state.tables or state.libraries or state.records or state.lives)
    assert state.stability is None


def test_a_packaged_worker_answers_the_start_up_requests_without_deriving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def new_process() -> None:
        """Nothing calibrated yet, as in a process that has just started."""
        monkeypatch.setattr(core.CALIBRATION_CACHE, "reports", {})

    store = Store()
    job = Session(store, provenance.ProvenanceIndex.load()).build_job()
    run = HANDLERS[store.engine.request]
    new_process()
    cold_stages: list[str] = []
    cold = run(initial_state(), lambda stage, fraction, message: cold_stages.append(stage), job=job)
    assert "channels" in cold_stages, "a worker that starts with nothing derives the channel library"

    new_process()
    asset = tmp_path / "warm_start.pkl"
    asset.write_bytes(pickle.dumps(derive()))

    # a packaged build: no PLAN.md beside the core, the warm start beside the app
    new_process()
    monkeypatch.setattr(provenance, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(workers, "WARM_START_PATH", asset)
    monkeypatch.setattr(device_layer, "stability_map", lambda: pytest.fail("the boundary was derived again"))
    state = initial_state()
    calibrated = set(core.CALIBRATION_CACHE.reports)
    assert calibrated and state.presets and state.libraries and state.stability is not None

    stages: list[str] = []
    record = run(state, lambda stage, fraction, message: stages.append(stage), job=job)
    assert "channels" not in stages, "no channel was derived"
    assert set(core.CALIBRATION_CACHE.reports) == calibrated, "no calibration was derived"
    assert record.key() == cold.key(), "the run is the one a worker that starts with nothing computes"
    layer = HANDLERS["derive"](state, lambda stage, fraction, message: None, device=store.device_ref())
    assert layer.trap.stability is state.stability
