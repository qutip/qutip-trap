"""The warm start a packaged build ships: a worker that starts from it answers the start-up Run with the record a worker
that starts with nothing computes, and derives neither its calibration, its channels nor the stability boundary."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from qutip_trap_app import core, workers
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.views.state import Session, Store
from qutip_trap_app.warm_start import derive
from qutip_trap_app.workers import HANDLERS, initial_state


def test_a_checkout_worker_starts_with_nothing() -> None:
    state = initial_state()
    assert not (state.presets or state.tables or state.libraries or state.records or state.lives)
    assert state.stability is None


def test_a_packaged_worker_answers_the_start_up_job_without_deriving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store()
    job = Session(store, ProvenanceIndex.load()).build_job()
    request = store.engine.request
    monkeypatch.setattr(core.CALIBRATION_CACHE, "reports", {})
    cold_stages: list[str] = []
    cold = HANDLERS[request](
        initial_state(), lambda stage, fraction, message: cold_stages.append(stage), job=job
    )
    assert "channels" in cold_stages, "a worker that starts with nothing derives the channel library"

    # a packaged build: no PLAN.md beside the core, the warm start beside the app, nothing calibrated in the process yet
    monkeypatch.setattr(core.CALIBRATION_CACHE, "reports", {})
    asset = tmp_path / "warm_start.pkl"
    asset.write_bytes(pickle.dumps(derive()))
    monkeypatch.setattr(core.CALIBRATION_CACHE, "reports", {})
    monkeypatch.setattr(workers, "WARM_START_PATH", asset)
    monkeypatch.setattr(core, "repository_root", lambda: tmp_path)
    state = initial_state()
    calibrated = set(core.CALIBRATION_CACHE.reports)
    channels = {key: set(library.entries) for key, library in state.libraries.items()}
    assert calibrated and channels and state.presets

    stages: list[str] = []
    record = HANDLERS[request](state, lambda stage, fraction, message: stages.append(stage), job=job)
    assert "channels" not in stages, "no channel was derived"
    assert set(core.CALIBRATION_CACHE.reports) == calibrated, "no calibration was derived"
    assert {key: set(library.entries) for key, library in state.libraries.items()} == channels
    assert record.key() == cold.key(), "the run itself is the one a cold worker computes"

    boundary = state.stability
    messages: list[str] = []
    layer = HANDLERS["derive"](
        state, lambda stage, fraction, message: messages.append(message), device=store.device_ref()
    )
    assert boundary is not None and layer.trap.stability is boundary and state.stability is boundary
    assert not any("stability boundary, once" in m for m in messages), "the boundary was not computed again"
