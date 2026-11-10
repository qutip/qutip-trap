"""The simulation worker: core calls off the UI's thread, progress and results streamed back (PLAN.md Section 14.6).

One long-lived worker holds the live state a record cannot carry (the device, the calibration table, the core's run record,
the channel library) keyed by the record's run key, so that a zoom or a verify-deeper request needs no re-run. Requests and
results are picklable values; progress events name a stage, a fraction where one is known and a message. A cancel terminates
the worker and restarts it: the caches live in the records the UI keeps, so only the live handles are lost.

Where the worker runs is its :class:`Transport`, chosen once where the application starts (:func:`platform_transport`): a
spawned process natively (not daemonic, so the core's own parallel maps may spawn inside it), and in the browser, which has
no processes, a second Python runtime in a Web Worker, or the page's own runtime for a device that cannot hold two.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import pickle
import queue
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from qutip_trap_app import core, resim
from qutip_trap_app import device_layer as layer_mod
from qutip_trap_app import presets as presets_mod
from qutip_trap_app.record import (
    DeviceRef,
    JobSpec,
    LiveRun,
    Progress,
    Record,
    TableRecord,
    calibrate_for,
    complete_job,
    execute,
    options_digest,
    table_record,
)
from qutip_trap_app.replay import ChannelLibrary, replay
from qutip_trap_app.verify import verify_deeper

EventKind = Literal["progress", "result", "error"]


@dataclass(frozen=True)
class Ticket:
    id: str
    kind: str


@dataclass(frozen=True)
class Event:
    kind: EventKind
    ticket: str
    request: str
    stage: str = ""
    fraction: float | None = None
    message: str = ""
    payload: Any = None
    wall_time_s: float = 0.0


class WorkerError(RuntimeError):
    """A request failed in the worker; the message carries the worker's traceback."""


# ---- the requests ---------------------------------------------------------------------------------------------------------------


@dataclass
class _LiveState:
    records: dict[str, Record] = field(default_factory=dict)
    lives: dict[str, LiveRun] = field(default_factory=dict)
    libraries: dict[str, ChannelLibrary] = field(default_factory=dict)
    """Channel libraries per (device hash, table seed, options digest)."""
    presets: dict[str, core.DevicePreset] = field(default_factory=dict)
    """Built presets per ``DeviceRef.cache_key()`` (a build with the recipe re-derived takes seconds)."""
    tables: dict[str, core.CalibrationTable] = field(default_factory=dict)
    """The latest calibration table per device hash (the device layer compares its solutions with it)."""
    stability: layer_mod.StabilityMap | None = None
    """The Mathieu stability boundary every device's trap page draws, computed by the first derive."""


WARM_START_PATH = Path(__file__).resolve().with_name("warm_start.pkl")
"""The warm start a packaged build ships, written by ``python -m qutip_trap_app.warm_start``."""


@dataclass(frozen=True)
class WarmStart:
    """What a worker has derived that a later worker can start from: the built presets, their calibration (the core's
    cached reports and the tables read from them), the channel libraries and the stability boundary. Every entry but
    the boundary, which no device changes, is keyed by what it was derived from (the preset's arguments, the device hash,
    the seed, the numerics), so a request for anything else derives its own."""

    presets: dict[str, core.DevicePreset]
    calibration: dict[str, object]
    """The core's calibration reports by cache key (``core.CALIBRATION_CACHE``)."""
    tables: dict[str, core.CalibrationTable]
    libraries: dict[str, ChannelLibrary]
    stability: layer_mod.StabilityMap | None
    """None when no request derived a device layer."""

    @classmethod
    def after(cls, *requests: tuple[str, dict[str, Any]]) -> WarmStart:
        """What a worker that started with nothing holds once it has answered ``requests``, each a request's name and
        payload (in this process, whose calibration cache the reports are read from)."""
        state = _LiveState()
        for request, payload in requests:
            HANDLERS[request](state, lambda stage, fraction, message: None, **payload)
        return cls(
            presets=dict(state.presets),
            calibration=dict(core.CALIBRATION_CACHE.reports),
            tables=dict(state.tables),
            libraries=dict(state.libraries),
            stability=state.stability,
        )

    def state(self) -> _LiveState:
        """A worker's state that starts from this: the reports go into the process's calibration cache."""
        core.CALIBRATION_CACHE.reports.update(self.calibration)
        return _LiveState(
            presets=dict(self.presets),
            tables=dict(self.tables),
            libraries=dict(self.libraries),
            stability=self.stability,
        )


def initial_state() -> _LiveState:
    """The state a worker starts with. A checkout starts with nothing, so every derivation runs on the code as it
    stands; a packaged build, which carries no PLAN.md, starts from the warm start its build derived, so the start-up
    circuit's first Run finds its calibration table and channel library instead of deriving them (half a minute in a
    browser), and the first physics page the stability boundary."""
    if (core.repository_root() / "PLAN.md").exists():
        return _LiveState()
    warm: WarmStart = pickle.loads(WARM_START_PATH.read_bytes())
    return warm.state()


def _preset_for(state: _LiveState, ref: DeviceRef, progress: Progress) -> core.DevicePreset:
    """The built preset of a device reference, cached by its arguments and overrides; the hash is checked when the
    reference carries one (an empty hash is a request from the UI, which never builds a device itself)."""
    key = ref.cache_key()
    preset = state.presets.get(key)
    if preset is None:
        if ref.overrides:
            progress(
                "building device", None, "applying the Level 4 knobs and re-deriving the preparation recipe"
            )
        preset = ref.build(check=bool(ref.hash))
        state.presets[key] = preset
    elif ref.hash and preset.device.hash() != ref.hash:
        raise WorkerError(
            f"the device reference's hash {ref.hash[:12]} does not match the built device {preset.device.hash()[:12]}"
        )
    return preset


def _calibrated(
    state: _LiveState, job: JobSpec, progress: Progress, message: str
) -> tuple[JobSpec, core.DevicePreset, core.CalibrationTable]:
    """The job completed from its built device, with the job's calibration table (kept as the device's latest)."""
    preset = _preset_for(state, job.device, progress)
    job = complete_job(job, preset)
    progress("calibrating", None, message)
    table = calibrate_for(job, preset.device)
    state.tables[preset.device.hash()] = table
    return job, preset, table


def _at_level(job: JobSpec) -> str:
    """The fidelity level a job asked for, in its progress line; nothing for "auto", where the core picks one (the
    numerics strip names it once the run is in)."""
    return "" if job.level == "auto" else f" at {job.level}"


def _core_progress(progress: Progress) -> Callable[[core.Progress], None]:
    """The core's ``Progress`` of ``Machine.run`` (per pulse, branch, sample and readout) as worker progress events."""

    def forward(p: core.Progress) -> None:
        progress(f"running: {p.stage}", p.fraction, f"{p.stage} {p.done} of {p.total}")

    return forward


def _live(state: _LiveState, key: str) -> tuple[Record, LiveRun]:
    record, live = state.records.get(key), state.lives.get(key)
    if record is None or live is None:
        raise WorkerError(f"no live run for record key {key}: run the job again to re-simulate")
    return record, live


def _run_job(state: _LiveState, progress: Progress, *, job: JobSpec) -> Record:
    job, preset, _table = _calibrated(
        state, job, progress, "the surrogate calibration table (cached per device and seed)"
    )
    progress("running", None, f"{job.shots} shots on {preset.device.crystal.n_ions} ions{_at_level(job)}")
    record, live = execute(job, preset, progress=_core_progress(progress))
    state.records[record.key()] = record
    state.lives[record.key()] = live
    progress("recording", 1.0, "record built")
    return record


def _replay(state: _LiveState, progress: Progress, *, job: JobSpec) -> Record:
    job, preset, table = _calibrated(
        state, job, progress, "the surrogate calibration table (cached per device and seed)"
    )
    library = state.libraries.setdefault(
        f"{preset.device.hash()}/{table.seed}/{options_digest(job.options)}", ChannelLibrary()
    )
    record = replay(job, preset.device, table, library, progress=progress)
    state.records[record.key()] = record
    progress("recording", 1.0, "record built")
    return record


def _zoom(
    state: _LiveState,
    progress: Progress,
    *,
    key: str,
    step: int,
    sample: int = 0,
    branch: int = 0,
    n_store: int = resim.DEFAULT_ZOOM_POINTS,
) -> dict[str, Any]:
    record, live = _live(state, key)
    progress(
        "zooming", None, f"re-simulating {record.step(step).gate_id} with {n_store} stored points per segment"
    )
    record, z, stats = resim.zoom(record, live, step, sample, branch, n_store=n_store)
    progress("building", 0.9, "listing the Hamiltonian terms and collapse operators of the step")
    record, ham = resim.hamiltonian_record(record, live, step, sample, branch)
    state.records[key] = record
    return {"zoom": z, "boundaries": record.boundaries, "stats": stats, "hamiltonian": ham}


def _tomography(
    state: _LiveState, progress: Progress, *, key: str, step: int, sample: int = 0, branch: int = 0
) -> dict[str, Any]:
    record, live = _live(state, key)
    record, pm = resim.process_matrix(record, live, step, sample, branch, progress=progress)
    state.records[key] = record
    return {"process_matrix": pm, "boundaries": record.boundaries}


def _recheck(
    state: _LiveState, progress: Progress, *, key: str, step: int, sample: int = 0, branch: int = 0
) -> dict[str, Any]:
    record, live = _live(state, key)
    progress("rechecking", 0.0, "tolerances tightened by ten on the zoomed step")
    record, tol = resim.tolerance_recheck(record, live, step, sample, branch)
    progress("rechecking", 0.5, "every resolved cap raised by two, re-chained from the initial state")
    record, trunc = resim.truncation_recheck(record, live, step, sample, branch)
    state.records[key] = record
    return {"tolerance": tol, "truncation": trunc, "zooms": record.zooms, "boundaries": record.boundaries}


def _derive(
    state: _LiveState,
    progress: Progress,
    *,
    device: DeviceRef,
    table_record: TableRecord | None = None,
    sweeps: bool = True,
) -> layer_mod.DeviceLayer:
    preset = _preset_for(state, device, progress)
    if state.stability is None:
        progress("deriving", 0.0, "the Mathieu stability boundary, once for every device")
        state.stability = layer_mod.stability_map()
    return layer_mod.derive_device_layer(
        preset,
        stability=state.stability,
        overrides=device.overrides,
        table=state.tables.get(preset.device.hash()),
        table_record=table_record,
        sweeps=sweeps,
        progress=progress,
    )


def _recalibrate(state: _LiveState, progress: Progress, *, job: JobSpec) -> dict[str, Any]:
    _job, preset, table = _calibrated(
        state,
        job,
        progress,
        "the surrogate table for the edited device: closed forms, exact spot checks, detection records",
    )
    progress("recording", 1.0, "table built")
    return {"table": table_record(table), "device_hash": preset.device.hash()}


def _verify(
    state: _LiveState, progress: Progress, *, key: str, record: Record | None = None, shots: int | None = None
) -> dict[str, Any]:
    shallow = record if record is not None else state.records.get(key)
    if shallow is None:
        raise WorkerError(f"no record for key {key}")
    report, deep, deep_live = verify_deeper(
        shallow, live=state.lives.get(key), shots=shots, progress=progress
    )
    if deep is not None:
        state.records[deep.key()] = deep
        if deep_live is not None:
            state.lives[deep.key()] = deep_live
        if report.deep_level is None:
            state.records[key] = deep
    return {"report": report, "deep": deep}


def _request_run(state: _LiveState, progress: Progress, *, job: JobSpec, step: int = -1) -> Record:
    """A request made at Level 1 or 2 runs as its own job at the full engine (Section 14.4), and the requested gate's
    process matrix is computed at once, so Level 1 can show the actual unitary beside the requested one."""
    job, preset, _table = _calibrated(
        state,
        job,
        progress,
        "the surrogate calibration table, with the hand-set detunings applied where requested",
    )
    progress("running", None, f"{job.shots} shots{_at_level(job)}: {job.label or 'the requested job'}")
    record, live = execute(job, preset, progress=_core_progress(progress))
    key = record.key()
    state.lives[key] = live
    if step >= 0 and record.traces:
        progress("tomography", 0.8, "the process matrix of the requested gate's step")
        record, _pm = resim.process_matrix(record, live, step, 0, 0, progress=progress)
    state.records[key] = record
    progress("recording", 1.0, "record built")
    return record


def _preset(_state: _LiveState, progress: Progress, *, preset_id: str) -> Any:
    return presets_mod.run_preset(preset_id, progress)


HANDLERS: dict[str, Callable[..., Any]] = {
    "run_job": _run_job,
    "replay": _replay,
    "zoom": _zoom,
    "tomography": _tomography,
    "recheck": _recheck,
    "derive": _derive,
    "recalibrate": _recalibrate,
    "verify": _verify,
    "request_run": _request_run,
    "preset": _preset,
}
"""The requests the UI names, each resolved to its handler once, at submit time."""


# ---- running a request ---------------------------------------------------------------------------------------------------------

_Item = tuple[str, str, Callable[..., Any], dict[str, Any]]
"""(ticket id, request name, handler, payload)."""


def _run(state: _LiveState, results: Any, item: _Item) -> None:
    """One request: its progress events, then its result or its error, on ``results``."""
    ticket, request, handler, payload = item
    t0 = time.perf_counter()

    def progress(stage: str, fraction: float | None, message: str) -> None:
        results.put(
            Event(
                "progress",
                ticket,
                request,
                stage=stage,
                fraction=fraction,
                message=message,
                wall_time_s=time.perf_counter() - t0,
            )
        )

    try:
        value = handler(state, progress, **payload)
        results.put(Event("result", ticket, request, payload=value, wall_time_s=time.perf_counter() - t0))
    except Exception:
        results.put(
            Event(
                "error", ticket, request, message=traceback.format_exc(), wall_time_s=time.perf_counter() - t0
            )
        )


def _worker_main(requests: Any, results: Any) -> None:
    state = initial_state()
    while (item := requests.get()) is not None:
        _run(state, results, item)


class Transport(Protocol):
    """Where the worker's requests run. ``results`` is a queue of :class:`Event` read with ``get``/``get_nowait``."""

    results: Any

    @property
    def alive(self) -> bool:
        """Whether requests put now will be answered."""

    @property
    def failure(self) -> str:
        """Why the transport stopped answering, when it knows; empty while alive and when it does not."""

    def put(self, item: _Item) -> None:
        """Queue one request; its events arrive on ``results``."""

    def stop(self, timeout_s: float) -> None:
        """End the worker, first letting the request in flight finish for up to ``timeout_s`` where the transport can
        wait for it."""

    def kill(self) -> None:
        """End the worker now, whatever it is doing."""


class ProcessTransport:
    """The worker in a spawned, non-daemonic process."""

    failure = ""

    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        self.requests: Any = ctx.Queue()
        self.results: Any = ctx.Queue()
        self.process = ctx.Process(
            target=_worker_main,
            args=(self.requests, self.results),
            daemon=False,
            name="qutip-trap-app-worker",
        )
        self.process.start()

    @property
    def alive(self) -> bool:
        return bool(self.process.is_alive())

    def put(self, item: _Item) -> None:
        self.requests.put(item)

    def stop(self, timeout_s: float) -> None:
        if self.alive:
            self.requests.put(None)
        self.process.join(timeout_s)
        self.kill()

    def kill(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2.0)


INLINE_START_DELAY_S = 0.1
"""How long an inline request waits on the event loop before it starts: the renders its submission caused are sent first,
so the screen shows the job as running before the loop is held."""


class InlineTransport:
    """The worker on the caller's own thread and event loop: one Python runtime instead of two (about 300 MB less in a
    browser), and nothing else runs while a request does: no progress arrives, no click is answered and a cancel cannot
    interrupt. A request starts ``INLINE_START_DELAY_S`` after it is put, never inside ``put``."""

    alive = True
    failure = ""

    def __init__(self) -> None:
        self.state = initial_state()
        self.results: Any = queue.Queue()
        self._loop = asyncio.get_running_loop()

    def put(self, item: _Item) -> None:
        self._loop.call_later(INLINE_START_DELAY_S, _run, self.state, self.results, item)

    def stop(self, timeout_s: float) -> None:
        return None

    def kill(self) -> None:
        return None


WEB_WORKER_SCRIPT = "simulation-worker.js"
"""The simulation worker's script (``src/assets``), served beside the script of the worker that runs the page's Python."""


class WebWorkerTransport:
    """The worker in a second Python runtime, in a Web Worker started from the one that runs the page's Python: the
    browser's stand-in for a process. Requests and events cross as pickles; ``kill`` terminates the Web Worker.

    The script boots the runtime the page itself was loaded with, from the same two URLs (both in the browser's cache by
    then), and hands over to :func:`serve_web_worker`; requests put meanwhile wait for the boot. A boot or script failure
    arrives as ``failure`` and the transport is no longer alive."""

    def __init__(self) -> None:
        import js
        from pyodide.ffi import create_proxy, to_js

        self.results: Any = queue.Queue()
        self.failure = ""
        self._to_js = to_js
        self._proxies = [create_proxy(self._on_message), create_proxy(self._on_error)]
        self._worker: Any = js.Worker.new(WEB_WORKER_SCRIPT, type="module")
        self._worker.onmessage, self._worker.onerror = self._proxies
        # Flet's worker script keeps what it booted this runtime from on its global scope, relative to its own address
        here = js.location.href
        boot = {
            "pyodideUrl": str(js.URL.new(js.pyodideUrl, here).href),
            "appPackageUrl": str(js.URL.new(js.appPackageUrl, here).href),
        }
        self._worker.postMessage(to_js(boot, dict_converter=js.Object.fromEntries))

    @property
    def alive(self) -> bool:
        return self._worker is not None and not self.failure

    def _on_message(self, message: Any) -> None:
        data = message.data
        if isinstance(data, str):
            self.failure = data
        else:
            self.results.put(pickle.loads(data.to_py()))

    def _on_error(self, error: Any) -> None:
        self.failure = f"the simulation worker's script failed: {error.message}"

    def put(self, item: _Item) -> None:
        self._worker.postMessage(self._to_js(pickle.dumps(item)))

    def stop(self, timeout_s: float) -> None:
        self.kill()

    def kill(self) -> None:
        if self._worker is None:
            return
        self._worker.terminate()
        self._worker = None
        for proxy in self._proxies:
            proxy.destroy()


def serve_web_worker() -> Callable[[Any], None]:
    """Inside the simulation Web Worker, called once by its script: the function the script hands every later message to.
    A message is one pickled request; its events are posted back as pickles while it runs, so progress reaches the page
    during a solve."""
    import js
    from pyodide.ffi import to_js

    state = initial_state()

    class Outbox:
        def put(self, event: Event) -> None:
            js.postMessage(to_js(pickle.dumps(event)))

    outbox = Outbox()

    def handle(message: Any) -> None:
        _run(state, outbox, pickle.loads(message.to_py()))

    return handle


IN_BROWSER = sys.platform == "emscripten"
"""Whether this Python runs under WebAssembly in a browser, which has Web Workers and no processes."""


def _browser_transport() -> Callable[[], Transport]:
    """A browser's transport: a Web Worker, or the page's own runtime when the page's address asks ``?worker=inline``."""
    import js

    inline = js.URL.new(js.documentUrl).searchParams.get("worker") == "inline"
    return InlineTransport if inline else WebWorkerTransport


def platform_transport() -> Callable[[], Transport]:
    """The transport this platform runs the worker on, resolved once where the application starts."""
    return _browser_transport() if IN_BROWSER else ProcessTransport


class SimulationWorker:
    """One worker on the given transport; submit requests, poll events, or wait for one ticket."""

    def __init__(self, transport: Callable[[], Transport] = ProcessTransport) -> None:
        self._new_transport = transport
        self._transport: Transport | None = None
        self._buffer: list[Event] = []

    @property
    def alive(self) -> bool:
        return self._transport is not None and self._transport.alive

    @property
    def started(self) -> bool:
        """Whether a worker was started and not deliberately stopped; with ``alive`` False this means it died."""
        return self._transport is not None

    @property
    def failure(self) -> str:
        """Why a started worker is no longer alive, when its transport knows."""
        return "" if self._transport is None else self._transport.failure

    def start(self) -> None:
        if not self.alive:
            if self._transport is not None:
                self._transport.kill()  # a transport that died: release what it still holds
            self._transport = self._new_transport()

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._transport is not None:
            self._transport.stop(timeout_s)
        self._transport = None

    def cancel(self) -> None:
        """Hard cancel: terminate the worker and start a fresh one (the live handles are lost; the records' caches are not)."""
        if self._transport is not None:
            self._transport.kill()
        self._transport = None
        self._buffer.clear()
        self.start()

    def submit(self, request: str, **payload: Any) -> Ticket:
        handler = HANDLERS.get(request)
        if handler is None:
            raise WorkerError(f"unknown request {request!r}; known: {sorted(HANDLERS)}")
        self.start()
        assert self._transport is not None
        ticket = Ticket(uuid.uuid4().hex[:12], request)
        self._transport.put((ticket.id, request, handler, payload))
        return ticket

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        """Every event that has arrived (a first blocking wait of ``timeout_s`` when nothing is buffered)."""
        out: list[Event] = list(self._buffer)
        self._buffer.clear()
        if self._transport is None:
            return out
        block = timeout_s > 0.0 and not out
        while True:
            try:
                ev: Event = (
                    self._transport.results.get(timeout=timeout_s)
                    if block
                    else self._transport.results.get_nowait()
                )
            except queue.Empty:
                break
            block = False
            out.append(ev)
        return out

    def wait(self, ticket: Ticket, timeout_s: float = 600.0, on_progress: Any = None) -> Any:
        """Block until ``ticket`` finishes; other tickets' events are buffered for a later ``poll``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for ev in self.poll(timeout_s=0.2):
                if ev.ticket != ticket.id:
                    self._buffer.append(ev)
                    continue
                if ev.kind == "progress":
                    if on_progress is not None:
                        on_progress(ev)
                elif ev.kind == "result":
                    return ev.payload
                else:
                    raise WorkerError(ev.message)
            if not self.alive:
                raise WorkerError("the worker process died")
        raise WorkerError(f"request {ticket.kind} timed out after {timeout_s:g} s")
