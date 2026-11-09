"""The mode-factorized drive operator sigma_+ (x) prod_m D_m as a QuTiP data type.

Applied factor by factor, a drive term costs sum_m d_m multiply-adds per amplitude instead of the prod_m d_m of the
assembled matrix. :class:`FactorizedOperator` holds the operator as its factors and registers with QuTiP's dispatchers
for what the solvers call on it: ``matmul`` on a Dense state (the right-hand side), ``matmul`` of two factorized operators
(the c^dag c of a factorized collapse operator), scalar ``mul``, ``adjoint``, ``iszero`` and a structural ``isequal``.
Anything else converts to Dense through QuTiP's conversion graph, which assembles the matrix (the reason every ``mesolve``
segment builds the assembled CSR operator). The factors are the per-mode exponentials the space oracle-checks; an ENR
group's sum-generator exponential is not a product, so a drive on an ENR mode is never factorized.

The ``auto`` choice of ``BuilderOptions.kernel`` is the cost model below, fitted through ``PhasedSum.apply`` (the rotating
frame's right-hand side) over the sigma_+ and sigma_- terms of one to four ions on spaces of dimension 24 to 2048: the
assembled terms cost their non-zeros at 0.45 ns (0.59 ns per stored one), the factorized ones the elements of their gathered source blocks at
6.5 ns (gather, two products and scatter) and every product of a middle mode factor; a factorized register is the cheaper
on every space with two resolved modes or more, and the assembled one on a single mode below a few dozen states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np
import qutip as qt
from qutip.core import data as _data
from qutip.core.data import Data, Dense

KernelChoice = Literal["auto", "factorized", "assembled"]

COST_ASSEMBLED_US = 2.41
"""What an evaluation of assembled drive terms costs whatever their number (us): the phases, the zero fill, the dispatch."""
COST_ASSEMBLED_TERM_US = 0.25
"""Per assembled term (us)."""
COST_ASSEMBLED_NS_PER_NNZ = 0.45
"""Per non-zero of an assembled term (ns), counted as D prod_m d_m / d_ion: the CSR product's memory-bound rate (0.59 ns
per stored non-zero) over the elements the conversion keeps."""
COST_FACTORIZED_US = 3.21
"""What an evaluation of factorized drive terms costs whatever their number (us): the phases and one batch's calls."""
COST_FACTORIZED_NS_PER_ELEMENT = 6.47
"""Per element of the terms' gathered source blocks (ns): the gather, the products of the leading and trailing mode factors
and the scatter."""
COST_FACTORIZED_MIDDLE_US = 0.086
"""Per matrix product of a middle mode factor (us), one per level of the factors before it (three modes or more)."""


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def _middle_products(dims: Sequence[int], ion_factor: int, mode_factors: Sequence[int]) -> int:
    """The matrix products the middle mode factors of one term take in its block's layout (``_Term.order``): the first mode
    leads, the last trails, and a middle one is applied once per level of every kept factor before it."""
    modes = sorted(int(f) for f in mode_factors)
    if len(modes) < 3:
        return 0
    first, last = modes[0], modes[-1]
    between = [f for f in range(len(dims)) if f not in (ion_factor, first, last)]
    return sum(int(dims[first]) * _prod([dims[f] for f in between if f < m]) for m in modes[1:-1])


def kernel_costs_us(
    dims: Sequence[int], ion_factors: Sequence[int], mode_factors: Sequence[int]
) -> tuple[float, float]:
    """(assembled CSR, factorized) microseconds per evaluation of the drive terms sigma_+^i (x) prod_m D_m and their
    conjugates of every ion factor i in ``ion_factors`` on ``dims``, all held one way.

    An assembled term has D prod_m d_m / d_ion non-zeros; the factorized terms share one batch, which gathers the D/d_ion
    amplitudes of each term's source level.
    """
    d = _prod(dims)
    modes_size = _prod([dims[f] for f in mode_factors])
    blocks = [d // int(dims[f]) for f in ion_factors]
    assembled = COST_ASSEMBLED_US + sum(
        2.0 * (COST_ASSEMBLED_TERM_US + COST_ASSEMBLED_NS_PER_NNZ * 1e-3 * block * modes_size)
        for block in blocks
    )
    factorized = (
        COST_FACTORIZED_US
        + COST_FACTORIZED_NS_PER_ELEMENT * 1e-3 * 2.0 * sum(blocks)
        + COST_FACTORIZED_MIDDLE_US * 2.0 * sum(_middle_products(dims, f, mode_factors) for f in ion_factors)
    )
    return assembled, factorized


def prefer_factorized(dims: Sequence[int], ion_factor: int, mode_factors: Sequence[int]) -> bool:
    """The ``auto`` rule for one ion's drive terms: factorize when the cost model says their factorized batch is the cheaper
    (a register's batch of several ions only widens the margin)."""
    if not mode_factors:
        return False
    assembled, factorized = kernel_costs_us(dims, [ion_factor], mode_factors)
    return factorized < assembled


# ---- the data-layer type -------------------------------------------------------------------------------------------------------


class FactorizedOperator(Data):  # type: ignore[misc]
    """scale x (x)_f A_f on a product space of tensor ``dims`` (ions, then resolved modes), ``factors`` mapping a factor
    index to its square matrix (the identity on every factor not named), applied axis by axis."""

    def __init__(self, dims: Sequence[int], factors: Mapping[int, np.ndarray], scale: complex = 1.0) -> None:
        dims_t = tuple(int(d) for d in dims)
        if not dims_t or any(d < 1 for d in dims_t):
            raise ValueError("dims are positive tensor dimensions")
        facs: dict[int, np.ndarray] = {}
        for f, a in factors.items():
            k = int(f)
            if not 0 <= k < len(dims_t):
                raise IndexError(f"factor {k} out of range for {len(dims_t)} factors")
            arr = np.ascontiguousarray(a, dtype=complex)
            if arr.shape != (dims_t[k], dims_t[k]):
                raise ValueError(f"factor {k} has shape {arr.shape}, expected {(dims_t[k], dims_t[k])}")
            facs[k] = arr
        self.dims = dims_t
        self.factors = facs
        self.scale = complex(scale)
        n = _prod(dims_t)
        super().__init__((n, n))
        self.term = _term(dims_t, facs, self.scale)
        self._alone: FactorizedSum | None = None

    def copy(self) -> FactorizedOperator:
        return FactorizedOperator(self.dims, {k: v.copy() for k, v in self.factors.items()}, self.scale)

    def to_array(self) -> np.ndarray:
        """The assembled dense matrix (conversions only; never on the propagation path)."""
        out = np.array([[self.scale]], dtype=complex)
        for f, d in enumerate(self.dims):
            a = self.factors.get(f)
            out = np.kron(out, np.eye(d, dtype=complex) if a is None else a)
        return out

    def adjoint(self) -> FactorizedOperator:
        return FactorizedOperator(
            self.dims, {k: v.conj().T for k, v in self.factors.items()}, np.conj(self.scale)
        )

    def __reduce__(self) -> tuple[object, ...]:
        return (FactorizedOperator, (self.dims, self.factors, self.scale))

    def apply(self, arr: np.ndarray, scale: complex = 1.0) -> np.ndarray:
        """(scale x this operator) @ arr for ``arr`` of shape (D,) or (D, ncols), returned in the same shape (C order)."""
        a = np.asarray(arr)
        vector = a.ndim == 1
        if a.shape[0] != self.shape[0]:
            raise ValueError(f"state of dimension {a.shape[0]} does not live on dims {self.dims}")
        a2 = np.ascontiguousarray(a.reshape(-1, 1) if vector else a, dtype=complex)
        if self._alone is None:
            self._alone = FactorizedSum([self])
        single = self._alone._single
        if single is not None:
            out = single.apply(a2, complex(scale))
        else:
            out = np.zeros(a2.shape, dtype=complex)  # a zero operator: no live term
        return out.reshape(-1) if vector else out

    @property
    def max_abs(self) -> float:
        """The largest element modulus: |scale| prod_f max|A_f| (the identity contributes 1)."""
        out = abs(self.scale)
        for a in self.factors.values():
            out *= float(np.max(np.abs(a))) if a.size else 0.0
        return out


# ---- the application plan ------------------------------------------------------------------------------------------------------

_COLUMNS = -1
"""The column axis's place holder in a block's axis order."""


class _Term(NamedTuple):
    """How one factorized operator is applied, read off its factors once: every factor with a single non-zero element is a
    slice of the source block into the target block (``sliced``: factor, target row, source column), its element folded into
    ``prefactor`` with the scale; every other named factor is a step (``steps``: factor, kind, matrix or diagonal, in
    ascending factor order), the prefactor folded into the first; ``tail`` multiplies the slice when no step is left (a bare
    sigma_+), and is 1 otherwise. A factor with no non-zero element makes the prefactor 0: the term is zero."""

    dims: tuple[int, ...]
    sliced: tuple[tuple[int, int, int], ...]
    steps: tuple[tuple[int, Literal["dense", "diag"], np.ndarray], ...]
    prefactor: complex
    tail: complex

    @property
    def order(self) -> tuple[int, ...]:
        """The axes of the term's block in application order, ``_COLUMNS`` for the state's column axis: the first step's
        factor leads when there are two steps or more, the last step's factor trails (after the column axis) when it is
        dense, and every other kept factor sits between them in ascending order."""
        kept = [f for f in range(len(self.dims)) if f not in {s for s, _r, _c in self.sliced}]
        steps = self.steps
        trail = steps[-1][0] if steps and steps[-1][1] == "dense" else None
        lead = steps[0][0] if len(steps) >= 2 and steps[0][1] == "dense" else None
        middle = [f for f in kept if f not in (lead, trail)]
        return tuple(
            ([lead] if lead is not None else [])
            + middle
            + [_COLUMNS]
            + ([trail] if trail is not None else [])
        )

    @property
    def arrangement(self) -> tuple[object, ...]:
        """What the terms of one batch share: the block's shape in application order and every step's place and kind. The
        drive terms of the ions of one register share it (each slices a different ion's factor out of the same shape)."""
        order = self.order
        return (
            self.dims,
            tuple(_COLUMNS if f == _COLUMNS else self.dims[f] for f in order),
            tuple((order.index(f), k) for f, k, _m in self.steps),
        )


def _term(dims: tuple[int, ...], factors: Mapping[int, np.ndarray], scale: complex) -> _Term:
    prefactor = complex(scale)
    sliced: list[tuple[int, int, int]] = []
    remaining: list[tuple[int, Literal["dense", "diag"], np.ndarray]] = []
    for f in sorted(factors):
        a = factors[f]
        nz = np.argwhere(np.abs(a) > 0.0)
        if nz.shape[0] == 0:
            prefactor = 0.0 + 0.0j
            continue
        if nz.shape[0] == 1:
            i, j = int(nz[0][0]), int(nz[0][1])
            prefactor *= complex(a[i, j])
            sliced.append((f, i, j))
        elif np.count_nonzero(a - np.diag(np.diag(a))) == 0:
            remaining.append((f, "diag", np.ascontiguousarray(np.diag(a))))
        else:
            remaining.append((f, "dense", a))
    folded = prefactor != 0.0 and bool(remaining)
    steps = tuple(
        (f, kind, np.ascontiguousarray(m * prefactor if (folded and idx == 0) else m))
        for idx, (f, kind, m) in enumerate(remaining)
    )
    return _Term(dims, tuple(sliced), steps, prefactor, 1.0 + 0.0j if folded else prefactor)


class _Layout(NamedTuple):
    """A batch's gathered block for one column count: the flat source and target index of every element of every term's
    block (rows: terms), and per step how the block is reshaped around it ("left": the step's factor leads and the step is one
    product A @ (d, rest); "right": it trails, (rest, d) @ A^T; "middle": a product batched over the factors before it;
    "diag": a broadcast product), as (code, lead, d, trail)."""

    gather: np.ndarray
    scatter: np.ndarray
    ops: tuple[tuple[str, int, int, int], ...]


class _Batch:
    """Terms of one arrangement applied together: one gather of every term's source block, one batched product per step and
    the rows of the result in the block order the scatter reads.

    The block is laid out (``_Term.order``) so that each step is a single matrix product per term: two mode factors cost two
    BLAS calls per term, against one per level of the factors before the middle one in the unpermuted block. Each step
    multiplies the same elements in the same order as a product along that axis of the unpermuted block, so the layout
    changes the number of calls, not the arithmetic.
    """

    __slots__ = ("_layouts", "_ops", "k", "mats", "orders", "shape", "tail", "terms")

    def __init__(self, terms: Sequence[_Term]) -> None:
        first = terms[0]
        if any(t.arrangement != first.arrangement for t in terms):
            raise ValueError("a batch holds terms of one arrangement")
        self.terms = tuple(terms)
        self.k = len(terms)
        self.orders = tuple(t.order for t in terms)
        order = self.orders[0]
        self.shape = tuple(_COLUMNS if f == _COLUMNS else first.dims[f] for f in order)
        mats: list[np.ndarray] = []
        for s, (f, kind, _m) in enumerate(first.steps):
            stack = np.stack([t.steps[s][2] for t in terms])
            if kind == "dense" and order.index(f) == len(order) - 1:
                stack = np.ascontiguousarray(np.swapaxes(stack, 1, 2))
            mats.append(stack)
        self.mats = tuple(mats)
        self.tail = np.array([t.tail for t in terms], dtype=complex)
        self._layouts: dict[int, _Layout] = {}
        self._ops: dict[int, tuple[tuple[str, int, int, int], ...]] = {}

    def layout(self, ncols: int) -> _Layout:
        lay = self._layouts.get(ncols)
        if lay is None:
            lay = _Layout(
                self._block_index(ncols, "source"), self._block_index(ncols, "target"), self.ops(ncols)
            )
            self._layouts[ncols] = lay
        return lay

    def ops(self, ncols: int) -> tuple[tuple[str, int, int, int], ...]:
        """How the block is reshaped around every step at ``ncols`` columns (``_Layout.ops``), without the index arrays."""
        ops = self._ops.get(ncols)
        if ops is None:
            ops = self._build_ops(ncols)
            self._ops[ncols] = ops
        return ops

    def _block_index(self, ncols: int, which: Literal["source", "target"]) -> np.ndarray:
        dims = self.terms[0].dims
        base = np.arange(_prod(dims) * ncols).reshape(dims + (ncols,))
        rows = []
        for t, order in zip(self.terms, self.orders):
            index: list[int | slice] = [slice(None)] * (len(dims) + 1)
            for f, r, c in t.sliced:
                index[f] = c if which == "source" else r
            kept = [f for f in range(len(dims)) if f not in {s for s, _r, _c in t.sliced}] + [_COLUMNS]
            perm = [kept.index(f) for f in order]
            rows.append(np.ascontiguousarray(base[tuple(index)].transpose(perm)).reshape(-1))
        return np.stack(rows)

    def _build_ops(self, ncols: int) -> tuple[tuple[str, int, int, int], ...]:
        shape = [ncols if n == _COLUMNS else n for n in self.shape]
        order = self.orders[0]
        ops: list[tuple[str, int, int, int]] = []
        for f, kind, _m in self.terms[0].steps:
            p = order.index(f)
            lead, d, trail = _prod(shape[:p]), shape[p], _prod(shape[p + 1 :])
            if kind == "diag":
                ops.append(("diag", lead, d, trail))
            elif p == 0:
                ops.append(("left", lead, d, trail))
            elif p == len(shape) - 1:
                ops.append(("right", lead, d, trail))
            else:
                ops.append(("middle", lead, d, trail))
        return tuple(ops)

    def transform(self, x_flat: np.ndarray, ncols: int, alpha: np.ndarray) -> np.ndarray:
        """Row k: alpha[k] x term k on its source block of the flattened (D, ncols) state ``x_flat``, in the layout's order."""
        return self.steps(x_flat[self.layout(ncols).gather], ncols, alpha)

    def steps(self, y: np.ndarray, ncols: int, alpha: np.ndarray) -> np.ndarray:
        """Row k: alpha[k] x term k on its gathered source block ``y`` (row k, in the layout's order); alpha rides on the
        first step's matrix (d^2 multiplications) rather than on the block."""
        k = self.k
        ops = self.ops(ncols)
        for s, (code, lead, d, trail) in enumerate(ops):
            m = self.mats[s]
            if s == 0:
                m = m * (alpha[:, None, None] if m.ndim == 3 else alpha[:, None])
            if code == "left":
                y = np.matmul(m, y.reshape(k, d, trail))
            elif code == "right":
                y = np.matmul(y.reshape(k, lead, d), m)
            elif code == "middle":
                y = np.matmul(m[:, None], y.reshape(k, lead, d, trail))
            else:
                y = y.reshape(k, lead, d, trail) * m[:, None, :, None]
        if not ops:
            y = y * (self.tail * alpha)[:, None]
        return y.reshape(k, -1)


class _Single:
    """A sum of one live term applied through views: its source block is a slice of the state (no index array), put in the
    layout's order by a transpose, the batch's steps applied to it as two-dimensional products, and the result assigned into
    the slice of the target block; the same elements multiplied in the same order as the batch's gather, steps and
    scatter, with none of their per-term bookkeeping."""

    __slots__ = ("_plans", "batch", "inverse", "perm", "position", "source", "target")

    def __init__(self, batch: _Batch, position: int) -> None:
        (term,) = batch.terms
        dims = term.dims
        self.batch = batch
        self.position = position
        source: list[int | slice] = [slice(None)] * (len(dims) + 1)
        target: list[int | slice] = [slice(None)] * (len(dims) + 1)
        for f, r, c in term.sliced:
            source[f], target[f] = c, r
        self.source = tuple(source)
        self.target = tuple(target)
        kept = [f for f in range(len(dims)) if f not in {s for s, _r, _c in term.sliced}] + [_COLUMNS]
        self.perm = tuple(kept.index(f) for f in batch.orders[0])
        self.inverse = tuple(int(i) for i in np.argsort(self.perm))
        self._plans: dict[
            int, tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[str, int, int, int, np.ndarray], ...]]
        ] = {}

    def _plan(
        self, ncols: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[str, int, int, int, np.ndarray], ...]]:
        """(the state's tensor shape, the block's layout shape, every step with its one matrix) at ``ncols`` columns."""
        plan = self._plans.get(ncols)
        if plan is None:
            batch = self.batch
            state = batch.terms[0].dims + (ncols,)
            layout = tuple(ncols if n == _COLUMNS else n for n in batch.shape)
            steps = tuple((*op, m[0]) for op, m in zip(batch.ops(ncols), batch.mats))
            plan = (state, layout, steps)
            self._plans[ncols] = plan
        return plan

    def apply(self, x: np.ndarray, alpha: complex) -> np.ndarray:
        """alpha x the term @ x for a C-contiguous complex (D, ncols) ``x``: a new (D, ncols) array."""
        ncols = x.shape[1]
        state, layout, steps = self._plan(ncols)
        y = np.ascontiguousarray(x.reshape(state)[self.source].transpose(self.perm))
        first = True
        for code, lead, d, trail, m in steps:
            if first:
                m = m * alpha
                first = False
            if code == "left":
                y = m @ y.reshape(d, trail)
            elif code == "right":
                y = y.reshape(lead, d) @ m
            elif code == "middle":
                y = np.matmul(m, y.reshape(lead, d, trail))
            else:
                y = y.reshape(lead, d, trail) * m[None, :, None]
        if first:
            y = y * (self.batch.tail[0] * alpha)
        result = y.reshape(layout).transpose(self.inverse)
        if not self.batch.terms[0].sliced:
            return np.ascontiguousarray(result).reshape(-1, ncols)
        out = np.zeros(state, dtype=complex)
        out[self.target] = result
        return out.reshape(-1, ncols)


class _Scatter(NamedTuple):
    """One run of consecutive terms whose target blocks are disjoint: where its rows sit (batch, first row, end row; one
    segment per batch it draws on), the flat target index of its elements and, when they cover the state exactly once, the
    permutation that puts them in state order."""

    segments: tuple[tuple[int, int, int], ...]
    target: np.ndarray
    permutation: np.ndarray | None


class FactorizedSum:
    """sum_k c_k A_k applied to a state at once, for factorized operators A_k on one product space.

    The terms of one arrangement share a :class:`_Batch` (the drive terms of a register, sigma_+^i or sigma_-^i (x) prod_m D_m,
    differ only in the slice they read and write and in their factor matrices), so a step costs one batched product for all
    of them. The results are added in the terms' order, in runs of consecutive terms whose target blocks are disjoint, so that
    every element receives its contributions in the order a term-by-term loop gives it. A zero term (a factor with no
    non-zero element) contributes nothing.
    """

    __slots__ = ("_batches", "_runs", "_scatters", "_single", "d", "dims", "n_terms")

    def __init__(self, operators: Sequence[FactorizedOperator]) -> None:
        if not operators:
            raise ValueError("a factorized sum needs at least one operator")
        dims = operators[0].dims
        if any(op.dims != dims for op in operators):
            raise ValueError("the operators of a factorized sum live on one space")
        self.dims = dims
        self.d = _prod(dims)
        self.n_terms = len(operators)
        terms = [op.term for op in operators]
        groups: dict[tuple[object, ...], list[int]] = {}
        for k, t in enumerate(terms):
            if t.prefactor != 0.0:
                groups.setdefault(t.arrangement, []).append(k)
        self._batches: tuple[tuple[_Batch, np.ndarray], ...] = tuple(
            (_Batch([terms[k] for k in ks]), np.array(ks, dtype=np.intp)) for ks in groups.values()
        )
        where = {int(k): (b, r) for b, (_bt, ks) in enumerate(self._batches) for r, k in enumerate(ks)}
        # runs of consecutive live terms with disjoint targets, from the one-column index
        runs: list[list[tuple[int, int]]] = []
        covered = np.zeros(self.d, dtype=bool)
        for k in sorted(where):
            b, r = where[k]
            tgt = self._batches[b][0].layout(1).scatter[r]
            if not runs or bool(np.any(covered[tgt])):
                runs.append([])
                covered[:] = False
            runs[-1].append((b, r))
            covered[tgt] = True
        self._runs = tuple(tuple(run) for run in runs)
        self._scatters: dict[int, tuple[_Scatter, ...]] = {}
        self._single: _Single | None = None
        if len(where) == 1:
            ((batch, (k,)),) = self._batches
            self._single = _Single(batch, int(k))

    def _scatter_plan(self, ncols: int) -> tuple[_Scatter, ...]:
        plan = self._scatters.get(ncols)
        if plan is not None:
            return plan
        out: list[_Scatter] = []
        for run in self._runs:
            segments: list[list[int]] = []
            for b, r in run:
                if segments and segments[-1][0] == b and segments[-1][2] == r:
                    segments[-1][2] = r + 1
                else:
                    segments.append([b, r, r + 1])
            target = np.concatenate([self._batches[b][0].layout(ncols).scatter[r] for b, r in run])
            full = target.size == self.d * ncols and np.unique(target).size == target.size
            out.append(
                _Scatter(
                    tuple((b, lo, hi) for b, lo, hi in segments),
                    target,
                    np.argsort(target, kind="stable") if full else None,
                )
            )
        plan = tuple(out)
        self._scatters[ncols] = plan
        return plan

    def apply(self, x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        """sum_k coefs[k] A_k @ x for a C-contiguous complex (D, ncols) ``x``: a new (D, ncols) array."""
        if self._single is not None:
            return self._single.apply(x, complex(coefs[self._single.position]))
        ncols = x.shape[1]
        flat = x.reshape(-1)
        rows = [bt.transform(flat, ncols, coefs[ks]) for bt, ks in self._batches]
        out: np.ndarray | None = None
        for sc in self._scatter_plan(ncols):
            if len(sc.segments) == 1:
                b, lo, hi = sc.segments[0]
                z = rows[b][lo:hi].reshape(-1)
            else:
                z = np.concatenate([rows[b][lo:hi].reshape(-1) for b, lo, hi in sc.segments])
            if sc.permutation is not None:
                if out is None:
                    out = z[sc.permutation]
                else:
                    out += z[sc.permutation]
            else:
                if out is None:
                    out = np.zeros(self.d * ncols, dtype=complex)
                out[sc.target] += z
        if out is None:
            out = np.zeros(self.d * ncols, dtype=complex)
        return out.reshape(self.d, ncols)


# ---- dispatcher specialisations ----------------------------------------------------------------------------------------------------


def _to_dense(matrix: FactorizedOperator) -> Dense:
    return Dense(matrix.to_array(), copy=False)


def _from_dense(matrix: Dense) -> FactorizedOperator:
    """A dense matrix as a one-factor operator (the direction QuTiP's conversion registry requires)."""
    arr = np.array(matrix.as_ndarray(), dtype=complex, copy=True)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("a factorized operator is square")
    return FactorizedOperator((arr.shape[0],), {0: arr}, 1.0)


def _matmul_factorized_dense(left: FactorizedOperator, right: Dense, scale: complex = 1) -> Dense:
    return Dense(left.apply(right.as_ndarray(), scale), copy=False)


def _matmul_factorized_factorized(
    left: FactorizedOperator, right: FactorizedOperator, scale: complex = 1
) -> FactorizedOperator:
    """(x)_f A_f (x)_f B_f = (x)_f A_f B_f on a shared factor structure; assembled when the structures differ."""
    if left.dims != right.dims:
        if left.shape != right.shape:
            raise ValueError("operator shapes do not match")
        return _from_dense(Dense(left.to_array() @ right.to_array(), copy=False))
    factors: dict[int, np.ndarray] = {}
    for f in range(len(left.dims)):
        a, b = left.factors.get(f), right.factors.get(f)
        if a is None and b is None:
            continue
        if a is None:
            assert b is not None
            factors[f] = b.copy()
        elif b is None:
            factors[f] = a.copy()
        else:
            factors[f] = a @ b
    return FactorizedOperator(left.dims, factors, left.scale * right.scale * scale)


def _mul(matrix: FactorizedOperator, value: complex) -> FactorizedOperator:
    return FactorizedOperator(matrix.dims, matrix.factors, matrix.scale * value)


def _adjoint(matrix: FactorizedOperator) -> FactorizedOperator:
    return matrix.adjoint()


def _iszero(matrix: FactorizedOperator, tol: float = -1) -> bool:
    if tol < 0:
        tol = float(qt.settings.core["atol"])
    return matrix.max_abs <= tol


def _scalar_ratio(x: np.ndarray, y: np.ndarray, atol: float, rtol: float) -> complex | None:
    """The c with ``x == c y`` elementwise, or None when ``x`` is not a scalar multiple of ``y``."""
    big = float(np.max(np.abs(y))) if y.size else 0.0
    if big == 0.0:
        return None
    i = np.unravel_index(int(np.argmax(np.abs(y))), y.shape)
    c = complex(x[i] / y[i])
    return c if np.allclose(x, c * y, atol=atol, rtol=rtol) else None


def _isequal(a: FactorizedOperator, b: FactorizedOperator, atol: float = -1, rtol: float = -1) -> bool:
    """Equality within tolerance. On one factor structure it is exact and never assembles: over the union of the factor
    indices each pair must be a scalar multiple, and the ratios must carry one scale into the other (scale_a (x)_f A_f =
    scale_b (x)_f B_f iff A_f = c_f B_f with scale_a prod_f c_f = scale_b); ``QobjEvo.compress`` merges terms on a "yes".
    Two structures over one dimension are compared assembled."""
    if atol < 0:
        atol = float(qt.settings.core["atol"])
    if rtol < 0:
        rtol = float(qt.settings.core["rtol"])
    if a.shape != b.shape:
        return False
    if a.dims != b.dims:
        return bool(np.allclose(a.to_array(), b.to_array(), atol=atol, rtol=rtol))
    zero_a, zero_b = _iszero(a, atol), _iszero(b, atol)
    if zero_a or zero_b:
        return zero_a and zero_b
    ratio = complex(1.0)
    for f in sorted(set(a.factors) | set(b.factors)):
        ident = np.eye(a.dims[f], dtype=complex)
        c = _scalar_ratio(a.factors.get(f, ident), b.factors.get(f, ident), atol, rtol)
        if c is None:
            return False
        ratio *= c
    return bool(abs(a.scale * ratio - b.scale) <= atol + rtol * abs(b.scale))


def _register_data_type() -> None:
    """Register :class:`FactorizedOperator` with QuTiP's conversion registry and dispatchers (once, at import)."""
    if FactorizedOperator in _data.to.dtypes:
        return
    _data.to.add_conversions(
        [
            (Dense, FactorizedOperator, _to_dense, 1.0),
            (FactorizedOperator, Dense, _from_dense, 10.0),
        ]
    )
    _data.to.register_aliases(["factorized", "Factorized"], FactorizedOperator)
    _data.matmul.add_specialisations(
        [
            (FactorizedOperator, Dense, Dense, _matmul_factorized_dense),
            (FactorizedOperator, FactorizedOperator, FactorizedOperator, _matmul_factorized_factorized),
        ]
    )
    _data.mul.add_specialisations([(FactorizedOperator, FactorizedOperator, _mul)])
    _data.imul.add_specialisations([(FactorizedOperator, FactorizedOperator, _mul)])
    _data.adjoint.add_specialisations([(FactorizedOperator, FactorizedOperator, _adjoint)])
    _data.iszero.add_specialisations([(FactorizedOperator, _iszero)])
    _data.isequal.add_specialisations([(FactorizedOperator, FactorizedOperator, _isequal)])


_register_data_type()


def factorized_qobj(
    dims: Sequence[int], factors: Mapping[int, np.ndarray | qt.Qobj], scale: complex = 1.0
) -> qt.Qobj:
    """A ``Qobj`` on the joint space of ``dims`` whose data is the factorized operator (identities on the factors not named)."""
    mats = {int(f): (a.full() if isinstance(a, qt.Qobj) else np.asarray(a)) for f, a in factors.items()}
    data = FactorizedOperator(dims, mats, scale)
    d = [int(x) for x in dims]
    return qt.Qobj(data, dims=[d, d], copy=False)
