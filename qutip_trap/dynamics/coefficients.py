"""QuTiP function coefficients built without re-reading the function: the builders' per-segment coefficients.

``qt.coefficient(fn, args=...)`` reads ``fn``'s signature with ``inspect.signature`` and probes ``fn`` at t = 0 on every call,
a few microseconds each, and a segment's Hamiltonian makes a dozen. :func:`function_coefficient` reads each function's
signature once, with QuTiP's own ``coefficient_function_parameters``, and passes the result to ``FunctionCoefficient``
through the ``_f_pythonic`` and ``_f_parameters`` arguments it takes for that (QuTiP 5.3; the package pins qutip < 5.4).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping

from qutip.core.cy.coefficient import Coefficient, FunctionCoefficient, coefficient_function_parameters


@functools.cache
def _style(fn: Callable[..., object]) -> tuple[bool, frozenset[str] | None]:
    """Whether ``fn`` takes its arguments as keywords, and the names it takes (None: any), as QuTiP reads them."""
    pythonic, parameters = coefficient_function_parameters(fn)
    return bool(pythonic), None if parameters is None else frozenset(parameters)


def function_coefficient(fn: Callable[..., complex | float], args: Mapping[str, object]) -> Coefficient:
    """The ``FunctionCoefficient`` that ``qt.coefficient(fn, args=args)`` returns, for a module-level function returning a
    number; arguments ``fn`` does not take are dropped, as QuTiP drops them."""
    pythonic, parameters = _style(fn)
    kept = dict(args) if parameters is None else {k: v for k, v in args.items() if k in parameters}
    return FunctionCoefficient(
        fn, kept, _f_pythonic=pythonic, _f_parameters=None if parameters is None else set(parameters)
    )
