"""Computed drawings and charts (PLAN.md Section 14.5 "No cartoons").

Every element drawn here is a number from a view-model: ion positions from the equilibrium solver, mode arrows from the
Hessian's eigenvectors, the stability boundary from the monodromy, beams from the configured geometry, levels from the
species table, loops from the recorded trajectories. The one illustrative choice, a level diagram's vertical spacing, is
drawn on a compressed scale and labelled "not to scale". Drawings use Flet's canvas; heatmaps are rendered to PNG by the
small encoder below (zlib plus the PNG chunk format). Colours follow the theme roles so both themes read, and meaning is
never carried by colour alone.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import flet as ft
import flet.canvas as cv
import flet_charts as fc
import numpy as np

from qutip_trap_app.viewmodel.catalogue import fmt_number, power_of_ten
from qutip_trap_app.views import theme

# ---- PNG heatmaps ---------------------------------------------------------------------------------------------------------------------

_SEQUENTIAL: tuple[tuple[float, float, float], ...] = (
    (0.267, 0.005, 0.329),
    (0.283, 0.141, 0.458),
    (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553),
    (0.164, 0.471, 0.558),
    (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518),
    (0.267, 0.749, 0.441),
    (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150),
    (0.993, 0.906, 0.144),
)
"""A perceptually ordered sequential ramp (the viridis anchors), dark for zero: the dataviz rule for a magnitude field."""


def sequential_rgb(t: float) -> tuple[int, int, int]:
    x = min(max(float(t), 0.0), 1.0) * (len(_SEQUENTIAL) - 1)
    k = min(int(math.floor(x)), len(_SEQUENTIAL) - 2)
    f = x - k
    a, b = _SEQUENTIAL[k], _SEQUENTIAL[k + 1]
    return tuple(int(round(255 * (a[i] + f * (b[i] - a[i])))) for i in range(3))  # type: ignore[return-value]


def png_bytes(rgba: np.ndarray) -> bytes:
    """A PNG from an (H, W, 4) uint8 array."""
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def heatmap_png(values: np.ndarray, *, cell_w: int, cell_h: int) -> bytes:
    """Rows are the vertical axis (drawn bottom-up: row 0 at the bottom), columns the horizontal one, on a log scale over six
    decades below the maximum: Fock populations span from 1 to 1e-6 and a linear ramp would show one bright cell."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    top = max(float(np.max(v, initial=0.0)), 1e-300)
    with np.errstate(divide="ignore"):
        t = np.clip((np.log10(np.clip(v, 1e-300, None)) - (math.log10(top) - 6.0)) / 6.0, 0.0, 1.0)
    rows, cols = t.shape
    img = np.zeros((rows * cell_h, cols * cell_w, 4), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            rgb = sequential_rgb(float(t[r, c]))
            y0 = (rows - 1 - r) * cell_h
            img[y0 : y0 + cell_h, c * cell_w : (c + 1) * cell_w, :3] = rgb
            img[y0 : y0 + cell_h, c * cell_w : (c + 1) * cell_w, 3] = 255
    return png_bytes(img)


HEATMAP_MAX_WIDTH = 520.0
"""The widest a heatmap grows to fill its card: beyond it the cells of a small matrix would be slabs."""
_HEATMAP_Y_LABELS = 30.0
"""The width of a heatmap's row labels, and of the gap that keeps its column labels under their columns."""


def heatmap(
    values: np.ndarray,
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    x_title: str,
    y_title: str,
    cell_w: int = 14,
    cell_h: int = 10,
    room: float | None = None,
    tooltip: str = "",
) -> ft.Control:
    """A labelled heatmap: the PNG with its row labels on the left and column labels beneath (a table alternative is the
    caller's, DESIGN.md Section 4). ``cell_w`` and ``cell_h`` are the smallest a cell is drawn; given the ``room`` of its
    card, a map of few columns grows its cells to fill it (up to ``HEATMAP_MAX_WIDTH``), so that a pulse of nine stored
    times is a picture rather than a postage stamp."""
    rows, cols = values.shape
    if room is not None:
        fill = (min(room, HEATMAP_MAX_WIDTH) - _HEATMAP_Y_LABELS - 4.0) / max(cols, 1)
        grown = int(min(max(fill, cell_w), 4.0 * cell_w))
        cell_w, cell_h = grown, int(min(max(cell_h, 0.45 * grown), 2.2 * cell_h))
    img = ft.Image(
        src=heatmap_png(values, cell_w=cell_w, cell_h=cell_h),
        width=cols * cell_w,
        height=rows * cell_h,
        fit=ft.BoxFit.FILL,
        border_radius=ft.BorderRadius.all(4),
        tooltip=tooltip or None,
    )
    ylab = ft.Column(
        [
            ft.Text(
                str(y_labels[r]),
                size=theme.SIZE_MICRO,
                height=cell_h,
                width=_HEATMAP_Y_LABELS,
                text_align=ft.TextAlign.RIGHT,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
            # a label per row while the rows are tall enough to hold one, else every second
            if cell_h >= 12 or r % 2 == 0
            else ft.Container(height=cell_h, width=_HEATMAP_Y_LABELS)
            for r in reversed(range(rows))
        ],
        spacing=0,
        tight=True,
    )
    # a column label needs about 6 px a character and a gap: label every column that many cells apart
    widest = max((len(str(x_labels[c])) for c in range(cols)), default=1)
    step = max(1, math.ceil((6.0 * widest + 8.0) / cell_w))
    xlab = ft.Row(
        [
            ft.Text(
                str(x_labels[c]),
                size=theme.SIZE_MICRO,
                width=cell_w * min(step, cols - c),
                no_wrap=True,
                text_align=ft.TextAlign.CENTER if step == 1 else ft.TextAlign.LEFT,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
            for c in range(0, cols, step)
        ],
        spacing=0,
        tight=True,
    )
    scale = ft.Row(
        [
            ft.Text(power_of_ten(-6), size=theme.SIZE_MICRO, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Image(
                src=heatmap_png(np.logspace(-6, 0, 12)[None, :], cell_w=10, cell_h=8),
                width=120,
                height=8,
                fit=ft.BoxFit.FILL,
                border_radius=ft.BorderRadius.all(2),
            ),
            ft.Text("1 (log scale)", size=theme.SIZE_MICRO, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        spacing=6,
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    lines: list[ft.Control] = []
    if y_title:
        lines.append(_axis_title(y_title))
    lines += [
        ft.Row([ylab, img], spacing=4, tight=True, vertical_alignment=ft.CrossAxisAlignment.START),
        ft.Row([ft.Container(width=_HEATMAP_Y_LABELS), xlab], spacing=4, tight=True),
        ft.Row(
            [ft.Container(width=_HEATMAP_Y_LABELS), _axis_title(x_title), ft.Container(width=12), scale],
            spacing=4,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    ]
    return ft.Column(lines, spacing=4, tight=True)


# ---- formulas ---------------------------------------------------------------------------------------------------------------------------


class Run(NamedTuple):
    """A run of a formula's characters and where it stands: on the line (0), raised as a superscript (+1) or lowered as
    a subscript (-1)."""

    text: str
    level: int = 0


def sup(text: str) -> Run:
    return Run(text, 1)


def sub(text: str) -> Run:
    return Run(text, -1)


FORMULA_SIZE = 17
_SCRIPT_SIZE = 12
_SCRIPT_SHIFT = 9.0
"""How far a superscript's or subscript's box is pushed off the line's centre (it is centred in the row otherwise)."""


def formula(terms: Sequence[Sequence[Run | str]]) -> ft.Control:
    """A formula set as mathematics: Greek letters as letters, superscripts raised and subscripts lowered at the script
    size, which the text engine has no markup for. ``terms`` are the pieces a narrow card may break the line between; a
    term itself never breaks."""

    def piece(run: Run | str) -> ft.Control:
        text, level = (run, 0) if isinstance(run, str) else run
        if level == 0:
            return ft.Text(text, size=FORMULA_SIZE, no_wrap=True)
        return ft.Container(
            content=ft.Text(text, size=_SCRIPT_SIZE, no_wrap=True),
            padding=ft.Padding.only(bottom=_SCRIPT_SHIFT)
            if level > 0
            else ft.Padding.only(top=_SCRIPT_SHIFT),
        )

    return ft.Row(
        [
            ft.Row(
                [piece(run) for run in term],
                spacing=0,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
            for term in terms
        ],
        wrap=True,
        spacing=6,
        run_spacing=2,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


# ---- canvas helpers ------------------------------------------------------------------------------------------------------------------


def _stroke(color: str, width: float = 1.5, dash: Sequence[float] | None = None) -> ft.Paint:
    return ft.Paint(
        color=color,
        stroke_width=width,
        style=ft.PaintingStyle.STROKE,
        stroke_dash_pattern=None if dash is None else [float(d) for d in dash],
    )


def _fill(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _label(
    x: float, y: float, text: str, *, size: int = 10, color: str = ft.Colors.ON_SURFACE_VARIANT
) -> cv.Text:
    return cv.Text(x, y, text, style=ft.TextStyle(size=size, color=color))


def _axis_title(text: str) -> ft.Text:
    return ft.Text(text, size=theme.SIZE_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT)


def _captioned(canvas: ft.Control, caption: str, width: float) -> ft.Control:
    """A drawing with its caption beneath it as text, which wraps at the drawing's width: drawn on the canvas, a caption
    longer than a narrow card is cut at the canvas's edge."""
    return ft.Column(
        [canvas, ft.Text(caption, size=theme.SIZE_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT, width=width)],
        spacing=4,
        tight=True,
    )


def _grid() -> fc.ChartGridLines:
    """Hairline gridlines one step off the surface, solid (the dataviz rule for recessive chart chrome)."""
    return fc.ChartGridLines(color=ft.Colors.OUTLINE_VARIANT, width=1)


class Axes:
    """A data-to-pixel map for one canvas with margins, drawing its own frame and a few ticks."""

    def __init__(
        self,
        width: float,
        height: float,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        *,
        margin: tuple[float, float, float, float] = (36.0, 10.0, 22.0, 10.0),
    ) -> None:
        self.width, self.height = float(width), float(height)
        self.left, self.top, self.bottom, self.right = margin
        self.xlim, self.ylim = xlim, ylim
        if xlim[1] == xlim[0]:
            self.xlim = (xlim[0] - 1.0, xlim[0] + 1.0)
        if ylim[1] == ylim[0]:
            self.ylim = (ylim[0] - 1.0, ylim[0] + 1.0)

    @property
    def plot_w(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> float:
        return self.height - self.top - self.bottom

    def x(self, v: float) -> float:
        return self.left + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.plot_w

    def y(self, v: float) -> float:
        return self.top + (self.ylim[1] - v) / (self.ylim[1] - self.ylim[0]) * self.plot_h

    def frame(
        self,
        x_title: str = "",
        y_title: str = "",
        *,
        x_ticks: Sequence[tuple[float, str]] = (),
        y_ticks: Sequence[tuple[float, str]] = (),
    ) -> list[cv.Shape]:
        out: list[cv.Shape] = [
            cv.Rect(
                self.left, self.top, self.plot_w, self.plot_h, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)
            )
        ]
        for v, text in x_ticks:
            px = self.x(v)
            out.append(
                cv.Line(
                    px,
                    self.top + self.plot_h,
                    px,
                    self.top + self.plot_h + 4,
                    paint=_stroke(ft.Colors.OUTLINE, 1.0),
                )
            )
            out.append(_label(px - 12, self.top + self.plot_h + 6, text))
        for v, text in y_ticks:
            py = self.y(v)
            out.append(cv.Line(self.left - 4, py, self.left, py, paint=_stroke(ft.Colors.OUTLINE, 1.0)))
            out.append(_label(2, py - 6, text))
        if x_title:
            out.append(_label(self.left + self.plot_w / 2 - 30, self.height - 12, x_title))
        if y_title:
            out.append(
                cv.Text(4, 2, y_title, style=ft.TextStyle(size=10, color=ft.Colors.ON_SURFACE_VARIANT))
            )
        return out

    def polyline(
        self,
        xs: Sequence[float] | np.ndarray,
        ys: Sequence[float] | np.ndarray,
        color: str,
        width: float = 1.5,
        dash: Sequence[float] | None = None,
    ) -> cv.Path:
        elements: list[cv.Path.PathElement] = []
        for k, (xv, yv) in enumerate(zip(xs, ys)):
            px, py = self.x(float(xv)), self.y(float(yv))
            elements.append(cv.Path.MoveTo(px, py) if k == 0 else cv.Path.LineTo(px, py))
        return cv.Path(elements, paint=_stroke(color, width, dash))

    def arrow(
        self, x0: float, y0: float, x1: float, y1: float, color: str, width: float = 2.0
    ) -> list[cv.Shape]:
        px0, py0, px1, py1 = self.x(x0), self.y(y0), self.x(x1), self.y(y1)
        out: list[cv.Shape] = [cv.Line(px0, py0, px1, py1, paint=_stroke(color, width))]
        dx, dy = px1 - px0, py1 - py0
        length = math.hypot(dx, dy)
        if length > 1e-9:
            ux, uy = dx / length, dy / length
            size = min(7.0, length / 2)
            out.append(
                cv.Path(
                    [
                        cv.Path.MoveTo(px1, py1),
                        cv.Path.LineTo(px1 - size * ux + 0.5 * size * uy, py1 - size * uy - 0.5 * size * ux),
                        cv.Path.LineTo(px1 - size * ux - 0.5 * size * uy, py1 - size * uy + 0.5 * size * ux),
                        cv.Path.Close(),
                    ],
                    paint=_fill(color),
                )
            )
        return out


def _ticks(lo: float, hi: float, n: int, fmt: str) -> list[tuple[float, str]]:
    return [(v, format(v, fmt)) for v in np.linspace(lo, hi, n)]


# ---- Level 4 drawings --------------------------------------------------------------------------------------------------------------------


CANVAS_TEXT = 10
"""The smallest text a drawing sets: its labels are the drawing's only words, so they keep the app's micro size."""
_GLYPH_WIDTH = 0.56
"""A character's mean advance over its font size in the app's sans: the width a drawing reserves for a label."""


def _text_width(text: str, size: int = CANVAS_TEXT) -> float:
    return _GLYPH_WIDTH * size * len(text)


class _Box(NamedTuple):
    """A label's rectangle on a canvas."""

    left: float
    top: float
    right: float
    bottom: float

    def overlaps(self, other: _Box) -> bool:
        return (
            self.left < other.right
            and other.left < self.right
            and self.top < other.bottom
            and other.top < self.bottom
        )


def _backed_label(
    x: float, y: float, text: str, *, color: str, size: int = CANVAS_TEXT
) -> tuple[_Box, list[cv.Shape]]:
    """A label on a patch of the card's surface, so that a line passing under it does not strike it through."""
    box = _Box(x - 3.0, y - 1.0, x + _text_width(text, size) + 3.0, y + size + 4.0)
    patch = cv.Rect(
        box.left,
        box.top,
        box.right - box.left,
        box.bottom - box.top,
        border_radius=ft.BorderRadius.all(3),
        paint=_fill(ft.Colors.SURFACE_CONTAINER_LOWEST),
    )
    return box, [patch, _label(x, y, text, size=size, color=color)]


def lifetime_text(seconds: float) -> str:
    """A level's lifetime in the unit a reader would say it in: 8.07 ns, 52.7 ms, 3.2 yr."""
    if seconds >= 3.156e7:
        return f"{seconds / 3.156e7:.3g} yr"
    if seconds >= 86400.0:
        return f"{seconds / 86400.0:.3g} d"
    return fmt_number(seconds, "s")


_TERM_COLUMNS = "SPDFG"
"""A Grotrian diagram's columns: one per orbital letter, in order."""


def _term_column(name: str) -> int:
    """The column of a level: its orbital letter's (S1/2 in the first, P1/2 and P3/2 in the second); a level in another
    coupling scheme (3D[3/2]1/2) takes a column of its own after them."""
    if "[" in name:
        return len(_TERM_COLUMNS)
    return next((_TERM_COLUMNS.index(ch) for ch in name if ch in _TERM_COLUMNS), len(_TERM_COLUMNS))


def level_diagram(
    levels: Sequence[tuple[str, float, float | None]],
    transitions: Sequence[tuple[str, str, str, float]],
    qubit_level: str,
    *,
    room: float,
) -> ft.Control:
    """Fine-structure levels as a Grotrian diagram: one column per orbital letter, the ground level at the foot, the
    excited levels above a break in the energy axis and spaced by their energies among themselves (labelled not to scale:
    the optical gap would otherwise press them into one line), the tabulated transitions as arrows with their
    wavelengths, the qubit level marked. At most 560 px wide, narrower when ``room`` (the card's width) is."""
    width, height = min(560.0, room), 320.0
    names = [n for n, _e, _t in levels]
    energies = np.asarray([e for _n, e, _t in levels], dtype=float)
    ax = Axes(width, height, (0.0, 1.0), (0.0, 1.06), margin=(26.0, 22.0, 10.0, 10.0))
    excited = energies > 0.0
    lo, hi = (
        (float(np.min(energies[excited])), float(np.max(energies[excited]))) if excited.any() else (0.0, 1.0)
    )
    # the ground level at 0, the excited levels from 0.32 to 1 by energy: the break between is the labelled exception
    scaled = np.where(excited, 0.32 + 0.68 * (energies - lo) / (hi - lo if hi > lo else 1.0), 0.0)
    # a fine-structure doublet is a few THz apart in hundreds: each level sits at least a label's height above the one
    # below it (the order kept), and the excited levels are fitted back under the top
    gap = 26.0 / ax.plot_h * 1.06
    order = np.argsort(scaled, kind="stable")
    for below, above in zip(order, order[1:]):
        scaled[above] = max(scaled[above], scaled[below] + gap)
    top = float(np.max(scaled)) if scaled.size else 1.0
    if top > 1.0:
        scaled = np.where(excited, 0.32 + (scaled - 0.32) * 0.68 / (top - 0.32), scaled)
    used = sorted({_term_column(n) for n in names})
    slot = 1.0 / max(len(used), 1)
    bar = 0.62 * slot
    left_of = {n: used.index(_term_column(n)) * slot + 0.5 * (slot - bar) for n in names}
    centre_of = {n: left_of[n] + 0.5 * bar for n in names}
    shapes: list[cv.Shape] = []
    # the energy axis with its break between the ground level and the excited ones
    x_axis = ax.left - 14.0
    y_break = ax.y(0.16)
    shapes += [
        cv.Line(x_axis, ax.y(0.0), x_axis, y_break + 5.0, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        cv.Line(x_axis, y_break - 5.0, x_axis, ax.y(1.03), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
    ]
    shapes += [
        cv.Line(x_axis - 5.0, y + 2.5, x_axis + 5.0, y - 2.5, paint=_stroke(ft.Colors.OUTLINE, 1.0))
        for y in (y_break - 5.0, y_break + 5.0)
    ]
    boxes: list[_Box] = []
    # drawn in this order: the bars, the arrows over them, then every label on its patch over both
    level_labels: list[cv.Shape] = []
    for i, name in enumerate(names):
        x0, y = ax.x(left_of[name]), ax.y(float(scaled[i]))
        is_qubit = name == qubit_level
        color = ft.Colors.PRIMARY if is_qubit else ft.Colors.ON_SURFACE
        shapes.append(
            cv.Line(x0, y, ax.x(left_of[name] + bar), y, paint=_stroke(color, 3.0 if is_qubit else 2.0))
        )
        life = levels[i][2]
        text = name + (" · qubit" if is_qubit else "") + ("" if life is None else f" · {lifetime_text(life)}")
        box, drawn = _backed_label(x0, y - CANVAS_TEXT - 7.0, text, color=color)
        boxes.append(box)
        level_labels += drawn
    labels: list[cv.Shape] = []
    for _name, lower, upper, lam in transitions:
        if lower not in left_of or upper not in left_of:
            continue
        i_lo, i_up = names.index(lower), names.index(upper)
        x0, y0 = ax.x(centre_of[lower]), ax.y(float(scaled[i_lo]))
        x1, y1 = ax.x(centre_of[upper]), ax.y(float(scaled[i_up]))
        color = ft.Colors.TERTIARY if lam < 500e-9 else ft.Colors.SECONDARY
        shapes.extend(
            ax.arrow(centre_of[lower], float(scaled[i_lo]), centre_of[upper], float(scaled[i_up]), color, 1.2)
        )
        text = f"{lam * 1e9:.1f} nm"
        # the wavelength rides its arrow at the first place along it that is clear of every label already set
        for frac in (0.5, 0.36, 0.64, 0.24, 0.76, 0.14, 0.86):
            x, y = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
            box, drawn = _backed_label(
                x - 0.5 * _text_width(text), y - 0.5 * CANVAS_TEXT - 2.0, text, color=color
            )
            if not any(box.overlaps(other) for other in boxes):
                break
        boxes.append(box)
        labels += drawn
    shapes += level_labels + labels
    return _captioned(
        cv.Canvas(shapes, width=width, height=height),
        "Energy not to scale: the axis is broken. Arrows: tabulated E1 lines.",
        width,
    )


def sublevel_fan(rows: Sequence[tuple[str, float]], *, title: str, room: float) -> ft.Control:
    """Zeeman sublevels of one level as bars at their energies relative to the lowest (to scale within the level), each
    named above its bar with its offset beneath; at most 560 px wide, narrower when ``room`` is."""
    width, height = min(560.0, room), 170.0
    if not rows:
        return ft.Container()
    energies = np.asarray([e for _l, e in rows], dtype=float)
    lo, hi = float(np.min(energies)), float(np.max(energies))
    ax = Axes(width, height, (0.0, 1.0), (lo, hi if hi > lo else lo + 1.0), margin=(8.0, 38.0, 22.0, 8.0))
    shapes: list[cv.Shape] = [_label(8, 2, title)]
    slot = 1.0 / len(rows)
    for k, (label, e) in enumerate(sorted(rows, key=lambda r: r[1])):
        x0 = (k + 0.08) * slot
        px, py = ax.x(x0), ax.y(e)
        shapes.append(cv.Line(px, py, ax.x(x0 + 0.84 * slot), py, paint=_stroke(ft.Colors.PRIMARY, 2.0)))
        shapes.append(_label(px, py - CANVAS_TEXT - 6.0, label, color=ft.Colors.ON_SURFACE))
        shapes.append(_label(px, py + 5.0, f"{(e - lo) / 1e6:+.3f} MHz"))
    return cv.Canvas(shapes, width=width, height=height)


def crystal_picture(
    positions_m: np.ndarray,
    eigenvector: np.ndarray | None,
    e_hat: tuple[float, float, float] | None,
    *,
    species: Sequence[str] = (),
    title: str = "",
    room: float,
) -> ft.Control:
    """The chain to scale along the trap axis (z) with the mode's displacement pattern as arrows: c_{i,m} along e_hat,
    drawn in the z (horizontal) and x (vertical) plane; a y component is written beside the arrow. At most 520 px wide,
    narrower when ``room`` is."""
    width, height = min(520.0, room), 150.0
    pos = np.asarray(positions_m, dtype=float)
    z = pos[:, 2]
    span = float(np.max(z) - np.min(z)) if z.size > 1 else 1e-6
    span = max(span, 1e-7)
    ax = Axes(
        width,
        height,
        (float(np.min(z)) - 0.35 * span, float(np.max(z)) + 0.35 * span),
        (-1.0, 1.0),
        margin=(12.0, 18.0, 12.0, 12.0),
    )
    shapes: list[cv.Shape] = [
        cv.Line(
            ax.left, ax.y(0.0), ax.left + ax.plot_w, ax.y(0.0), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)
        )
    ]
    if title:
        shapes.append(_label(8, 2, title))
    for i in range(pos.shape[0]):
        shapes.append(cv.Circle(ax.x(float(z[i])), ax.y(0.0), 7, paint=_fill(ft.Colors.PRIMARY)))
        name = species[i] if i < len(species) else ""
        shapes.append(_label(ax.x(float(z[i])) - 10, ax.y(0.0) + 12, f"ion {i} {name}"))
        if eigenvector is not None and e_hat is not None:
            c = float(eigenvector[i])
            dz, dx, dy = c * e_hat[2], c * e_hat[0], c * e_hat[1]
            end_z = float(z[i]) + dz * 0.3 * span
            shapes.extend(ax.arrow(float(z[i]), 0.0, end_z, dx * 0.85, ft.Colors.TERTIARY, 2.5))
            if abs(dy) > 1e-6:
                shapes.append(
                    _label(
                        ax.x(float(z[i])) + 8,
                        ax.y(0.0) - 26,
                        f"y: {dy:+.2f}",
                        color=ft.Colors.TERTIARY,
                    )
                )
    return _captioned(
        cv.Canvas(shapes, width=width, height=height),
        f"Chain to scale over {span * 1e6:.2f} µm; arrows: the mode's eigenvector, on its own scale.",
        width,
    )


def stability_diagram(
    q: np.ndarray,
    a_lower: np.ndarray,
    a_upper: np.ndarray,
    points: Sequence[tuple[str, float, float]],
    *,
    room: float,
) -> ft.Control:
    """The first stability region of the Mathieu equation from the monodromy boundary, with the device's (q, a) per axis;
    at most 420 px wide, narrower when ``room`` is."""
    width, height = min(420.0, room), 260.0
    ok = np.isfinite(a_lower) & np.isfinite(a_upper)
    qs, lo, hi = q[ok], a_lower[ok], a_upper[ok]
    ymin = float(min(np.min(lo), -0.05)) if lo.size else -0.5
    ymax = float(max(np.max(hi), 0.1)) if hi.size else 1.0
    ax = Axes(
        width,
        height,
        (0.0, float(np.max(q)) if q.size else 1.0),
        (ymin - 0.05, ymax + 0.05),
        margin=(44.0, 12.0, 26.0, 12.0),
    )
    shapes: list[cv.Shape] = ax.frame(
        "q",
        "a",
        x_ticks=_ticks(0.0, float(np.max(q)) if q.size else 1.0, 5, ".2f"),
        y_ticks=_ticks(ymin, ymax, 5, ".2f"),
    )
    if qs.size > 1:
        poly: list[cv.Path.PathElement] = [cv.Path.MoveTo(ax.x(float(qs[0])), ax.y(float(lo[0])))]
        for xv, yv in zip(qs[1:], lo[1:]):
            poly.append(cv.Path.LineTo(ax.x(float(xv)), ax.y(float(yv))))
        for xv, yv in zip(qs[::-1], hi[::-1]):
            poly.append(cv.Path.LineTo(ax.x(float(xv)), ax.y(float(yv))))
        poly.append(cv.Path.Close())
        shapes.append(
            cv.Path(
                poly,
                paint=ft.Paint(
                    color=ft.Colors.with_opacity(0.25, ft.Colors.PRIMARY), style=ft.PaintingStyle.FILL
                ),
            )
        )
        shapes.append(ax.polyline(qs, lo, ft.Colors.PRIMARY, 1.5))
        shapes.append(ax.polyline(qs, hi, ft.Colors.PRIMARY, 1.5))
        shapes.append(
            _label(
                ax.x(float(qs[len(qs) // 3])),
                ax.y(float(0.5 * (lo[len(qs) // 3] + hi[len(qs) // 3]))) - 6,
                "stable: 0 < β < 1",
                color=ft.Colors.PRIMARY,
            )
        )
    for name, qv, av in points:
        px, py = ax.x(qv), ax.y(av)
        shapes.append(cv.Circle(px, py, 6, paint=_fill(ft.Colors.SURFACE_CONTAINER_LOWEST)))
        shapes.append(cv.Circle(px, py, 4.5, paint=_fill(ft.Colors.TERTIARY)))
        shapes.append(_label(px + 8, py - 6, f"{name}: q = {qv:.3f}, a = {av:.4f}", color=ft.Colors.TERTIARY))
    return cv.Canvas(shapes, width=width, height=height)


def beam_geometry(
    positions_m: np.ndarray,
    beams: Sequence[tuple[int, tuple[float, float, float], tuple[float, float, float], str, str]],
    field_direction: tuple[float, float, float],
    *,
    room: float,
) -> ft.Control:
    """Top view (z along the chain horizontal, x vertical): ions to scale, each beam as an arrow along k_hat through its
    pointing, the quantization axis B drawn from the origin; a beam with a y component says so. At most 520 px wide,
    narrower when ``room`` is."""
    width, height = min(520.0, room), 260.0
    pos = np.asarray(positions_m, dtype=float)
    span = max(float(np.max(pos[:, 2]) - np.min(pos[:, 2])) if pos.shape[0] > 1 else 1e-6, 1e-6)
    half = 1.6 * span
    cz = float(np.mean(pos[:, 2]))
    ax = Axes(
        width,
        height,
        (cz - half, cz + half),
        (-half * height / width, half * height / width),
        margin=(12.0, 12.0, 12.0, 12.0),
    )
    shapes: list[cv.Shape] = []
    # the beams' names, drawn last on their patches so that no beam strikes one through
    names: list[cv.Shape] = []
    for i in range(pos.shape[0]):
        shapes.append(
            cv.Circle(ax.x(float(pos[i, 2])), ax.y(float(pos[i, 0])), 6, paint=_fill(ft.Colors.PRIMARY))
        )
        shapes.append(_label(ax.x(float(pos[i, 2])) - 8, ax.y(float(pos[i, 0])) + 9, f"ion {i}"))
    for k, k_hat, pointing, color, label in beams:
        z0, x0 = float(pointing[2]), float(pointing[0])
        length = 0.9 * half
        start_z, start_x = z0 - k_hat[2] * length, x0 - k_hat[0] * length
        shapes.extend(
            ax.arrow(
                start_z, start_x, z0 + 0.35 * k_hat[2] * length, x0 + 0.35 * k_hat[0] * length, color, 1.5
            )
        )
        tag = label + (f" (y {k_hat[1]:+.2f})" if abs(k_hat[1]) > 1e-6 else "")
        # beside the beam's tail where the tail is in the drawing, else where the beam enters it (a beam's tail lies
        # outside a drawing scaled to the chain), on the inner side of that edge; neighbours' names take three lines
        px, py = ax.x(start_z), min(max(ax.y(start_x), 0.0), height)
        stagger = (CANVAS_TEXT + 2.0) * (k % 3)
        label_y = py + 3.0 + stagger if py < 0.5 * height else py - CANVAS_TEXT - 5.0 - stagger
        label_x = min(max(px + 4.0, 2.0), width - _text_width(tag) - 2.0)
        names += _backed_label(label_x, label_y, tag, color=color)[1]
    bx, bz = float(field_direction[0]), float(field_direction[2])
    shapes.extend(
        ax.arrow(
            cz + 0.6 * half,
            -0.9 * half * height / width,
            cz + 0.6 * half + 0.25 * half * bz,
            -0.9 * half * height / width + 0.25 * half * bx,
            ft.Colors.ON_SURFACE,
            2.0,
        )
    )
    # the field's name beside the middle of its arrow, clear of the beams' names along the lower edge
    shapes.append(
        _label(
            ax.x(cz + 0.6 * half + 0.125 * half * bz) + 8.0,
            ax.y(-0.9 * half * height / width + 0.125 * half * bx) - 0.5 * CANVAS_TEXT,
            f"B (y {field_direction[1]:+.2f})",
        )
    )
    shapes += names
    return _captioned(
        cv.Canvas(shapes, width=width, height=height),
        f"From above (z along the chain, x up), to scale over {span * 1e6:.1f} µm; beams along k.",
        width,
    )


@dataclass(frozen=True)
class PhaseLoop:
    """One curve of the phase-space drawing: ``color_index`` picks the mode's colour, ``dashed`` marks the pair's second ion."""

    label: str
    alpha: np.ndarray
    note: str
    color_index: int = 0
    dashed: bool = False


def phase_space(loops: Sequence[PhaseLoop], *, width: float = 420.0, height: float = 300.0) -> ft.Control:
    """Spin-branch loops alpha_im(t) in the complex plane: a closed loop returns to its start (the filled dot); the end is a
    ring. One colour per mode; the pair's second ion dashed (its loop coincides with the first ion's on an in-phase mode and
    mirrors it on an out-of-phase one). The key stands under the drawing, where it covers no loop."""
    palette = list(theme.series())
    filled = [np.asarray(lp.alpha, dtype=complex) for lp in loops if np.asarray(lp.alpha).size]
    allv = np.concatenate(filled) if filled else np.zeros(1, complex)
    r = float(max(np.max(np.abs(allv.real)), np.max(np.abs(allv.imag)), 1e-3)) * 1.15
    ax = Axes(width, height, (-r, r), (-r, r), margin=(44.0, 20.0, 34.0, 12.0))
    shapes: list[cv.Shape] = ax.frame(x_ticks=_ticks(-r, r, 3, ".2f"), y_ticks=_ticks(-r, r, 3, ".2f"))
    # the titles clear of the tick labels: Im α over the frame's corner, Re α under the middle of the axis
    shapes.append(_label(ax.left, 2.0, "Im α"))
    shapes.append(_label(ax.left + ax.plot_w / 2 - 14.0, height - 15.0, "Re α"))
    shapes.append(
        cv.Line(ax.x(-r), ax.y(0.0), ax.x(r), ax.y(0.0), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0))
    )
    shapes.append(
        cv.Line(ax.x(0.0), ax.y(-r), ax.x(0.0), ax.y(r), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0))
    )
    key: list[tuple[ft.Control, str]] = []
    for lp in loops:
        a = np.asarray(lp.alpha, dtype=complex)
        if a.size == 0:
            continue
        color = palette[lp.color_index % len(palette)]
        shapes.append(ax.polyline(a.real, a.imag, color, 1.8, dash=(6.0, 4.0) if lp.dashed else None))
        shapes.append(
            cv.Circle(ax.x(float(a.real[0])), ax.y(float(a.imag[0])), 4, paint=_fill(ft.Colors.ON_SURFACE))
        )
        shapes.append(
            cv.Circle(ax.x(float(a.real[-1])), ax.y(float(a.imag[-1])), 4, paint=_stroke(color, 2.0))
        )
        key.append(
            (legend_swatch(color, line=True), f"{lp.label}{' (dashed)' if lp.dashed else ''}: {lp.note}")
        )
    return ft.Column(
        [
            cv.Canvas(shapes, width=width, height=height),
            legend(key, notes=["filled dot: start", "ring: end; a closed loop ends on its start"]),
        ],
        spacing=6,
        tight=True,
    )


# ---- charts with flet-charts -------------------------------------------------------------------------------------------------------------

_cap = ft.BorderRadius.only(top_left=3, top_right=3)
"""A bar's rounded data-end, square at the baseline (the dataviz mark spec)."""


def _tooltip_ink() -> ft.TextStyle:
    """The ink of a chart tooltip: the inverse surface's own, as on every other tooltip of the app. The library's default
    is the series colour, which no single background carries at 4.5:1 for every series; the text names its series
    instead (no meaning by colour alone)."""
    return ft.TextStyle(
        size=theme.SIZE_SMALL, weight=ft.FontWeight.W_500, color=ft.Colors.ON_INVERSE_SURFACE, height=1.35
    )


_TOOLTIP_RADIUS = ft.BorderRadius.all(6)
_TOOLTIP_PADDING = ft.Padding.symmetric(horizontal=10, vertical=6)
TOOLTIP_MAX_WIDTH = 260.0
"""A chart tooltip wraps here; it shifts to stay inside its chart rather than being cut by the card's edge."""


def nice_step(span: float, intervals: int) -> float:
    """1, 2 or 5 times a power of ten: the round step that divides ``span`` into the count nearest ``intervals``."""
    power = 10.0 ** math.floor(math.log10(span / intervals))
    return min((m * power for m in (1.0, 2.0, 5.0, 10.0)), key=lambda step: abs(span / step - intervals))


def _tick_text(value: float, step: float) -> str:
    """A tick at the decimals its step needs, so the ticks of one axis read as one column (0.00, 0.05, 0.10)."""
    decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    text = f"{value:.{decimals}f}"
    return text.replace("-", "", 1) if float(text) == 0.0 else text


class TickAxis(NamedTuple):
    """One axis of a line chart: where it starts and ends, and its ticks as (value, text)."""

    lo: float
    hi: float
    ticks: list[tuple[float, str]]
    step: float


def linear_axis(lo: float, hi: float, intervals: int, *, snap: bool) -> TickAxis:
    """Ticks at whole multiples of a nice step. ``snap`` widens the axis to the ticks around the data (the value axis, so
    the top gridline is a labelled one); without it the axis is the data's own extent (the time axis)."""
    if hi <= lo:
        # one value (a flat envelope): the three round ticks around it, at the scale of the value itself
        step = nice_step(abs(lo) or 1.0, 4)
        below = math.floor(lo / step)
        start = (below if lo > below * step else below - 1) * step
        return TickAxis(
            start,
            start + 2 * step,
            [(start + k * step, _tick_text(start + k * step, step)) for k in range(3)],
            step,
        )
    step = nice_step(hi - lo, intervals)
    first, last = math.floor(lo / step + 1e-9), math.ceil(hi / step - 1e-9)
    if snap:
        lo, hi = first * step, last * step
    values = [k * step for k in range(first, last + 1) if lo - 1e-9 * step <= k * step <= hi + 1e-9 * step]
    return TickAxis(lo, hi, [(v, _tick_text(v, step)) for v in values], step)


def decade_axis(lo: float, hi: float, intervals: int, *, snap: bool) -> TickAxis:
    """A log axis over log10 values: a tick per whole decade (every second or third on a long axis), printed as the power
    of ten it is."""
    first, last = math.floor(lo + 1e-9), math.ceil(hi - 1e-9)
    if last <= first:
        last = first + 1
    if snap:
        lo, hi = float(first), float(last)
    step = max(1, math.ceil((last - first) / intervals))
    decades = [d for d in range(first, last + 1) if lo - 1e-9 <= d <= hi + 1e-9 and d % step == 0]
    return TickAxis(lo, hi, [(float(d), power_of_ten(d)) for d in decades], float(step))


def _tick_labels(axis: TickAxis) -> list[fc.ChartAxisLabel]:
    return [
        fc.ChartAxisLabel(
            value=v,
            label=ft.Text(text, size=theme.SIZE_MICRO, color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True),
        )
        for v, text in axis.ticks
    ]


def legend_swatch(color: ft.ColorValue, *, line: bool = False, outlined: bool = False) -> ft.Control:
    """A legend's key: a short stroke for a line, a square for a bar; the outlined square is the companion bar's wash."""
    if line:
        return ft.Container(width=14, height=3, bgcolor=color, border_radius=ft.BorderRadius.all(2))
    return ft.Container(
        width=11,
        height=11,
        bgcolor=ft.Colors.with_opacity(0.12, color) if outlined else color,
        border=ft.Border.all(1.5, color) if outlined else None,
        border_radius=ft.BorderRadius.all(3),
    )


def legend(items: Sequence[tuple[ft.Control, str]], notes: Sequence[str] = ()) -> ft.Control:
    """A chart's legend on one wrapping line: each key with its name, then any notes (a marked value, a threshold)."""
    entries: list[ft.Control] = [
        ft.Row(
            [swatch, ft.Text(label, size=theme.SIZE_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=5,
            # tight, or each entry takes a whole line of the wrapping row it sits in
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        for swatch, label in items
    ]
    entries += [ft.Text(note, size=theme.SIZE_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT) for note in notes]
    return ft.Row(
        entries, wrap=True, spacing=14, run_spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER
    )


def line_chart(
    series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    x_title: str,
    y_title: str,
    height: float = 180.0,
    log_y: bool = False,
    log_x: bool = False,
    markers: Sequence[tuple[float, str]] = (),
) -> ft.Control:
    """Time series or curves on gridlines at round values. A log axis is drawn over log10 values and ticked at whole
    decades printed as powers of ten (the chart library has no log scale); a tiny or huge linear axis is drawn in units of
    a power of ten that its title names. The value axis's title stands above the chart, upright, and a hovered point says
    its series and its value as the screens print one."""
    palette = list(theme.series())
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _label, x, y in series:
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if log_x:
            keep = xs > 0
            xs, ys = xs[keep], ys[keep]
        if log_y:
            keep = ys > 0
            xs, ys = xs[keep], ys[keep]
        # (the abscissa as drawn, the ordinate as drawn, the ordinate as it is, which the tooltip prints)
        prepared.append((np.log10(xs) if log_x else xs, np.log10(ys) if log_y else ys, ys))
    drawn = [(xs, ys, true) for xs, ys, true in prepared if ys.size]
    if not drawn:
        return ft.Text("nothing to plot", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
    # a linear axis over a tiny or huge range is drawn in units of a power of ten, named beside the title, so that its
    # ticks stay short (a log axis already carries its scale)
    scale_note = ""
    top = max(float(np.nanmax(np.abs(ys))) for _xs, ys, _true in drawn)
    if not log_y and math.isfinite(top) and top > 0.0 and (top < 1e-2 or top >= 1e4):
        exponent = int(math.floor(math.log10(top)))
        prepared = [(xs, ys / 10.0**exponent, true) for xs, ys, true in prepared]
        drawn = [(xs, ys, true) for xs, ys, true in prepared if ys.size]
        scale_note = f" (× {power_of_ten(exponent)})"
    x_lo = min(float(np.nanmin(xs)) for xs, _ys, _true in drawn)
    x_hi = max(float(np.nanmax(xs)) for xs, _ys, _true in drawn)
    y_lo = min(float(np.nanmin(ys)) for _xs, ys, _true in drawn)
    y_hi = max(float(np.nanmax(ys)) for _xs, ys, _true in drawn)
    x_axis = (decade_axis if log_x else linear_axis)(x_lo, x_hi, 7, snap=False)
    y_axis = (decade_axis if log_y else linear_axis)(y_lo, y_hi, 4, snap=True)
    ink = _tooltip_ink()
    data = [
        fc.LineChartData(
            points=[
                fc.LineChartDataPoint(
                    x=float(a),
                    y=float(b),
                    tooltip=fc.LineChartDataPointTooltip(
                        text=f"{label}: {fmt_number(float(c))}", text_style=ink
                    ),
                )
                for a, b, c in zip(xs, ys, true)
            ],
            stroke_width=2,
            color=palette[k % len(palette)],
            rounded_stroke_cap=True,
        )
        for k, ((label, _x, _y), (xs, ys, true)) in enumerate(zip(series, prepared))
    ]
    chart: ft.Control = fc.LineChart(
        data_series=data,
        height=height,
        min_x=x_axis.lo,
        max_x=x_axis.hi,
        min_y=y_axis.lo,
        max_y=y_axis.hi,
        # the library asks for a label at every multiple of the spacing and draws the ones it is handed
        left_axis=fc.ChartAxis(labels=_tick_labels(y_axis), label_spacing=y_axis.step, label_size=44),
        bottom_axis=fc.ChartAxis(
            title=_axis_title(x_title), labels=_tick_labels(x_axis), label_spacing=x_axis.step, label_size=22
        ),
        horizontal_grid_lines=fc.ChartGridLines(
            interval=y_axis.step, color=ft.Colors.OUTLINE_VARIANT, width=1
        ),
        tooltip=fc.LineChartTooltip(
            bgcolor=ft.Colors.INVERSE_SURFACE,
            border_radius=_TOOLTIP_RADIUS,
            padding=_TOOLTIP_PADDING,
            max_width=TOOLTIP_MAX_WIDTH,
            fit_inside_horizontally=True,
            fit_inside_vertically=True,
        ),
        interactive=True,
    )
    key = legend(
        [
            (legend_swatch(palette[k % len(palette)], line=True), label)
            for k, (label, _x, _y) in enumerate(series)
        ],
        notes=[text for _v, text in markers],
    )
    return ft.Column([_axis_title(y_title + scale_note), chart, key], spacing=6)


@dataclass(frozen=True)
class Bar:
    """One bar of a group: its height, its tooltip, and whether it is the outlined companion of a filled bar (the target
    beside the simulated, the start beside the end)."""

    value: float
    tooltip: str
    outlined: bool = False
    color: ft.ColorValue = ft.Colors.PRIMARY


# The charts never set ``expand``: Flet wraps a control whose ``expand`` is set (False included) in a Flutter ``Expanded``
# when its parent is a Row or Column, which a wrapping Row (a ``Wrap``) cannot hold (a release build draws the grey error
# box) and a Column in a scroll view has no height to share out; a chart sizes itself to its box, so a chart without a width
# fills the width of its column at its ``height``.


def bar_chart(
    groups: Sequence[tuple[str, Sequence[Bar]]],
    *,
    bar_width: float,
    height: float,
    width: float | None = None,
    max_y: float | None = None,
    y_ticks: Sequence[float] = (),
    label_every: int = 1,
    x_title: str = "",
    y_title: str = "",
    on_tap: Callable[[int], None] | None = None,
    interactive: bool = True,
) -> ft.Control:
    """Grouped bars, thin with a rounded data end, on hairline gridlines (the dataviz mark specs); labelled below by group
    (every ``label_every``-th), on the left at ``y_ticks`` when given, the value axis's title upright above the chart;
    ``on_tap`` receives a tapped group's index."""
    ink = _tooltip_ink()
    chart_groups = [
        fc.BarChartGroup(
            x=k,
            rods=[
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=b.value,
                    width=bar_width,
                    color=ft.Colors.with_opacity(0.12, b.color) if b.outlined else b.color,
                    border_side=ft.BorderSide(1.5, b.color) if b.outlined else None,
                    tooltip=fc.BarChartRodTooltip(text=b.tooltip, text_style=ink),
                    border_radius=_cap,
                )
                for b in bars
            ],
            spacing=2,
        )
        for k, (_label, bars) in enumerate(groups)
    ]

    def on_event(e: fc.BarChartEvent) -> None:
        if on_tap is not None and e.type == fc.ChartEventType.TAP_UP and e.group_index is not None:
            on_tap(int(e.group_index))

    if not y_ticks:
        # no ticks named: round ones over the bars, in the charts' own label style rather than the library's
        top = (
            max_y
            if max_y is not None
            else max((b.value for _label, bars in groups for b in bars), default=1.0)
        )
        axis = linear_axis(0.0, top if top > 0.0 else 1.0, 4, snap=True)
        y_ticks = [v for v, _text in axis.ticks if max_y is None or v <= max_y * (1.0 + 1e-9)]
        if max_y is None:
            max_y = axis.hi
    y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else None
    chart: ft.Control = fc.BarChart(
        groups=chart_groups,
        bottom_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(
                    value=k, label=ft.Text(label, size=theme.SIZE_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT)
                )
                for k, (label, _bars) in enumerate(groups)
                if k % label_every == 0
            ],
            label_size=24,
            title=_axis_title(x_title) if x_title else None,
        ),
        left_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(
                    value=v,
                    label=ft.Text(f"{v:g}", size=theme.SIZE_MICRO, color=ft.Colors.ON_SURFACE_VARIANT),
                )
                for v in y_ticks
            ],
            # the library asks for a label at every multiple of the spacing and draws the ones it is handed
            label_spacing=y_step,
            label_size=40,
        ),
        horizontal_grid_lines=fc.ChartGridLines(interval=y_step, color=ft.Colors.OUTLINE_VARIANT, width=1),
        max_y=max_y,
        min_y=0.0,
        height=height,
        width=width,
        tooltip=fc.BarChartTooltip(
            bgcolor=ft.Colors.INVERSE_SURFACE,
            border_radius=_TOOLTIP_RADIUS,
            padding=_TOOLTIP_PADDING,
            max_width=TOOLTIP_MAX_WIDTH,
            fit_inside_horizontally=True,
            fit_inside_vertically=True,
        ),
        interactive=interactive,
        on_event=on_event if on_tap is not None else None,
    )
    return ft.Column([_axis_title(y_title), chart], spacing=6) if y_title else chart


def two_histograms(bright: np.ndarray, dark: np.ndarray, threshold: float | None) -> ft.Control:
    """Bright and dark count distributions side by side per photon number, the threshold named."""
    n = max(bright.size, dark.size)
    # the last populated photon number of either distribution (the bright one reaches further), plus a margin; at least
    # eight bins so a narrow pair still reads as a histogram
    last = max((int(np.flatnonzero(a > 1e-6).max(initial=-1)) for a in (bright, dark)), default=-1)
    n_show = int(min(n, max(last + 3, 8))) if n else 1
    groups = []
    for k in range(n_show):
        pb = float(bright[k]) if k < bright.size else 0.0
        pd = float(dark[k]) if k < dark.size else 0.0
        groups.append(
            (
                str(k),
                [
                    Bar(pb, f"bright ion: P({k}) = {fmt_number(pb)}"),
                    Bar(pd, f"dark ion: P({k}) = {fmt_number(pd)}", color=ft.Colors.TERTIARY),
                ],
            )
        )
    key = legend(
        [(legend_swatch(ft.Colors.PRIMARY), "bright ion"), (legend_swatch(ft.Colors.TERTIARY), "dark ion")],
        notes=[] if threshold is None else [f"called bright above {threshold:g} counts"],
    )
    return ft.Column(
        [
            bar_chart(
                groups,
                bar_width=9,
                height=200,
                label_every=max(1, n_show // 10),
                x_title="photons counted",
                y_title="probability",
            ),
            key,
        ],
        spacing=6,
    )


def fock_bars(start: np.ndarray, end: np.ndarray) -> ft.Control:
    """P(n) at the start (outlined) and the end (filled) of a step; a distribution given as an EMPTY array is not known
    and gets no bar, rather than a row of zeros that would read as a measured vacuum."""
    groups = []
    for k in range(min(8, max(start.size, end.size))):
        bars = []
        if start.size:
            a = float(start[k]) if k < start.size else 0.0
            bars.append(Bar(a, f"start: P({k}) = {fmt_number(a)}", outlined=True, color=ft.Colors.TERTIARY))
        if end.size:
            b = float(end[k]) if k < end.size else 0.0
            bars.append(Bar(b, f"end: P({k}) = {fmt_number(b)}"))
        groups.append((str(k), bars))
    return bar_chart(
        groups, bar_width=10, height=150, max_y=1.0, y_ticks=(0.0, 0.5, 1.0), x_title="vibration number n"
    )


def bloch_disc(
    ion: int, vector: tuple[float, float, float], *, target: tuple[float, float, float] | None
) -> ft.Control:
    """One ion's Bloch vector (<X>, <Y>, <Z>) as an arrow in the x-z disc (the equator horizontal, |0> up), the y component
    written beside it and the arrow as long as the vector (a shrunken arrow is mixture or entanglement); the target vector,
    when given, dashed."""
    size = 120.0
    x, y, z = (float(v) for v in vector)
    r = size / 2.0 - 8.0
    cx, cy = size / 2.0, size / 2.0
    shapes: list[cv.Shape] = [
        cv.Circle(cx, cy, r, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        cv.Line(cx - r, cy, cx + r, cy, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 0.6, [3, 3])),
        cv.Line(cx, cy - r, cx, cy + r, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 0.6, [3, 3])),
        _label(cx - 7, -2.0, "|0⟩"),
        _label(cx - 7, size - 12.0, "|1⟩"),
        _label(size - 15.0, cy - 7, "+x"),
    ]
    ax = Axes(size, size, (-1.0, 1.0), (-1.0, 1.0), margin=(8.0, 8.0, 8.0, 8.0))
    if target is not None:
        tx, _ty, tz = (float(v) for v in target)
        if math.hypot(tx, tz) > 1e-6:
            shapes.append(
                cv.Line(
                    ax.x(0.0), ax.y(0.0), ax.x(tx), ax.y(tz), paint=_stroke(ft.Colors.TERTIARY, 1.5, [4, 3])
                )
            )
    length = math.sqrt(x * x + y * y + z * z)
    if math.hypot(x, z) > 1e-6:
        shapes.extend(ax.arrow(0.0, 0.0, x, z, ft.Colors.PRIMARY, 2.5))
    shapes.append(cv.Circle(cx, cy, 2.5, paint=_fill(ft.Colors.ON_SURFACE)))
    canvas = cv.Canvas(shapes, width=size, height=size)
    return ft.Column(
        [
            canvas,
            ft.Text(f"ion {ion}", size=12, weight=ft.FontWeight.W_600),
            ft.Text(
                f"length {length:.3f} · ⟨Y⟩ {y:+.2f}",
                size=theme.SIZE_CAPTION,
                color=ft.Colors.ON_SURFACE_VARIANT,
            ),
        ],
        spacing=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )
