"""How a value, a caption and an axis are printed (DESIGN.md Sections 4 and 11): the one number format's edges, the
sentence case of captions, the round ticks of the charts and the words the numerics strip names an engine in."""

from __future__ import annotations

from typing import get_args

import pytest

from qutip_trap_app.record import FidelityLevelRun
from qutip_trap_app.viewmodel.catalogue import fmt_number, power_of_ten, split_power_of_ten, unit_text
from qutip_trap_app.views import theme
from qutip_trap_app.views.common import ENGINE_WORDS, sentence_case
from qutip_trap_app.views.drawing import decade_axis, lifetime_text, linear_axis, nice_step


def test_a_value_takes_the_prefix_of_what_it_prints() -> None:
    """A frequency a rounding error under 1 MHz prints at four digits as 1 MHz, not as 1000 kHz."""
    assert fmt_number(999_999.9999, "Hz") == "1 MHz"
    assert fmt_number(1_000_000.0, "Hz") == "1 MHz"
    assert fmt_number(999_400.0, "Hz") == "999.4 kHz"
    assert fmt_number(-999_999.9999, "Hz") == "-1 MHz"
    assert fmt_number(1.699e-6, "s") == "1.699 µs"


def test_a_power_of_ten_splits_around_its_exponent() -> None:
    """A stat tile sets the exponent itself, raised in the value's own face: the split gives it in plain digits."""
    assert power_of_ten(-4) == "10⁻⁴" and power_of_ten(12) == "10¹²"
    assert split_power_of_ten(fmt_number(4.972e-4)) == ("4.972 × 10", "-4", "")
    assert split_power_of_ten(fmt_number(-5.781e-4, "rad")) == ("-5.781 × 10", "-4", " rad")
    assert split_power_of_ten(fmt_number(6.428e-22)) == ("6.428 × 10", "-22", "")
    assert split_power_of_ten("147.1 kHz") == ("147.1 kHz", "", "")


def test_a_unit_prints_its_powers_as_powers() -> None:
    assert unit_text("Hz/G^2") == "Hz/G²" and unit_text("1/s") == "s⁻¹" and unit_text("Hz") == "Hz"
    assert fmt_number(621.7, "Hz/G^2") == "621.7 Hz/G²"


@pytest.mark.parametrize(
    ("caption", "printed"),
    [
        ("error of GPi2 on ion 0", "Error of GPi2 on ion 0"),
        ("mode 0: axial #1", "Mode 0: axial #1"),
        ("first-order field sensitivity", "First-order field sensitivity"),
        ("How long the gate takes", "How long the gate takes"),
        # a symbol keeps its case: capitalised it would name another quantity
        ("n' = 0, n = 0 (carrier)", "n' = 0, n = 0 (carrier)"),
        ("q0 bright read as dark", "q0 bright read as dark"),
        ("eta of the entangling pair", "eta of the entangling pair"),
        ("ms on ions 0 and 1", "ms on ions 0 and 1"),
        ("⟨n⟩ after Doppler cooling", "⟨n⟩ after Doppler cooling"),
    ],
)
def test_a_caption_is_in_sentence_case(caption: str, printed: str) -> None:
    assert sentence_case(caption) == printed


def test_a_chart_axis_ticks_at_round_values() -> None:
    assert nice_step(0.088, 4) == pytest.approx(0.02) and nice_step(100.0, 7) == pytest.approx(20.0)
    # the value axis widens to the ticks around the data, so the top gridline is a labelled one
    value = linear_axis(0.0, 0.088, 4, snap=True)
    assert (value.lo, value.hi) == pytest.approx((0.0, 0.1))
    assert [text for _v, text in value.ticks] == ["0.00", "0.02", "0.04", "0.06", "0.08", "0.10"]
    # the time axis keeps the data's own extent and ticks inside it
    time = linear_axis(5.4, 105.4, 7, snap=False)
    assert (time.lo, time.hi) == (5.4, 105.4)
    assert [text for _v, text in time.ticks] == ["20", "40", "60", "80", "100"]
    # a flat envelope gets the three round ticks around its value, the value between them
    flat = linear_axis(195.5, 195.5, 4, snap=True)
    assert [text for _v, text in flat.ticks] == ["150", "200", "250"] and flat.lo < 195.5 < flat.hi
    assert [text for _v, text in linear_axis(-0.6, 0.6, 4, snap=True).ticks][2] == "0.0", (
        "never a negative zero"
    )


def test_a_log_axis_ticks_at_whole_decades() -> None:
    """The chart library has no log scale: the axis is over log10 values, ticked at the decades and printed as them."""
    axis = decade_axis(2.0, 8.0, 7, snap=False)
    assert [text for _v, text in axis.ticks] == ["10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷", "10⁸"]
    snapped = decade_axis(-5.7, -5.1, 4, snap=True)
    assert (snapped.lo, snapped.hi) == (-6.0, -5.0)
    long = decade_axis(-12.0, 0.0, 4, snap=True)
    assert [v for v, _text in long.ticks] == [-12.0, -9.0, -6.0, -3.0, 0.0], (
        "every third decade on a long axis"
    )
    # the library asks for a label at every multiple of the spacing from zero: the ticks are those multiples
    assert all(v % long.step == 0 for v, _text in long.ticks)


def test_a_lifetime_reads_in_the_unit_it_is_said_in() -> None:
    assert lifetime_text(8.07e-9) == "8.07 ns" and lifetime_text(52.7e-3) == "52.7 ms"
    assert lifetime_text(9.96e7) == "3.16 yr" and lifetime_text(2.0 * 86400.0) == "2 d"


def test_the_strip_names_every_engine_in_words() -> None:
    """The numerics strip looks the run's level up by name: every level a record can carry has its words."""
    assert set(ENGINE_WORDS) == set(get_args(FidelityLevelRun))


def _luminance(colour: str) -> float:
    """WCAG 2.2 relative luminance of an ``#RRGGBB`` colour."""
    channels = [int(colour[k : k + 2], 16) / 255.0 for k in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("tokens", [theme.LIGHT, theme.DARK], ids=["light", "dark"])
def test_every_text_stands_at_aa_contrast_on_its_surface(tokens: theme.Tokens) -> None:
    """The pairs the screens set words in, in both schemes, at 4.5:1 (WCAG 2.2 AA): ink and muted ink on the page, the
    card and the inset; the accents a drawing writes labels in; a link; a chart tooltip's ink on the inverse surface (the
    library's default was the series colour, which no one background carries for every series); the status pills."""
    pairs = {
        "ink on the page": (tokens.on_surface, tokens.surface),
        "ink on a card": (tokens.on_surface, tokens.card),
        "muted ink on the page": (tokens.on_surface_variant, tokens.surface),
        "muted ink on a card": (tokens.on_surface_variant, tokens.card),
        "muted ink on an inset": (tokens.on_surface_variant, tokens.low),
        "a link on the page": (tokens.primary, tokens.surface),
        "a primary label on a card": (tokens.primary, tokens.card),
        "a secondary label on a card": (tokens.secondary, tokens.card),
        "a tertiary label on a card": (tokens.tertiary, tokens.card),
        "the primary button's words": (tokens.on_primary, tokens.primary),
        "a selection's words": (tokens.on_secondary_container, tokens.secondary_container),
        "a tooltip's ink": (tokens.on_inverse_surface, tokens.inverse_surface),
        "an error's words": (tokens.on_error_container, tokens.error_container),
    } | {f"the {kind} pill": (fg, bg) for kind, (bg, fg) in tokens.status.items()}
    failing = {
        name: round(_contrast(fg, bg), 2) for name, (fg, bg) in pairs.items() if _contrast(fg, bg) < 4.5
    }
    assert not failing, failing
