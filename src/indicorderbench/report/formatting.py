"""Number formatting shared by the text and HTML reports."""

from __future__ import annotations

DASH = "—"


def pct(x: float | None) -> str:
    """A rate as a percentage with one decimal; a dash when there is no rate."""
    return DASH if x is None else f"{x * 100:.1f}%"


def signed_pct(x: float | None) -> str:
    """A rate difference with an explicit sign (``+12.5%``, ``-50.0%``)."""
    if x is None:
        return DASH
    text = f"{x * 100:+.1f}%"
    return "+0.0%" if text == "-0.0%" else text


def ms(x: float | None) -> str:
    """Milliseconds rounded to a whole number."""
    return DASH if x is None else f"{x:.0f} ms"
