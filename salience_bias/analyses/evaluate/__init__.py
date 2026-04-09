from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from salience_bias.analyses.evaluate.api import ConvexMixtureResult

__all__ = [
    "ConvexMixtureResult",
    "evaluate_convex_mixture",
    "evaluate_multi",
    "evaluate_single",
    "make_gaze_map",
]


def __getattr__(name: str) -> Any:
    if name == "evaluate_single":
        from salience_bias.analyses.evaluate.api import evaluate_single

        return evaluate_single
    if name == "evaluate_convex_mixture":
        from salience_bias.analyses.evaluate.api import evaluate_convex_mixture

        return evaluate_convex_mixture
    if name == "evaluate_multi":
        from salience_bias.analyses.evaluate.api import evaluate_multi

        return evaluate_multi
    if name == "make_gaze_map":
        from salience_bias.analyses.evaluate.core.gaze import make_gaze_map

        return make_gaze_map
    if name == "ConvexMixtureResult":
        from salience_bias.analyses.evaluate.api import ConvexMixtureResult

        return ConvexMixtureResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
