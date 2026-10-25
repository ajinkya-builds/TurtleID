"""Multi-stage high-accuracy TurtleID stack (offline batch and online runtime)."""

__all__ = ["StackV2Predictor"]


def __getattr__(name: str):
    if name == "StackV2Predictor":
        from core.stack_v2.runtime import StackV2Predictor

        return StackV2Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
