"""Computer-vision utilities for ore microscopy analysis."""

from .post_segformer import ProcessingState, TalcCVPipeline

__all__ = [
    "MobileSamRefiner",
    "ProcessingState",
    "TalcCVPipeline",
    "make_sulfide_overlay",
    "segment_sulfides",
]


def __getattr__(name: str):
    if name == "MobileSamRefiner":
        from .sulfide_candidates import MobileSamRefiner

        return MobileSamRefiner
    if name == "segment_sulfides":
        from .sulfide_candidates import segment_sulfides

        return segment_sulfides
    if name == "make_sulfide_overlay":
        from .sulfide_candidates import make_overlay

        return make_overlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
