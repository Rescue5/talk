"""Computer-vision utilities vendored for backend inference."""

__all__ = [
    "MobileSamRefiner",
    "load_sulfide_config",
    "make_sulfide_overlay",
    "segment_sulfides",
]


def __getattr__(name: str):
    if name == "MobileSamRefiner":
        from .sulfide_candidates import MobileSamRefiner

        return MobileSamRefiner
    if name == "load_sulfide_config":
        from .sulfide_candidates import load_sulfide_config

        return load_sulfide_config
    if name == "segment_sulfides":
        from .sulfide_candidates import segment_sulfides

        return segment_sulfides
    if name == "make_sulfide_overlay":
        from .sulfide_candidates import make_overlay

        return make_overlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
