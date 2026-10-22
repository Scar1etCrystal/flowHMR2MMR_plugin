"""Reusable algorithms for MoCap Doctor.

Blender-dependent modules are intentionally not imported here, which keeps
the inclusive range utilities usable from ordinary Python.
"""

from .ranges import (
    clip_ranges,
    diff_ranges,
    frames_to_ranges,
    intersect_ranges,
    normalize_range,
    normalize_ranges,
    ranges_to_frames,
    subtract_ranges,
    symmetric_difference_ranges,
    union_ranges,
)


__all__ = [
    "clip_ranges",
    "diff_ranges",
    "frames_to_ranges",
    "intersect_ranges",
    "normalize_range",
    "normalize_ranges",
    "ranges_to_frames",
    "subtract_ranges",
    "symmetric_difference_ranges",
    "union_ranges",
]

