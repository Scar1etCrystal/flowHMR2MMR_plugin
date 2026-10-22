"""Pure-Python operations on inclusive integer frame ranges."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

FrameRange: TypeAlias = tuple[int, int]
RangeLike: TypeAlias = Sequence[int]


def _bounds(
    frame_start: int | None,
    frame_end: int | None,
) -> tuple[int | None, int | None]:
    start = None if frame_start is None else int(frame_start)
    end = None if frame_end is None else int(frame_end)
    if start is not None and end is not None and start > end:
        raise ValueError(f"frame_start ({start}) is after frame_end ({end})")
    return start, end


def normalize_range(
    value: RangeLike,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> FrameRange | None:
    """Order and optionally clip one inclusive range.

    Reversed endpoints are accepted. ``None`` means the range lies completely
    outside the requested bounds.
    """

    if len(value) != 2:
        raise ValueError(f"a frame range needs two endpoints, got {value!r}")
    lower, upper = _bounds(frame_start, frame_end)
    a, b = int(value[0]), int(value[1])
    if b < a:
        a, b = b, a
    if lower is not None:
        a = max(a, lower)
    if upper is not None:
        b = min(b, upper)
    return None if a > b else (a, b)


def normalize_ranges(
    ranges: Iterable[RangeLike],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    merge_gap: int = 0,
    min_length: int = 1,
) -> list[FrameRange]:
    """Sort, clip, merge, and length-filter inclusive ranges.

    ``merge_gap=0`` merges touching ranges. ``merge_gap=1`` also bridges one
    unlisted frame, matching the validated contact detector.
    """

    merge_gap, min_length = int(merge_gap), int(min_length)
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    if min_length < 1:
        raise ValueError("min_length must be at least 1")
    items = [
        item
        for value in ranges
        if (
            item := normalize_range(
                value,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        )
        is not None
    ]
    items.sort()
    merged: list[list[int]] = []
    for a, b in items:
        if not merged or a > merged[-1][1] + merge_gap + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(a, b) for a, b in merged if b - a + 1 >= min_length]


def clip_ranges(
    ranges: Iterable[RangeLike],
    frame_start: int,
    frame_end: int,
    *,
    merge_gap: int = 0,
    min_length: int = 1,
) -> list[FrameRange]:
    return normalize_ranges(
        ranges,
        frame_start=frame_start,
        frame_end=frame_end,
        merge_gap=merge_gap,
        min_length=min_length,
    )


def union_ranges(
    *range_sets: Iterable[RangeLike],
    frame_start: int | None = None,
    frame_end: int | None = None,
    merge_gap: int = 0,
    min_length: int = 1,
) -> list[FrameRange]:
    return normalize_ranges(
        (item for values in range_sets for item in values),
        frame_start=frame_start,
        frame_end=frame_end,
        merge_gap=merge_gap,
        min_length=min_length,
    )


def intersect_ranges(
    left: Iterable[RangeLike],
    right: Iterable[RangeLike],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> list[FrameRange]:
    lhs = normalize_ranges(left, frame_start=frame_start, frame_end=frame_end)
    rhs = normalize_ranges(right, frame_start=frame_start, frame_end=frame_end)
    out: list[FrameRange] = []
    i = j = 0
    while i < len(lhs) and j < len(rhs):
        a1, b1 = lhs[i]
        a2, b2 = rhs[j]
        a, b = max(a1, a2), min(b1, b2)
        if a <= b:
            out.append((a, b))
        if b1 < b2:
            i += 1
        else:
            j += 1
    return out


def subtract_ranges(
    base: Iterable[RangeLike],
    remove: Iterable[RangeLike],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    min_length: int = 1,
) -> list[FrameRange]:
    """Subtract inclusive intervals, preserving exact boundary frames."""

    if int(min_length) < 1:
        raise ValueError("min_length must be at least 1")
    source = normalize_ranges(base, frame_start=frame_start, frame_end=frame_end)
    cuts = normalize_ranges(remove, frame_start=frame_start, frame_end=frame_end)
    out: list[FrameRange] = []
    cut_index = 0
    for start, end in source:
        cursor = start
        while cut_index < len(cuts) and cuts[cut_index][1] < cursor:
            cut_index += 1
        index = cut_index
        while index < len(cuts) and cuts[index][0] <= end:
            cut_start, cut_end = cuts[index]
            if cut_start > cursor:
                out.append((cursor, min(end, cut_start - 1)))
            cursor = max(cursor, cut_end + 1)
            if cursor > end:
                break
            index += 1
        if cursor <= end:
            out.append((cursor, end))
    return [item for item in out if item[1] - item[0] + 1 >= int(min_length)]


def symmetric_difference_ranges(
    left: Iterable[RangeLike],
    right: Iterable[RangeLike],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> list[FrameRange]:
    left = list(left)
    right = list(right)
    return union_ranges(
        subtract_ranges(left, right, frame_start=frame_start, frame_end=frame_end),
        subtract_ranges(right, left, frame_start=frame_start, frame_end=frame_end),
    )


def diff_ranges(
    before: Iterable[RangeLike],
    after: Iterable[RangeLike],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> dict[str, list[FrameRange]]:
    """Return the frames added, removed, and retained by an edit."""

    old = normalize_ranges(before, frame_start=frame_start, frame_end=frame_end)
    new = normalize_ranges(after, frame_start=frame_start, frame_end=frame_end)
    return {
        "added": subtract_ranges(new, old),
        "removed": subtract_ranges(old, new),
        "unchanged": intersect_ranges(old, new),
    }


def ranges_to_frames(ranges: Iterable[RangeLike]) -> set[int]:
    frames: set[int] = set()
    for a, b in normalize_ranges(ranges):
        frames.update(range(a, b + 1))
    return frames


def frames_to_ranges(
    frames: Iterable[int],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    merge_gap: int = 0,
    min_length: int = 1,
) -> list[FrameRange]:
    lower, upper = _bounds(frame_start, frame_end)
    values = sorted(
        {
            int(frame)
            for frame in frames
            if (lower is None or int(frame) >= lower)
            and (upper is None or int(frame) <= upper)
        }
    )
    if not values:
        return []
    raw: list[FrameRange] = []
    start = previous = values[0]
    for frame in values[1:]:
        if frame == previous + 1:
            previous = frame
        else:
            raw.append((start, previous))
            start = previous = frame
    raw.append((start, previous))
    return normalize_ranges(raw, merge_gap=merge_gap, min_length=min_length)

