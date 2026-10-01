from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import List, Optional, Sequence, Tuple

from .models import Angle, Clip, PlayByPlay


@dataclass(frozen=True)
class ClipGroup:
    """Ordered, independent camera sources depicting one play."""

    sources: Tuple[Clip, ...]

    @property
    def sideline_sources(self) -> Tuple[Clip, ...]:
        return tuple(source for source in self.sources if source.angle == Angle.SIDELINE)

    @property
    def endzone_sources(self) -> Tuple[Clip, ...]:
        return tuple(source for source in self.sources if source.angle == Angle.ENDZONE)

    @property
    def sideline(self) -> Optional[Clip]:
        return self.sideline_sources[0] if self.sideline_sources else None

    @property
    def endzone(self) -> Optional[Clip]:
        return self.endzone_sources[0] if self.endzone_sources else None


@dataclass(frozen=True)
class Alignment:
    play: PlayByPlay
    source_group: Optional[ClipGroup]
    score: float
    operation: str

    @property
    def pair(self) -> Optional[ClipGroup]:
        """Compatibility alias for older callers."""
        return self.source_group


def group_angle_sources(clips: Sequence[Clip]) -> List[ClipGroup]:
    """Group the usual sideline→end-zone replay cadence without joining media.

    Non-field shots are ignored. Each sideline source opens a new play after an
    end-zone source has been observed; any following alternate/end-zone sources
    remain independently addressable inside the same group.
    """
    groups: List[ClipGroup] = []
    pending: List[Clip] = []
    for clip in sorted(clips, key=lambda value: value.start_s):
        if clip.angle not in (Angle.SIDELINE, Angle.ENDZONE):
            continue
        if clip.angle == Angle.SIDELINE:
            # The canonical source is sideline film. Every new sideline shot is
            # therefore a new play candidate; later alternate angles attach to it.
            if pending:
                groups.append(ClipGroup(tuple(pending)))
                pending = []
            pending.append(clip)
        elif pending:
            pending.append(clip)
    if pending:
        groups.append(ClipGroup(tuple(pending)))
    return groups


def pair_angles(clips: Sequence[Clip]) -> List[ClipGroup]:
    """Compatibility name; groups may contain more than two angle sources."""
    return group_angle_sources(clips)


def _match_cost(play: PlayByPlay, group: ClipGroup) -> float:
    live_durations = [
        (source.play_end_s - source.snap_s)
        for source in group.sources
        if source.snap_s is not None and source.play_end_s is not None and source.play_end_s > source.snap_s
    ]
    source_durations = [source.end_s - source.start_s for source in group.sources]
    durations = live_durations or source_durations
    duration = sorted(durations)[len(durations) // 2]
    expected = 13.0 if play.play_type == "pass" else 10.0
    angle_penalty = 0.0 if group.sideline_sources and group.endzone_sources else 1.0
    return min(4.0, abs(duration - expected) / 5.0) + angle_penalty


FILMED_PLAY_TYPES = {"pass", "run", "punt", "kickoff", "field_goal", "extra_point",
                     "qb_kneel", "qb_spike", "no_play"}


def align_monotonic(plays: Sequence[PlayByPlay], pairs: Sequence[ClipGroup], skip_play_cost: float = 5.0,
                    skip_pair_cost: float = 4.0, include_all_filmed_plays: bool = False) -> List[Alignment]:
    """Globally align ordered PFR plays and film pairs with explicit gaps."""
    eligible = [play for play in plays if play.eligible or
                (include_all_filmed_plays and play.play_type in FILMED_PLAY_TYPES)]
    n, m = len(eligible), len(pairs)
    costs = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    costs[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            current = costs[i][j]
            if current == inf:
                continue
            if i < n and current + skip_play_cost < costs[i + 1][j]:
                costs[i + 1][j] = current + skip_play_cost
                back[i + 1][j] = (i, j, "missing_clip")
            if j < m and current + skip_pair_cost < costs[i][j + 1]:
                costs[i][j + 1] = current + skip_pair_cost
                back[i][j + 1] = (i, j, "extra_clip")
            if i < n and j < m:
                candidate = current + _match_cost(eligible[i], pairs[j])
                if candidate < costs[i + 1][j + 1]:
                    costs[i + 1][j + 1] = candidate
                    back[i + 1][j + 1] = (i, j, "match")

    aligned: List[Alignment] = []
    i, j = n, m
    while i or j:
        previous = back[i][j]
        if previous is None:
            break
        pi, pj, operation = previous
        if operation == "match":
            aligned.append(Alignment(eligible[pi], pairs[pj], _match_cost(eligible[pi], pairs[pj]), operation))
        elif operation == "missing_clip":
            aligned.append(Alignment(eligible[pi], None, skip_play_cost, operation))
        i, j = pi, pj
    return list(reversed(aligned))
