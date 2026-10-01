from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Angle(str, Enum):
    SIDELINE = "sideline"
    ENDZONE = "endzone"
    NON_PLAY = "non_play"
    UNKNOWN = "unknown"


class PlayStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PlayByPlay:
    game_id: str
    ordinal: int
    quarter: Optional[int]
    clock: Optional[str]
    possession: Optional[str]
    down: Optional[int]
    distance: Optional[int]
    yard_line: Optional[str]
    description: str
    play_type: str
    eligible: bool
    source_row_id: Optional[str] = None


@dataclass(frozen=True)
class Clip:
    game_id: str
    clip_id: str
    angle: Angle
    start_s: float
    end_s: float
    snap_s: Optional[float] = None
    play_end_s: Optional[float] = None
    confidence: float = 0.0


@dataclass(frozen=True)
class ActionWindow:
    action_id: str
    clip_id: str
    action_order: int
    formation_start_s: Optional[float]
    snap_s: Optional[float]
    dead_s: Optional[float]
    playback_end_s: Optional[float]
    confidence: float = 0.0
    status: str = "candidate"


@dataclass(frozen=True)
class ActionPair:
    pair_id: str
    primary_action_id: str
    alternate_action_id: Optional[str]
    score: float
    status: str = "candidate"


@dataclass(frozen=True)
class TrackingSample:
    game_id: str
    play_id: str
    frame_id: int
    t: float
    nfl_id: Optional[str]
    team: Optional[str]
    jersey_number: Optional[int]
    x: float
    y: float
    speed: Optional[float] = None
    acceleration: Optional[float] = None
    direction: Optional[float] = None
    orientation: Optional[float] = None
    event: Optional[str] = None


@dataclass
class QualityReport:
    play_id: str
    status: PlayStatus = PlayStatus.PENDING
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.status = PlayStatus.REJECTED
        if reason not in self.reasons:
            self.reasons.append(reason)

    def accept(self) -> None:
        if self.reasons:
            raise ValueError("Cannot accept a play that has rejection reasons")
        self.status = PlayStatus.ACCEPTED

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value
