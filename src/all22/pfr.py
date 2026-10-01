from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup, Comment

from .db import transaction
from .models import PlayByPlay


SCRIMMAGE_RE = re.compile(r"\b(pass|sacked|scramble|left end|left tackle|left guard|up the middle|right guard|right tackle|right end)\b", re.I)
EXCLUDED_RE = re.compile(r"\b(punt|field goal|kickoff|kneel|spike|timeout|end of|no play)\b", re.I)


def classify_play(description: str) -> tuple[str, bool]:
    text = " ".join(description.split())
    if EXCLUDED_RE.search(text):
        return "excluded", False
    if re.search(r"\bpass\b|\bsacked\b|\bscramble\b", text, re.I):
        return "pass", True
    if SCRIMMAGE_RE.search(text):
        return "run", True
    return "administrative", False


def _int(value: str) -> Optional[int]:
    match = re.search(r"\d+", value or "")
    return int(match.group()) if match else None


def parse_html(path: Path, game_id: str) -> List[PlayByPlay]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    fragments = [soup]
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "play_by_play" in comment or "pbp" in comment:
            fragments.append(BeautifulSoup(str(comment), "lxml"))

    table = None
    for fragment in fragments:
        table = fragment.find("table", id=re.compile(r"play_by_play|pbp", re.I))
        if table:
            break
    if not table:
        raise ValueError("No PFR play-by-play table found")

    plays: List[PlayByPlay] = []
    quarter = None
    for row in table.select("tbody tr"):
        if "thead" in (row.get("class") or []):
            continue
        cells = {cell.get("data-stat"): cell.get_text(" ", strip=True) for cell in row.select("th[data-stat],td[data-stat]")}
        description = cells.get("detail") or cells.get("description") or ""
        if not description:
            continue
        quarter = _int(cells.get("quarter", "")) or quarter
        play_type, eligible = classify_play(description)
        plays.append(PlayByPlay(
            game_id=game_id,
            ordinal=len(plays) + 1,
            quarter=quarter,
            clock=cells.get("time") or cells.get("game_time"),
            possession=cells.get("team") or cells.get("possession"),
            down=_int(cells.get("down", "")),
            distance=_int(cells.get("yds_to_go", "") or cells.get("distance", "")),
            yard_line=cells.get("location") or cells.get("yard_line"),
            description=description,
            play_type=play_type,
            eligible=eligible,
            source_row_id=row.get("id"),
        ))
    return plays


def store_plays(db_path: Path, plays: Iterable[PlayByPlay]) -> int:
    rows = list(plays)
    if not rows:
        return 0
    game_id = rows[0].game_id
    with transaction(db_path) as connection:
        connection.execute("INSERT OR IGNORE INTO games(game_id) VALUES(?)", (game_id,))
        connection.executemany(
            """INSERT OR REPLACE INTO pbp_plays
            (game_id,ordinal,quarter,clock,possession,down_no,distance,yard_line,description,play_type,eligible,source_row_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(p.game_id, p.ordinal, p.quarter, p.clock, p.possession, p.down, p.distance,
              p.yard_line, p.description, p.play_type, int(p.eligible), p.source_row_id) for p in rows],
        )
    return len(rows)
