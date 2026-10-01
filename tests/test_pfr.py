from pathlib import Path

from all22.pfr import classify_play, parse_html


def test_classification():
    assert classify_play("A.Player pass deep left") == ("pass", True)
    assert classify_play("A.Runner right tackle for 4 yards") == ("run", True)
    assert classify_play("Punt 52 yards") == ("excluded", False)


def test_comment_wrapped_pfr_table():
    plays = parse_html(Path(__file__).parent / "fixtures" / "pfr.html", "sample-game")
    assert len(plays) == 3
    assert [play.quarter for play in plays] == [1, 1, 1]
    assert [play.play_type for play in plays] == ["pass", "run", "excluded"]
    assert sum(play.eligible for play in plays) == 2
