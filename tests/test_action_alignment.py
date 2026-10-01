from all22.action_alignment import ActionUnit, align_actions_to_pbp, is_filmed_event
from all22.models import PlayByPlay


def _play(ordinal: int, description: str, kind: str = "run") -> PlayByPlay:
    return PlayByPlay("g", ordinal, 1, None, None, 1, 10, None, description, kind, kind in {"run", "pass"})


def test_timeouts_are_ignored_but_false_starts_remain():
    assert not is_filmed_event(_play(1, "Timeout #1 by PHI", "excluded"))
    assert is_filmed_event(_play(2, "False Start on PHI. No Play", "excluded"))


def test_alignment_has_explicit_gaps_and_does_not_force_incompatible_event():
    plays = [_play(1, "Runner left tackle", "run"), _play(2, "Punter punts 50 yards", "punt")]
    units = [ActionUnit("pair1", "a1", 6, .9, "run"),
             ActionUnit("pair2", "a2", 5, .9, "run")]
    result = align_actions_to_pbp(plays, units)
    assert result[0].operation == "match"
    assert any(item.operation == "missing_film" and item.play.ordinal == 2 for item in result)
    assert any(item.operation == "extra_film" and item.unit.pair_id == "pair2" for item in result)


def test_uncertain_duration_is_marked_for_review():
    result = align_actions_to_pbp([_play(1, "Quarterback pass", "pass")],
                                  [ActionUnit("p", "a", 13, .5)])
    assert result[0].operation == "match"
    assert result[0].status == "review"
