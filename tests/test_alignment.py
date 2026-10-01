from all22.alignment import align_monotonic, group_angle_sources, pair_angles
from all22.models import Angle, Clip, PlayByPlay


def play(ordinal, kind):
    return PlayByPlay("g", ordinal, 1, "10:00", "PHI", 1, 10, "PHI 20", kind, kind, True)


def test_pair_and_align():
    clips = [
        Clip("g", "s1", Angle.SIDELINE, 0, 12), Clip("g", "e1", Angle.ENDZONE, 12, 24),
        Clip("g", "s2", Angle.SIDELINE, 24, 34), Clip("g", "e2", Angle.ENDZONE, 34, 44),
    ]
    pairs = pair_angles(clips)
    result = align_monotonic([play(1, "pass"), play(2, "run")], pairs)
    assert len(pairs) == 2
    assert [item.operation for item in result] == ["match", "match"]


def test_group_preserves_multiple_sources_per_play():
    clips = [
        Clip("g", "s1a", Angle.SIDELINE, 0, 5),
        Clip("g", "s1b", Angle.SIDELINE, 5, 12),
        Clip("g", "e1a", Angle.ENDZONE, 12, 20),
        Clip("g", "e1b", Angle.ENDZONE, 20, 24),
        Clip("g", "s2", Angle.SIDELINE, 24, 34),
        Clip("g", "e2", Angle.ENDZONE, 34, 44),
    ]
    groups = group_angle_sources(clips)
    assert [[source.clip_id for source in group.sources] for group in groups] == [
        ["s1a"], ["s1b", "e1a", "e1b"], ["s2", "e2"]
    ]
    assert len(groups[1].sideline_sources) == 1
    assert len(groups[1].endzone_sources) == 2


def test_alignment_can_include_filmed_special_teams_plays():
    kickoff = PlayByPlay("g", 1, 1, "15:00", "PHI", None, 0, "PHI 35",
                         "Kickoff", "kickoff", False)
    run = play(2, "run")
    groups = pair_angles([
        Clip("g", "s1", Angle.SIDELINE, 0, 10), Clip("g", "e1", Angle.ENDZONE, 10, 20),
        Clip("g", "s2", Angle.SIDELINE, 20, 30), Clip("g", "e2", Angle.ENDZONE, 30, 40),
    ])
    result = align_monotonic([kickoff, run], groups, include_all_filmed_plays=True)
    assert [item.play.play_type for item in result] == ["kickoff", "run"]
