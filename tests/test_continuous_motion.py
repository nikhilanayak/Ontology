from all22.video import MotionSample, action_windows_from_motion


def test_action_windows_bridge_a_short_camera_edit_during_action():
    samples = []
    for index in range(100):
        score = 20 if 10 <= index <= 30 or 45 <= index <= 70 else 2
        samples.append(MotionSample(index / 5, score, index == 40))
    windows = action_windows_from_motion(samples, sample_fps=5)
    assert len(windows) == 1
    assert windows[0].start_s <= 2
    assert windows[0].end_s >= 14


def test_action_windows_remain_separate_across_long_quiet_period():
    samples = []
    for index in range(130):
        score = 20 if 10 <= index <= 30 or 60 <= index <= 85 else 2
        samples.append(MotionSample(index / 5, score, index == 45))
    windows = action_windows_from_motion(samples, sample_fps=5)
    assert len(windows) == 2
    assert windows[0].end_s < windows[1].start_s
