from all22.video import MotionSample, action_windows_from_motion


def test_action_windows_do_not_cross_camera_cuts():
    samples = []
    for index in range(100):
        score = 20 if 10 <= index <= 30 or 45 <= index <= 70 else 2
        samples.append(MotionSample(index / 5, score, index == 40))
    windows = action_windows_from_motion(samples, sample_fps=5)
    assert len(windows) == 2
    assert windows[0].end_s < 8
    assert windows[1].start_s > 8
