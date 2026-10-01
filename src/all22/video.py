from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .models import Angle, Clip


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class MotionSample:
    timestamp: float
    score: float
    camera_cut: bool = False


@dataclass(frozen=True)
class ActionWindow:
    start_s: float
    end_s: float
    peak_score: float


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate",
         "-of", "json", str(path)], check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    stream = next(item for item in payload["streams"] if item["codec_type"] == "video")
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return VideoInfo(path, float(payload["format"]["duration"]), int(stream["width"]), int(stream["height"]), numerator / denominator)


def sampled_frames(path: Path, sample_fps: float = 2.0) -> Iterator[tuple[float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    every = max(1, round(source_fps / sample_fps))
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every == 0:
                yield frame_index / source_fps, frame
            frame_index += 1
    finally:
        capture.release()


def camera_compensated_motion(path: Path, sample_fps: float = 5.0, start_s: float = 0.0,
                              end_s: Optional[float] = None) -> List[MotionSample]:
    """Measure residual on-field motion after removing camera pan/zoom.

    This scans the continuous film rather than assuming a scene cut is a play
    boundary. Sparse field features estimate a global affine camera transform;
    only residual changes inside the green field hull contribute to the score.
    """
    info = probe(path)
    stop = min(end_s, info.duration) if end_s is not None else info.duration
    if start_s < 0 or stop <= start_s or sample_fps <= 0:
        raise ValueError("Invalid continuous-motion scan interval")
    width, height = 320, 180
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start_s),
               "-t", str(stop - start_s), "-i", str(path), "-vf",
               f"fps={sample_fps},scale={width}:{height}", "-an", "-pix_fmt", "bgr24",
               "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg motion-analysis pipe")
    samples: List[MotionSample] = []
    previous = None
    previous_histogram = None
    frame_bytes = width * height * 3
    frame_index = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = start_s + frame_index / sample_fps
            frame_index += 1
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(histogram, histogram)
            if previous is None:
                previous = gray
                previous_histogram = histogram
                continue
            points = cv2.goodFeaturesToTrack(previous, maxCorners=400, qualityLevel=.01,
                                             minDistance=4, blockSize=5)
            matrix = None
            inlier_ratio = 0.0
            if points is not None and len(points) >= 12:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                valid = status.reshape(-1).astype(bool) if status is not None else np.zeros(len(points), bool)
                if moved is not None and valid.sum() >= 8:
                    matrix, inliers = cv2.estimateAffinePartial2D(
                        points[valid], moved[valid], method=cv2.RANSAC, ransacReprojThreshold=2.0,
                    )
                    if inliers is not None:
                        inlier_ratio = float(inliers.mean())
            if matrix is None:
                matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            stabilized = cv2.warpAffine(previous, matrix, (width, height), flags=cv2.INTER_LINEAR)
            difference = cv2.absdiff(stabilized, gray)
            green = cv2.inRange(hsv, (25, 35, 25), (105, 255, 255))
            field = cv2.morphologyEx(green, cv2.MORPH_CLOSE, close_kernel)
            field = cv2.dilate(field, close_kernel, iterations=1)
            values = difference[field > 0]
            raw_score = float(np.percentile(values, 90)) if values.size >= 1000 else 0.0
            # A low affine inlier ratio is common during a live play because
            # many tracked corners belong to players.  It is therefore not a
            # scene-cut signal.  Color-distribution discontinuity is much more
            # selective and does not split a play merely because the camera
            # starts panning.
            histogram_distance = cv2.compareHist(
                previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
            )
            camera_cut = histogram_distance >= .72
            samples.append(MotionSample(timestamp, 0.0 if camera_cut else raw_score, camera_cut))
            previous = gray
            previous_histogram = histogram
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if process.stderr:
            process.stderr.close()
        if return_code:
            raise RuntimeError(stderr.strip() or "FFmpeg continuous-motion scan failed")
    return samples


def action_windows_from_motion(samples: List[MotionSample], sample_fps: float = 5.0,
                               minimum_s: float = 2.0, maximum_s: float = 18.0) -> List[ActionWindow]:
    if len(samples) < 10:
        return []
    scores = np.asarray([sample.score for sample in samples], dtype=float)
    smoothed = np.convolve(scores, np.ones(5) / 5, mode="same")
    usable = smoothed[np.asarray([not sample.camera_cut for sample in samples])]
    if not usable.size:
        return []
    baseline = float(np.percentile(usable, 35))
    mad = float(np.median(np.abs(usable - np.median(usable)))) or 1.0
    high_threshold = max(float(np.percentile(usable, 67)), baseline + 1.5 * mad)
    low_threshold = max(float(np.percentile(usable, 42)), baseline + .35 * mad)
    seeds = smoothed >= high_threshold
    active = seeds.copy()
    # Grow strong motion through weaker intervals. This captures the whole
    # snap-to-whistle burst rather than just its fastest two or three seconds.
    for index in np.flatnonzero(seeds):
        cursor = index - 1
        while cursor >= 0 and not samples[cursor].camera_cut and smoothed[cursor] >= low_threshold:
            active[cursor] = True
            cursor -= 1
        cursor = index + 1
        while cursor < len(active) and not samples[cursor].camera_cut and smoothed[cursor] >= low_threshold:
            active[cursor] = True
            cursor += 1
    # Bridge short lulls and one-frame edits within an action. A cut is useful
    # for suppressing its own discontinuity score, but must not itself become
    # a hard boundary: pans and zoom jumps occur during many long plays.
    maximum_gap = max(1, round(sample_fps * 3.0))
    for index in range(1, len(active) - 1):
        if active[index]:
            continue
        left = next((offset for offset in range(1, maximum_gap + 1)
                     if index - offset >= 0 and active[index - offset]), None)
        right = next((offset for offset in range(1, maximum_gap + 1)
                      if index + offset < len(active) and active[index + offset]), None)
        if left is not None and right is not None:
            active[index] = True
    windows: List[ActionWindow] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        stop = index + 1
        while stop < len(active) and active[stop]:
            stop += 1
        duration = samples[stop - 1].timestamp - samples[index].timestamp + 1 / sample_fps
        if minimum_s <= duration <= maximum_s:
            windows.append(ActionWindow(max(samples[0].timestamp, samples[index].timestamp - .6),
                                        samples[stop - 1].timestamp + .8,
                                        float(smoothed[index:stop].max())))
        index = stop
    return windows


def clips_from_motion_scan(path: Path, game_id: str, samples: List[MotionSample],
                           sample_fps: float = 5.0) -> List[Clip]:
    """Build camera sources and their live-action bounds from one full scan.

    A source is bounded by a genuine visual edit. Motion is then interpreted
    *within* that source, so the tail of the preceding angle cannot be joined
    to the next angle merely because both contain moving players.
    """
    info = probe(path)
    cuts: List[float] = []
    for sample in samples:
        if sample.camera_cut and (not cuts or sample.timestamp - cuts[-1] >= 1.5):
            cuts.append(sample.timestamp)
    boundaries = [0.0] + [value for value in cuts if 0 < value < info.duration] + [info.duration]
    usable = np.asarray([sample.score for sample in samples if not sample.camera_cut], dtype=float)
    activity_threshold = max(20.0, float(np.percentile(usable, 60))) if usable.size else 20.0
    minimum_run = max(2, round(sample_fps * .6))
    maximum_gap = max(1, round(sample_fps * 2.0))
    clips: List[Clip] = []

    for source_start, source_end in zip(boundaries, boundaries[1:]):
        if source_end - source_start < 3.0:
            continue
        source_samples = [sample for sample in samples
                          if source_start < sample.timestamp < source_end and not sample.camera_cut]
        snap = play_end = None
        confidence = 0.0
        angle = Angle.NON_PLAY
        if len(source_samples) >= 10:
            scores = np.asarray([sample.score for sample in source_samples], dtype=float)
            smoothed = np.convolve(scores, np.ones(5) / 5, mode="same")
            active = smoothed >= activity_threshold
            # Close brief tracking/camera-stabilization gaps, but never cross
            # the source boundary established by a hard edit.
            for index in range(1, len(active) - 1):
                if active[index]:
                    continue
                left = next((offset for offset in range(1, maximum_gap + 1)
                             if index - offset >= 0 and active[index - offset]), None)
                right = next((offset for offset in range(1, maximum_gap + 1)
                              if index + offset < len(active) and active[index + offset]), None)
                if left is not None and right is not None:
                    active[index] = True
            runs = []
            index = 0
            while index < len(active):
                if not active[index]:
                    index += 1
                    continue
                stop = index + 1
                while stop < len(active) and active[stop]:
                    stop += 1
                if stop - index >= minimum_run:
                    energy = float(smoothed[index:stop].sum())
                    runs.append((energy, index, stop))
                index = stop
            if runs:
                _, first, stop = max(runs)
                snap = max(source_start, source_samples[first].timestamp - .6)
                play_end = min(source_end, source_samples[stop - 1].timestamp + .8)
                if play_end - snap >= 1.5:
                    peak = float(smoothed[first:stop].max())
                    confidence = min(1.0, max(0.0, peak / max(activity_threshold * 3, 1.0)))
                    angle, angle_confidence = classify_angle(frame_at(path, (snap + play_end) / 2))
                    confidence = min(confidence, max(.05, angle_confidence))
                else:
                    snap = play_end = None
        clip_id = f"{game_id}:{len(clips) + 1:04d}"
        clips.append(Clip(game_id, clip_id, angle, source_start, source_end,
                          snap, play_end, confidence))
    return clips


def visual_cut_candidates(path: Path, sample_fps: float = 2.0, threshold: float = 0.48) -> List[float]:
    """Return high-confidence hard-cut candidates using HSV histogram distance."""
    cuts: List[float] = []
    previous = None
    for timestamp, frame in sampled_frames(path, sample_fps):
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        if previous is not None:
            distance = cv2.compareHist(previous, histogram, cv2.HISTCMP_BHATTACHARYYA)
            if distance >= threshold:
                cuts.append(timestamp)
        previous = histogram
    return cuts


def ffmpeg_scene_cuts(path: Path, sample_fps: float = 4.0, threshold: float = 0.10,
                      start_s: float = 0.0, end_s: Optional[float] = None) -> List[float]:
    """Detect source changes quickly; timestamps are absolute within the video."""
    info = probe(path)
    stop = min(end_s, info.duration) if end_s is not None else info.duration
    if start_s < 0 or stop <= start_s:
        raise ValueError("Invalid scene-detection interval")
    command = ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start_s), "-t", str(stop - start_s),
               "-i", str(path), "-vf", f"fps={sample_fps},select='gt(scene,{threshold})',showinfo",
               "-an", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg scene detection failed")
    relative = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    # Nearby detections are usually a transition/fade, not distinct sources.
    cuts: List[float] = []
    for value in relative:
        absolute = start_s + value
        if not cuts or absolute - cuts[-1] >= 1.5:
            cuts.append(absolute)
    return cuts


def clips_from_cuts(game_id: str, duration: float, cuts: List[float], minimum_s: float = 2.0,
                    start_s: float = 0.0, end_s: Optional[float] = None) -> List[Clip]:
    stop = min(end_s, duration) if end_s is not None else duration
    boundaries = [start_s] + sorted(value for value in cuts if start_s < value < stop) + [stop]
    clips = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < minimum_s:
            continue
        clip_id = f"{game_id}:{len(clips) + 1:04d}"
        clips.append(Clip(game_id, clip_id, Angle.UNKNOWN, start, end))
    return clips


def frame_at(path: Path, timestamp: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not decode {path} at {timestamp:.3f}s")
        return frame
    finally:
        capture.release()


def classify_angle(frame: np.ndarray) -> Tuple[Angle, float]:
    """Classify coaches-film angle from long, near-vertical painted yard lines.

    Counting long white segments is more stable than summing all edge lengths:
    hash marks and sidelines otherwise make wide sideline film look horizontal.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (180, 75, 255))
    lines = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=max(30, frame.shape[1] // 28),
                            minLineLength=max(60, frame.shape[1] // 16),
                            maxLineGap=max(20, frame.shape[1] // 38))
    if lines is None:
        return Angle.UNKNOWN, 0.0
    long_vertical = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 180
        angle = min(angle, 180 - angle)
        if angle >= 45 and abs(dy) >= frame.shape[0] * 0.40:
            long_vertical += 1
    # At 1080p, end-zone samples in the validation film produce 0–7 such
    # segments, while sideline samples produce 35–65. Scale for other heights.
    boundary = max(4, math.ceil(28 * frame.shape[0] / 1080))
    angle = Angle.SIDELINE if long_vertical >= boundary else Angle.ENDZONE
    confidence = min(1.0, abs(long_vertical - boundary) / max(boundary, 1))
    return angle, float(confidence)


def angle_transition_cuts(path: Path, sample_fps: float = 1.0, start_s: float = 0.0,
                          end_s: Optional[float] = None, stable_samples: int = 2) -> List[float]:
    """Find camera changes that color-histogram scene detection can miss."""
    info = probe(path)
    stop = min(end_s, info.duration) if end_s is not None else info.duration
    width, height = 480, 270
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start_s),
               "-t", str(stop - start_s), "-i", str(path), "-vf",
               f"fps={sample_fps},scale={width}:{height}", "-an", "-pix_fmt", "bgr24",
               "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg analysis pipe")
    observed: List[Tuple[float, Angle]] = []
    frame_bytes = width * height * 3
    frame_index = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            timestamp = start_s + frame_index / sample_fps
            value, confidence = classify_angle(frame)
            if value != Angle.UNKNOWN and confidence >= 0.30:
                observed.append((timestamp, value))
            frame_index += 1
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if process.stderr:
            process.stderr.close()
        if return_code:
            raise RuntimeError(stderr.strip() or "FFmpeg angle scan failed")
    cuts: List[float] = []
    committed = observed[0][1] if observed else Angle.UNKNOWN
    for index in range(len(observed) - stable_samples + 1):
        run = observed[index:index + stable_samples]
        candidate = run[0][1]
        if candidate != committed and all(item[1] == candidate for item in run):
            cuts.append(run[0][0])
            committed = candidate
    return cuts


def source_cut_candidates(path: Path, scene_fps: float = 4.0, scene_threshold: float = 0.10,
                          start_s: float = 0.0, end_s: Optional[float] = None) -> List[float]:
    cuts = ffmpeg_scene_cuts(path, scene_fps, scene_threshold, start_s, end_s)
    cuts.extend(angle_transition_cuts(path, 1.0, start_s, end_s))
    merged: List[float] = []
    for value in sorted(cuts):
        if not merged or value - merged[-1] >= 2.5:
            merged.append(value)
    return merged


def motion_boundaries(path: Path, start_s: float, end_s: float, sample_fps: float = 10.0) -> Tuple[Optional[float], Optional[float], float]:
    """Propose snap/end timestamps from sustained motion onset and decay."""
    capture = cv2.VideoCapture(str(path))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    every = max(1, round(source_fps / sample_fps))
    capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    previous = None
    scores, times = [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp >= end_s:
                break
            if index % every == 0:
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
                if previous is not None:
                    scores.append(float(cv2.absdiff(gray, previous).mean()))
                    times.append(timestamp)
                previous = gray
            index += 1
    finally:
        capture.release()
    if len(scores) < 20:
        return None, None, 0.0
    values = np.asarray(scores)
    baseline_window = values[:max(10, len(values) // 4)]
    median = float(np.median(baseline_window))
    mad = float(np.median(np.abs(baseline_window - median))) or 0.1
    threshold = median + max(1.0, 4.0 * mad)
    active = values > threshold
    snap_index = next((i for i in range(2, len(active) - 3) if active[i:i + 3].all()), None)
    if snap_index is None:
        return None, None, 0.0
    play_end_index = snap_index
    quiet_needed = max(8, round(sample_fps * 1.5))
    minimum_live_frames = max(20, round(sample_fps * 3.0))
    for i in range(snap_index + minimum_live_frames, len(active) - quiet_needed):
        if not active[i:i + quiet_needed].any():
            play_end_index = i
            break
    else:
        play_end_index = len(times) - 1
    confidence = min(1.0, max(0.0, (float(values[snap_index]) - threshold) / max(threshold, 1.0)))
    return times[snap_index], times[play_end_index], confidence


def analyze_clips(path: Path, game_id: str, cuts: List[float], start_s: float = 0.0,
                  end_s: Optional[float] = None) -> List[Clip]:
    info = probe(path)
    results = []
    for clip in clips_from_cuts(game_id, info.duration, cuts, start_s=start_s, end_s=end_s):
        sample_times = [clip.start_s + (clip.end_s - clip.start_s) * fraction for fraction in (.25, .5, .75)]
        votes = [classify_angle(frame_at(path, timestamp)) for timestamp in sample_times]
        scores = {candidate: sum(confidence for value, confidence in votes if value == candidate)
                  for candidate in (Angle.SIDELINE, Angle.ENDZONE)}
        angle = max(scores, key=scores.get)
        angle_confidence = scores[angle] / max(sum(scores.values()), 1e-9)
        snap, play_end, motion_confidence = motion_boundaries(path, clip.start_s, clip.end_s)
        results.append(Clip(clip.game_id, clip.clip_id, angle, clip.start_s, clip.end_s,
                            snap, play_end, min(angle_confidence, motion_confidence)))
    return results


def coalesce_short_fragments(clips: List[Clip], maximum_fragment_s: float = 5.0) -> List[Clip]:
    """Absorb short same-angle detector fragments into the neighboring source."""
    merged: List[Clip] = []
    for clip in clips:
        if (merged and merged[-1].angle == clip.angle and
                abs(merged[-1].end_s - clip.start_s) < 0.05 and
                (merged[-1].end_s - merged[-1].start_s <= maximum_fragment_s or
                 clip.end_s - clip.start_s <= maximum_fragment_s)):
            previous = merged.pop()
            merged.append(Clip(
                previous.game_id, previous.clip_id, previous.angle, previous.start_s, clip.end_s,
                previous.snap_s if previous.snap_s is not None else clip.snap_s,
                clip.play_end_s if clip.play_end_s is not None else previous.play_end_s,
                max(previous.confidence, clip.confidence),
            ))
        else:
            merged.append(clip)
    return merged
