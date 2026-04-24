"""Per-trial saccade kinematic extraction (Exp 22).

Implements §5.1–5.3 of `llmdocs/exp_22_design.md`:

    target-jump detection  →  eye-saccade onset/offset  →  landing-window
    kinematics  (amplitude, gain, peak velocity, latency, duration).

Pure functions, numpy-only for the hot paths. Import as:

    from utils.saccade_kinematics import (
        detect_target_jumps, detect_eye_saccade, extract_trial_kinematics,
        extract_kinematics_for_sequence,
    )

Conventions:
    - `pos` is a 1-D numpy array of eye position in degrees, sampled at
      `sample_rate` Hz.
    - Velocity is `np.gradient(pos) * sample_rate`, smoothed with a 3-sample
      moving average (§5.2).
    - Upward vertical is the only direction used by Exp 22 (C1); the
      extractor itself is direction-agnostic (signed gain), but the
      orchestrator calls it only for positive vertical jumps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (match design §5)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 120.0           # Hz
DEFAULT_TARGET_JUMP_THR = 5.0         # degrees — §5.1
DEFAULT_V_ONSET = 30.0                # deg/s   — §5.2
DEFAULT_V_OFFSET = 20.0               # deg/s   — §5.2
DEFAULT_OFFSET_CONSEC = 3             # samples — hysteresis (§5.2)
DEFAULT_MIN_DUR_SAMPLES = 3           # samples — 25 ms floor (§5.2)
DEFAULT_MAX_DUR_SAMPLES = 24          # samples — 200 ms ceiling (§5.2)
DEFAULT_W_MAX_SAMPLES = 120           # samples — 1000 ms response window (§5.2)
DEFAULT_W_LAND_SAMPLES = 18           # samples — 150 ms landing window (§5.3)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrialKinematics:
    """Per-trial, per-eye kinematic features extracted from one target-jump.

    All fields are NaN for rejected trials (no saccade found, duration out of
    bounds, overflow past end of signal). NaN is not zero (C10).
    """
    trial_idx: int          # sequential valid-trial index within sequence
    t_jump: int             # sample index of the target jump
    A_target: float         # target jump amplitude (signed)
    amplitude: float        # A — signed, landing-window peak displacement
    gain: float             # G = A / A_target — dimensionless, signed
    peak_velocity: float    # V_p = max(|v|) in landing window — deg/s
    latency: float          # TT = (s − t_jump) / SAMPLE_RATE  — seconds
    duration: float         # D = 2 * (t_pv − s) / SAMPLE_RATE  — seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _smoothed_velocity(pos: np.ndarray, sample_rate: float) -> np.ndarray:
    """Forward-difference velocity then 3-sample moving average (§5.2).

    Matches the design's `v(t) = diff(eye_pos) * SAMPLE_RATE` literally and
    pads the first sample with 0 so `len(v) == len(pos)`.
    """
    dif = np.diff(pos) * sample_rate
    v = np.concatenate([[0.0], dif])
    k = np.ones(3) / 3.0
    return np.convolve(v, k, mode="same")


def detect_target_jumps(
    target: np.ndarray,
    direction: str = "positive",
    threshold: float = DEFAULT_TARGET_JUMP_THR,
) -> list[int]:
    """Indices `i+1` where |target[i+1] - target[i]| > threshold.

    `direction='positive'` returns only upward jumps (target_diff > thr);
    Exp 22 uses only this. Other directions supplied for unit-testing.
    """
    dif = np.diff(target)
    if direction == "positive":
        mask = dif > threshold
    elif direction == "negative":
        mask = dif < -threshold
    elif direction == "both":
        mask = np.abs(dif) > threshold
    else:
        raise ValueError(f"direction must be positive|negative|both; got {direction!r}")
    # a jump registered at i+1 when target_diff[i] crosses threshold
    return (np.where(mask)[0] + 1).tolist()


def _find_onset(
    v_abs: np.ndarray,
    t_jump: int,
    w_max: int,
    v_onset: float,
) -> Optional[int]:
    """First sample in [t_jump, t_jump+w_max] where |v| exceeds v_onset.

    Uses plain `np.where` — cheap, and these arrays are short."""
    hi = min(t_jump + w_max, len(v_abs))
    seg = v_abs[t_jump:hi]
    hits = np.where(seg > v_onset)[0]
    if hits.size == 0:
        return None
    return int(hits[0]) + t_jump


def _find_offset(
    v_abs: np.ndarray,
    onset: int,
    v_offset: float,
    consec: int,
    max_dur: int,
) -> Optional[int]:
    """First sample at/after `onset` where |v| < v_offset for `consec` samples.

    Returns the index of the first below-threshold sample of the streak.
    Caps search at `onset + max_dur + consec`; if no streak found, returns
    None (caller rejects trial)."""
    hi = min(onset + max_dur + consec, len(v_abs))
    below = v_abs[onset:hi] < v_offset
    if below.size < consec:
        return None
    # sliding AND over `consec`-wide window
    run = np.convolve(below.astype(int), np.ones(consec, dtype=int), mode="valid")
    hits = np.where(run == consec)[0]
    if hits.size == 0:
        return None
    return int(hits[0]) + onset


# ---------------------------------------------------------------------------
# Single-trial extraction (§5.3)
# ---------------------------------------------------------------------------
def detect_eye_saccade(
    pos: np.ndarray,
    v_abs: np.ndarray,
    t_jump: int,
    *,
    v_onset: float = DEFAULT_V_ONSET,
    v_offset: float = DEFAULT_V_OFFSET,
    w_max: int = DEFAULT_W_MAX_SAMPLES,
    offset_consec: int = DEFAULT_OFFSET_CONSEC,
    min_dur: int = DEFAULT_MIN_DUR_SAMPLES,
    max_dur: int = DEFAULT_MAX_DUR_SAMPLES,
) -> Optional[tuple[int, int]]:
    """Locate (onset_idx, offset_idx) for an eye saccade after a target jump.

    Returns None if no qualifying saccade exists or duration is out of
    [min_dur, max_dur] samples (§5.2).
    """
    onset = _find_onset(v_abs, t_jump, w_max, v_onset)
    if onset is None:
        return None
    offset = _find_offset(v_abs, onset, v_offset, offset_consec, max_dur)
    if offset is None:
        return None
    dur = offset - onset
    if dur < min_dur or dur > max_dur:
        return None
    return onset, offset


def extract_trial_kinematics(
    pos: np.ndarray,
    target: np.ndarray,
    t_jump: int,
    trial_idx: int,
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    w_land: int = DEFAULT_W_LAND_SAMPLES,
    v_onset: float = DEFAULT_V_ONSET,
    v_offset: float = DEFAULT_V_OFFSET,
    w_max: int = DEFAULT_W_MAX_SAMPLES,
    offset_consec: int = DEFAULT_OFFSET_CONSEC,
    min_dur: int = DEFAULT_MIN_DUR_SAMPLES,
    max_dur: int = DEFAULT_MAX_DUR_SAMPLES,
) -> TrialKinematics:
    """Extract kinematics for a single target-jump event. NaN if rejected."""
    if t_jump < 1 or t_jump >= len(target):
        a_target = np.nan
    else:
        a_target = float(target[t_jump] - target[t_jump - 1])

    # compute velocity locally — cheap and lets callers pass raw pos
    v = _smoothed_velocity(pos, sample_rate)
    v_abs = np.abs(v)

    out = TrialKinematics(
        trial_idx=trial_idx, t_jump=t_jump, A_target=a_target,
        amplitude=np.nan, gain=np.nan, peak_velocity=np.nan,
        latency=np.nan, duration=np.nan,
    )

    detect = detect_eye_saccade(
        pos, v_abs, t_jump,
        v_onset=v_onset, v_offset=v_offset, w_max=w_max,
        offset_consec=offset_consec, min_dur=min_dur, max_dur=max_dur,
    )
    if detect is None:
        return out
    s, _offset = detect

    hi = min(s + w_land, len(pos) - 1)
    if hi <= s:
        return out

    pos_start = float(pos[s])
    window = pos[s : hi + 1]
    displacements = window - pos_start
    if np.isnan(a_target) or a_target == 0:
        return out
    if a_target > 0:
        land_rel = int(np.argmax(displacements))
    else:
        land_rel = int(np.argmin(displacements))

    amplitude = float(displacements[land_rel])
    gain = amplitude / a_target

    vel_window = v_abs[s : hi + 1]
    peak_vel = float(np.max(vel_window))
    t_pv_rel = int(np.argmax(vel_window))

    out.amplitude = amplitude
    out.gain = gain
    out.peak_velocity = peak_vel
    out.latency = (s - t_jump) / sample_rate
    out.duration = 2.0 * t_pv_rel / sample_rate
    return out


# ---------------------------------------------------------------------------
# Per-sequence extraction helper
# ---------------------------------------------------------------------------
def extract_kinematics_for_sequence(
    eye_pos: np.ndarray,
    target_pos: np.ndarray,
    direction: str = "positive",
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    target_thr: float = DEFAULT_TARGET_JUMP_THR,
    w_land: int = DEFAULT_W_LAND_SAMPLES,
    v_onset: float = DEFAULT_V_ONSET,
    v_offset: float = DEFAULT_V_OFFSET,
    w_max: int = DEFAULT_W_MAX_SAMPLES,
    offset_consec: int = DEFAULT_OFFSET_CONSEC,
    min_dur: int = DEFAULT_MIN_DUR_SAMPLES,
    max_dur: int = DEFAULT_MAX_DUR_SAMPLES,
    drop_rejected: bool = True,
) -> pd.DataFrame:
    """Extract per-trial kinematics for every target-jump of `direction`.

    Returns a DataFrame with columns of `TrialKinematics` plus a
    `sequential_idx` column (0-indexed among trials kept by `drop_rejected`).
    If `drop_rejected=True` (default), rows with NaN kinematics are removed
    before sequential indexing — so `sequential_idx` matches the series
    position used by the fatigue models (§5.4/§5.5).
    """
    jumps = detect_target_jumps(target_pos, direction=direction, threshold=target_thr)
    if len(jumps) == 0:
        return pd.DataFrame(columns=list(TrialKinematics.__annotations__) + ["sequential_idx"])

    rows = []
    for k, t_jump in enumerate(jumps):
        tk = extract_trial_kinematics(
            eye_pos, target_pos, t_jump, trial_idx=k,
            sample_rate=sample_rate, w_land=w_land,
            v_onset=v_onset, v_offset=v_offset, w_max=w_max,
            offset_consec=offset_consec, min_dur=min_dur, max_dur=max_dur,
        )
        rows.append(asdict(tk))
    df = pd.DataFrame(rows)
    if drop_rejected:
        df = df.dropna(subset=["amplitude", "gain", "peak_velocity"]).reset_index(drop=True)
    df["sequential_idx"] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# Synthetic signal generator + self-test (C17)
# ---------------------------------------------------------------------------
def synthesize_saccade_trace(
    amplitude_deg: float = 30.0,
    peak_velocity: float = 500.0,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    pre_samples: int = 30,
    post_samples: int = 120,
    latency_samples: int = 6,
    baseline_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (eye_pos, target_pos) with one Gaussian-velocity saccade.

    Velocity profile: v(t) = V_p * exp(-(t - t_c)^2 / (2σ²)) with σ chosen so
    ∫v dt = amplitude_deg. Eye follows target with `latency_samples` delay.
    Produces a clean saccade with known ground truth for C17 validation.
    """
    # σ (in seconds) from amplitude = V_p * σ * sqrt(2π)
    sigma_sec = amplitude_deg / (peak_velocity * np.sqrt(2.0 * np.pi))
    sigma_samp = sigma_sec * sample_rate

    n = pre_samples + 1 + post_samples  # target jump exactly at index pre_samples+1
    target = np.full(n, baseline_deg, dtype=float)
    target[pre_samples + 1:] = baseline_deg + amplitude_deg

    # center saccade ~ latency after jump, enough buffer to let velocity decay
    t_center = pre_samples + 1 + latency_samples + 3 * sigma_samp
    t = np.arange(n, dtype=float)
    v = (peak_velocity / sample_rate) * np.exp(-((t - t_center) ** 2) / (2.0 * sigma_samp ** 2))
    # integrate to position
    eye = baseline_deg + np.cumsum(v)
    # the eye asymptote should match target — allow tiny numerical drift; renormalise
    final = eye[-1] - baseline_deg
    if abs(final - amplitude_deg) > 0.01:
        eye = baseline_deg + (eye - baseline_deg) * (amplitude_deg / final)
    return eye, target


def _c17_self_test() -> None:
    """Synthetic-signal test (C17).

    Generate a 30° saccade at 500°/s peak velocity, 120 Hz. Verify that the
    extractor recovers amplitude and peak velocity within 5%.
    """
    eye, target = synthesize_saccade_trace(
        amplitude_deg=30.0, peak_velocity=500.0, sample_rate=DEFAULT_SAMPLE_RATE,
    )
    df = extract_kinematics_for_sequence(eye, target, direction="positive")
    assert len(df) == 1, f"expected 1 detected saccade, got {len(df)}"
    row = df.iloc[0]
    amp_err = abs(row["amplitude"] - 30.0) / 30.0
    vp_err = abs(row["peak_velocity"] - 500.0) / 500.0
    gain_err = abs(row["gain"] - 1.0)
    print("C17 synthetic-signal test:")
    print(f"  target amplitude 30.0°  →  recovered {row['amplitude']:.3f}° "
          f"(err {amp_err*100:.2f}%)")
    print(f"  target peak vel  500°/s →  recovered {row['peak_velocity']:.1f}°/s "
          f"(err {vp_err*100:.2f}%)")
    print(f"  gain             1.0    →  recovered {row['gain']:.3f} "
          f"(err {gain_err*100:.2f}%)")
    print(f"  latency                 →  {row['latency']*1000:.1f} ms")
    print(f"  duration                →  {row['duration']*1000:.1f} ms")
    assert amp_err < 0.05, f"amplitude error {amp_err:.3f} exceeds 5%"
    assert vp_err < 0.05, f"peak-velocity error {vp_err:.3f} exceeds 5%"
    print("  PASS (within 5%)")


if __name__ == "__main__":
    _c17_self_test()
