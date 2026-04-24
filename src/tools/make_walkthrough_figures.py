"""Generate pedagogical figures for the metric-walkthrough deck.

Picks one HC vertical 1.0 Hz sequence, runs each stage of the extractor,
and saves figures at:

    results/exp_22_dynamic_fatigability/figures/walkthrough/
        1_raw_sequence.png
        2_raw_zoom.png
        3_velocity.png
        4_trial_indexed_series.png
        5_normalized_and_tail.png

Run from project root:
    ./env/bin/python src/tools/make_walkthrough_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.saccade_kinematics import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    DEFAULT_V_ONSET,
    DEFAULT_W_LAND_SAMPLES,
    _smoothed_velocity,
    detect_target_jumps,
    extract_kinematics_for_sequence,
)
from utils.fatigue_models import (  # noqa: E402
    normalize_series,
    compute_fatigue_indices,
)

OUT = os.path.abspath(
    "./results/exp_22_dynamic_fatigability/figures/walkthrough"
)
os.makedirs(OUT, exist_ok=True)

CSV_PATH = (
    "./data/Healthy control/2023-07-07 송나리/"
    "송나리 MG_Vertical Saccade  B (1Hz)_000.csv"
)

SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # 120 Hz
W_LAND = DEFAULT_W_LAND_SAMPLES    # 18 samples = 150 ms


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16-le", sep=",")
    df.columns = [c.strip() for c in df.columns]
    df = df[["LH", "RH", "LV", "RV", "TargetH", "TargetV"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna(how="any")
    return df


def fig_1_raw_sequence(df: pd.DataFrame) -> None:
    """Panel A: target (step) + eye (LV) over the full sequence."""
    t = np.arange(len(df)) / SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, df["TargetV"], color="black", linewidth=1.2,
            label="Target vertical (°)", drawstyle="steps-post")
    ax.plot(t, df["LV"], color="#fb8500", linewidth=0.9, alpha=0.9,
            label="Left-eye vertical (°)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vertical position (°)")
    ax.set_title(
        "One sequence: target jumps (step) + eye follows (saccades). "
        f"Sample rate = {int(SAMPLE_RATE)} Hz, N = {len(df)} samples "
        f"(= {len(df)/SAMPLE_RATE:.1f} s)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1_raw_sequence.png"), dpi=150)
    plt.close(fig)


def fig_2_raw_zoom(df: pd.DataFrame) -> None:
    """Panel B: zoom on one target jump + saccade, onset + landing window."""
    target = df["TargetV"].values
    eye = df["LV"].values
    v = _smoothed_velocity(eye, SAMPLE_RATE)
    jumps = detect_target_jumps(target, direction="positive", threshold=5.0)
    if not jumps:
        return
    # Pick the second jump (first sometimes has edge effects)
    t_jump = jumps[1] if len(jumps) > 1 else jumps[0]
    # Find onset
    v_abs = np.abs(v)
    seg = v_abs[t_jump : t_jump + 120]
    hits = np.where(seg > DEFAULT_V_ONSET)[0]
    if len(hits) == 0:
        return
    s = int(hits[0]) + t_jump
    # Zoom window: a bit before t_jump to a bit after landing
    lo = max(0, t_jump - 20)
    hi = min(len(eye), s + W_LAND + 40)
    t = np.arange(lo, hi) / SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(t, target[lo:hi], color="black", linewidth=1.6,
            label="Target", drawstyle="steps-post")
    ax.plot(t, eye[lo:hi], color="#fb8500", linewidth=1.4,
            label="Left-eye position")
    # Target-jump marker
    ax.axvline(t_jump / SAMPLE_RATE, color="grey", linestyle=":",
               linewidth=1.0, label="target jump $t_{\\mathrm{jump}}$")
    # Saccade-onset marker
    ax.axvline(s / SAMPLE_RATE, color="#219ebc", linestyle="--",
               linewidth=1.2, label="saccade onset $s$")
    # Landing window shading
    ax.axvspan(s / SAMPLE_RATE, (s + W_LAND) / SAMPLE_RATE,
               color="#8ecae6", alpha=0.25,
               label=f"landing window ({W_LAND} samples = 150 ms)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vertical position (°)")
    ax.set_title(
        "Zoom on one trial: target jumps, eye reaches threshold velocity "
        "at $s$, then we measure inside the 150 ms landing window"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2_raw_zoom.png"), dpi=150)
    plt.close(fig)


def fig_3_velocity(df: pd.DataFrame) -> None:
    """Panel C: velocity trace with 30°/s threshold for the same zoom."""
    target = df["TargetV"].values
    eye = df["LV"].values
    v = _smoothed_velocity(eye, SAMPLE_RATE)
    jumps = detect_target_jumps(target, direction="positive", threshold=5.0)
    if not jumps:
        return
    t_jump = jumps[1] if len(jumps) > 1 else jumps[0]
    v_abs = np.abs(v)
    seg = v_abs[t_jump : t_jump + 120]
    hits = np.where(seg > DEFAULT_V_ONSET)[0]
    if len(hits) == 0:
        return
    s = int(hits[0]) + t_jump
    lo = max(0, t_jump - 20)
    hi = min(len(eye), s + W_LAND + 40)
    t = np.arange(lo, hi) / SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(t, v[lo:hi], color="#219ebc", linewidth=1.2,
            label="eye velocity $v(t)$ (smoothed)")
    ax.axhline(DEFAULT_V_ONSET, color="red", linestyle=":", linewidth=1.0,
               label=f"onset threshold = {int(DEFAULT_V_ONSET)}°/s")
    ax.axhline(-DEFAULT_V_ONSET, color="red", linestyle=":", linewidth=1.0)
    ax.axvline(s / SAMPLE_RATE, color="#219ebc", linestyle="--",
               linewidth=1.2, label="saccade onset $s$ (first $|v| > 30$°/s)")
    ax.axvspan(s / SAMPLE_RATE, (s + W_LAND) / SAMPLE_RATE,
               color="#8ecae6", alpha=0.25, label="landing window")
    # Mark peak velocity
    t_pv = int(np.argmax(v_abs[s : s + W_LAND])) + s
    ax.plot(t_pv / SAMPLE_RATE, v[t_pv], "o", color="#fb8500",
            markersize=9, label=f"peak velocity $V_p$ at $t_{{pv}}$")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("eye velocity (°/s)")
    ax.set_title(
        "Same trial, velocity view: onset detected where $|v(t)|$ first "
        "crosses 30°/s; $V_p$ = max $|v|$ inside the landing window"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "3_velocity.png"), dpi=150)
    plt.close(fig)


def _extract_trial_series(df: pd.DataFrame, eye_col: str = "LV"):
    target = df["TargetV"].values.astype(float)
    eye = df[eye_col].values.astype(float)
    trials = extract_kinematics_for_sequence(
        eye, target, direction="positive", sample_rate=SAMPLE_RATE
    )
    return trials


def fig_4_trial_indexed_series(df: pd.DataFrame) -> None:
    """Panel D: G(t) and V_p(t) as per-trial series across the sequence."""
    trials = _extract_trial_series(df, "LV")
    if len(trials) == 0:
        return
    t = trials["trial_idx"].values
    g = trials["gain"].values
    vp = trials["peak_velocity"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax1.plot(t, g, "o-", color="#fb8500", markersize=4,
             linewidth=1.0, alpha=0.9)
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("gain $G$")
    ax1.set_title(f"Per-trial kinematics, one sequence "
                  f"({len(trials)} trials). Top: gain. Bottom: peak velocity.")
    ax1.grid(alpha=0.2)

    ax2.plot(t, vp, "o-", color="#219ebc", markersize=4,
             linewidth=1.0, alpha=0.9)
    ax2.set_ylabel("peak velocity $V_p$ (°/s)")
    ax2.set_xlabel("trial index $t$")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "4_trial_indexed_series.png"), dpi=150)
    plt.close(fig)


def fig_5_normalized_and_tail(df: pd.DataFrame) -> None:
    """Panel E: normalized G̃(t) and Ṽ(t), with baseline and tail windows shaded."""
    trials = _extract_trial_series(df, "LV")
    if len(trials) == 0:
        return
    t = trials["trial_idx"].values
    g = trials["gain"].values.astype(float)
    vp = trials["peak_velocity"].values.astype(float)

    k, ell = 10, 10
    g_base, g_tilde = normalize_series(g, k=k, kind="ratio")
    vp_base, vp_tilde = normalize_series(vp, k=k, kind="ratio")

    # Use compute_fatigue_indices on the already-normalized series
    idx_g = compute_fatigue_indices(g_tilde, kind="ratio", k=k, ell=ell,
                                    baseline=g_base)
    idx_v = compute_fatigue_indices(vp_tilde, kind="ratio", k=k, ell=ell,
                                    baseline=vp_base)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, g_tilde, "o-", color="#fb8500", markersize=4,
            linewidth=1.0, alpha=0.9, label=r"$\tilde{G}(t)$")
    ax.plot(t, vp_tilde, "o-", color="#219ebc", markersize=4,
            linewidth=1.0, alpha=0.9, label=r"$\tilde{V}_p(t)$")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)

    # Baseline window shading
    ax.axvspan(-0.5, k - 0.5, color="#b0b0b0", alpha=0.25,
               label=f"baseline window (first {k} trials)")
    # Tail window shading
    if len(t) >= ell:
        tail_lo = len(t) - ell - 0.5
        ax.axvspan(tail_lo, len(t) - 0.5, color="#ffd166", alpha=0.35,
                   label=f"tail window (last {ell} trials)")

    ax.set_xlabel("trial index $t$")
    ax.set_ylabel("normalized value")
    ax.set_title(
        f"Normalized to own baseline. "
        f"PD_gain = {idx_g.pd_:.3f}, PD_Vp = {idx_v.pd_:.3f}, "
        f"DI_PD = {idx_g.pd_ - idx_v.pd_:+.3f}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "5_normalized_and_tail.png"), dpi=150)
    plt.close(fig)


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-16-le", sep=",")
    df.columns = [c.strip() for c in df.columns]
    df = df[["LH", "RH", "LV", "RV", "TargetH", "TargetV"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna(how="any").reset_index(drop=True)
    print(f"loaded {len(df)} samples from {CSV_PATH}")

    fig_1_raw_sequence(df)
    print(f"wrote {OUT}/1_raw_sequence.png")
    fig_2_raw_zoom(df)
    print(f"wrote {OUT}/2_raw_zoom.png")
    fig_3_velocity(df)
    print(f"wrote {OUT}/3_velocity.png")
    fig_4_trial_indexed_series(df)
    print(f"wrote {OUT}/4_trial_indexed_series.png")
    fig_5_normalized_and_tail(df)
    print(f"wrote {OUT}/5_normalized_and_tail.png")


if __name__ == "__main__":
    main()
