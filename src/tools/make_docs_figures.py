"""Generate onboarding figures under docs/figures/.

Run from project root:
    ./env/bin/python src/tools/make_docs_figures.py

Figures produced:
    01_group_counts.png          patient/visit/CSV counts per group
    02_single_recording.png      one HC vertical 1 Hz recording, three views
    03_stimulus_conditions.png   same patient across all 2x3 stimulus conditions
    04_saccade_anatomy.png       one trial zoom with target jump, onset, peak Vp
    05_dc_offset.png             distribution of raw eye DC offsets across all CSVs
    06_cross_group_examples.png  HC vs MG vs CNP raw vertical 1 Hz traces
    07_kinematic_distributions.png  per-trial gain/Vp/latency/duration by group

All images are saved to docs/figures/ with 140 dpi.
"""

from __future__ import annotations

import glob
import os
import random
import re
import sys
from typing import Optional

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

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.abspath("./data")
DOCS_DIR = os.path.abspath("./docs/figures")
os.makedirs(DOCS_DIR, exist_ok=True)

PARQUET_PATH = os.path.abspath(
    "./results/exp_22_dynamic_fatigability/kinematic_features_per_trial.parquet"
)

SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # 120 Hz
RNG_SEED = 11

GROUP_ORDER = [
    ("Healthy control", "HC"),
    ("Definite MG", "MG-Def"),
    ("Probable MG", "MG-Prob"),
    ("Non-MG diplopia (CNP, etc)/3rd", "CNP-3rd"),
    ("Non-MG diplopia (CNP, etc)/4th", "CNP-4th"),
    ("Non-MG diplopia (CNP, etc)/6th", "CNP-6th"),
]

GROUP_COLORS = {
    "HC": "#2ca02c",
    "MG-Def": "#d62728",
    "MG-Prob": "#ff7f0e",
    "CNP-3rd": "#1f77b4",
    "CNP-4th": "#9467bd",
    "CNP-6th": "#8c564b",
}

POS_YLIM = 25.0
VEL_YLIM = 700.0


# ---------------------------------------------------------------------------
# CSV loading and centering
# ---------------------------------------------------------------------------
def load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, encoding="utf-16-le", sep=",")
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    return df


def fixation_offset(eye: np.ndarray, target: np.ndarray) -> float:
    mask = (np.abs(target) < 0.5) & np.isfinite(eye)
    if mask.sum() >= 10:
        return float(np.mean(eye[mask]))
    keep = np.isfinite(eye) & (np.abs(eye) < 50.0)
    return float(np.mean(eye[keep])) if keep.any() else 0.0


def preprocess_vertical(df: pd.DataFrame) -> dict:
    need = ["LV", "RV", "TargetV"]
    for c in need:
        if c not in df.columns:
            return {}
    target = df["TargetV"].values.astype(float)
    lv = df["LV"].values.astype(float)
    rv = df["RV"].values.astype(float)
    target_c = target - float(np.mean(target))
    lv_c = lv - fixation_offset(lv, target)
    rv_c = rv - fixation_offset(rv, target)
    eye = 0.5 * (lv_c + rv_c)
    vel = _smoothed_velocity(eye, SAMPLE_RATE)
    t = np.arange(len(df)) / SAMPLE_RATE
    return {"t": t, "target": target_c, "eye": eye, "vel": vel}


def visit_date(visit_folder: str) -> str:
    """Extract the YYYY-MM-DD prefix; drop the patient name to avoid font
    glitches on the Korean-language folder suffixes."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", visit_folder)
    return m.group(1) if m else visit_folder[:10]


def find_csv(group_folder: str, visit_folder: str, axis: str, freq: str
             ) -> Optional[str]:
    vdir = os.path.join(DATA_DIR, group_folder, visit_folder)
    if not os.path.isdir(vdir):
        return None
    for fn in os.listdir(vdir):
        low = fn.lower()
        if not fn.endswith(".csv"):
            continue
        if axis.lower() in low and freq.lower() in low:
            return os.path.join(vdir, fn)
    return None


def list_vertical_1hz_csvs(group_folder: str) -> list[tuple[str, str]]:
    gdir = os.path.join(DATA_DIR, group_folder)
    out: list[tuple[str, str]] = []
    if not os.path.isdir(gdir):
        return out
    for visit in sorted(os.listdir(gdir)):
        vdir = os.path.join(gdir, visit)
        if not os.path.isdir(vdir):
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}\s+.+$", visit):
            continue
        for fn in os.listdir(vdir):
            low = fn.lower()
            if fn.endswith(".csv") and "vertical" in low and "1hz" in low:
                out.append((visit, os.path.join(vdir, fn)))
                break
    return out


# ---------------------------------------------------------------------------
# Figure 01: group counts
# ---------------------------------------------------------------------------
def figure_group_counts() -> str:
    labels = []
    visits = []
    csvs = []
    unique_names = []
    for folder, label in GROUP_ORDER:
        gdir = os.path.join(DATA_DIR, folder)
        labels.append(label)
        if not os.path.isdir(gdir):
            visits.append(0)
            csvs.append(0)
            unique_names.append(0)
            continue
        visit_folders = [d for d in os.listdir(gdir)
                         if os.path.isdir(os.path.join(gdir, d))]
        visits.append(len(visit_folders))
        total_csvs = 0
        names: set[str] = set()
        for v in visit_folders:
            total_csvs += len(glob.glob(os.path.join(gdir, v, "*.csv")))
            m = re.match(r"^\d{4}-\d{2}-\d{2}\s+(.+)$", v)
            if m:
                names.add(m.group(1).strip())
        csvs.append(total_csvs)
        unique_names.append(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(labels))
    colors = [GROUP_COLORS[l] for l in labels]

    for ax, values, title in zip(
        axes,
        [unique_names, visits, csvs],
        ["Unique patients", "Visits", "CSV files"],
    ):
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(values) * 0.01,
                str(v),
                ha="center", va="bottom", fontsize=9,
            )
        ax.margins(y=0.15)

    fig.suptitle(
        "Cohort sizes per clinical group  (counts as of April 2026)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(DOCS_DIR, "01_group_counts.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 02: one full recording
# ---------------------------------------------------------------------------
def figure_single_recording() -> str:
    visit, csv = list_vertical_1hz_csvs("Healthy control")[0]
    df = load_csv(csv)
    sig = preprocess_vertical(df)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    t = sig["t"]
    total = t[-1]

    # (a) target across full session
    ax = axes[0]
    ax.plot(t, sig["target"], color="black", drawstyle="steps-post", linewidth=0.9)
    ax.set_title(
        "(a) Target vertical position, full session "
        f"(N = {len(t)} samples, sample rate = 120 Hz, duration = {total:.1f} s)",
        fontsize=11,
    )
    ax.set_ylabel("target (°)", fontsize=10)
    ax.set_xlim(0, total)
    ax.set_ylim(-18, 18)
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.grid(linewidth=0.3, alpha=0.5)

    # (b) position first 20 s
    zoom_end = 20.0
    sl = t <= zoom_end
    ax = axes[1]
    ax.plot(t[sl], sig["target"][sl], color="black", drawstyle="steps-post",
            linewidth=1.2, label="target")
    ax.plot(t[sl], sig["eye"][sl], color="#1f77b4", linewidth=1.2,
            label="eye position (mean of LV, RV, centred)")
    ax.set_title(
        "(b) Eye tracks target, first 20 s zoom",
        fontsize=11,
    )
    ax.set_ylabel("position (°)", fontsize=10)
    ax.set_xlim(0, zoom_end)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(linewidth=0.3, alpha=0.5)

    # (c) velocity first 20 s
    ax = axes[2]
    ax.plot(t[sl], sig["vel"][sl], color="#d62728", linewidth=1.0,
            label="eye velocity (smoothed)")
    ax.axhline(DEFAULT_V_ONSET, color="grey", linestyle=":", linewidth=0.7,
               label=f"onset threshold = ±{DEFAULT_V_ONSET:.0f} °/s")
    ax.axhline(-DEFAULT_V_ONSET, color="grey", linestyle=":", linewidth=0.7)
    ax.axhline(0, color="grey", linewidth=0.3, linestyle="--")
    ax.set_title("(c) Eye velocity, first 20 s zoom", fontsize=11)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("velocity (°/s)", fontsize=10)
    ax.set_xlim(0, zoom_end)
    ax.set_ylim(-VEL_YLIM, VEL_YLIM)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(linewidth=0.3, alpha=0.5)

    fig.suptitle(
        f"One recording from Healthy control, vertical 1 Hz  "
        f"(visit {visit_date(visit)})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(DOCS_DIR, "02_single_recording.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 03: six stimulus conditions for one patient
# ---------------------------------------------------------------------------
def figure_stimulus_conditions() -> str:
    # Pick the first HC patient that actually has all six CSVs.
    gfolder = "Healthy control"
    picked_visit = None
    for visit in sorted(os.listdir(os.path.join(DATA_DIR, gfolder))):
        vdir = os.path.join(DATA_DIR, gfolder, visit)
        if not os.path.isdir(vdir):
            continue
        files = os.listdir(vdir)
        have = 0
        for ax in ("Horizontal", "Vertical"):
            for fq in ("0.5Hz", "0.75Hz", "1Hz"):
                if any(ax.lower() in f.lower() and fq.lower() in f.lower()
                       for f in files):
                    have += 1
        if have == 6:
            picked_visit = visit
            break
    if picked_visit is None:
        raise RuntimeError("no HC patient has all six stimulus CSVs")

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True, sharey=True)
    freqs = ["0.5Hz", "0.75Hz", "1Hz"]
    axes_names = ["Horizontal", "Vertical"]
    zoom_end = 15.0

    for r, axis_name in enumerate(axes_names):
        for c, freq in enumerate(freqs):
            ax_plot = axes[r][c]
            csv = find_csv(gfolder, picked_visit, axis_name, freq)
            if csv is None:
                ax_plot.axis("off")
                continue
            df = load_csv(csv)
            if df is None:
                ax_plot.axis("off")
                continue
            if axis_name == "Vertical":
                eye_cols = ("LV", "RV", "TargetV")
            else:
                eye_cols = ("LH", "RH", "TargetH")
            target = df[eye_cols[2]].values.astype(float)
            le = df[eye_cols[0]].values.astype(float)
            re_ = df[eye_cols[1]].values.astype(float)
            t_c = target - float(np.mean(target))
            le_c = le - fixation_offset(le, target)
            re_c = re_ - fixation_offset(re_, target)
            eye = 0.5 * (le_c + re_c)
            t = np.arange(len(df)) / SAMPLE_RATE
            sl = t <= zoom_end
            ax_plot.plot(t[sl], t_c[sl], color="black", drawstyle="steps-post",
                         linewidth=1.2, label="target")
            ax_plot.plot(t[sl], eye[sl], color="#1f77b4", linewidth=1.1,
                         label="eye position")
            ax_plot.set_title(f"{axis_name}  {freq}", fontsize=11,
                              fontweight="bold")
            ax_plot.axhline(0, color="grey", linewidth=0.4, linestyle="--")
            ax_plot.grid(linewidth=0.3, alpha=0.5)
            ax_plot.set_xlim(0, zoom_end)
            ax_plot.set_ylim(-20, 20)
            if r == 1:
                ax_plot.set_xlabel("time (s)", fontsize=9)
            if c == 0:
                ax_plot.set_ylabel("position (°)", fontsize=9)
            if r == 0 and c == 0:
                ax_plot.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Six stimulus conditions (same patient, first 15 s zoom)  "
        f"(visit {visit_date(picked_visit)})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(DOCS_DIR, "03_stimulus_conditions.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 04: saccade anatomy
# ---------------------------------------------------------------------------
def figure_saccade_anatomy() -> str:
    visit, csv = list_vertical_1hz_csvs("Healthy control")[0]
    df = load_csv(csv)
    target = df["TargetV"].values.astype(float)
    eye = df["LV"].values.astype(float)
    target_c = target - float(np.mean(target))
    eye_c = eye - fixation_offset(eye, target)
    vel = _smoothed_velocity(eye_c, SAMPLE_RATE)

    # Pick the first positive target jump.
    jumps = detect_target_jumps(target, direction="positive", threshold=5.0)
    t_jump = jumps[0]

    # Locate saccade onset with the same rule the extractor uses.
    v_abs = np.abs(vel)
    hits = np.where(v_abs[t_jump:t_jump + 120] > DEFAULT_V_ONSET)[0]
    onset = t_jump + int(hits[0]) if hits.size else t_jump + 20

    window_lo = t_jump - 20
    window_hi = onset + DEFAULT_W_LAND_SAMPLES + 30
    window_lo = max(window_lo, 0)
    window_hi = min(window_hi, len(target) - 1)
    sl = slice(window_lo, window_hi + 1)
    t = np.arange(len(target)) / SAMPLE_RATE

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.plot(t[sl], target_c[sl], color="black", drawstyle="steps-post",
            linewidth=1.5, label="target")
    ax.plot(t[sl], eye_c[sl], color="#1f77b4", linewidth=1.6,
            label="eye position (LV, centred)")
    ax.axvline(t[t_jump], color="grey", linestyle=":", linewidth=1.0,
               label="t_jump (target step)")
    ax.axvline(t[onset], color="#2ca02c", linestyle="--", linewidth=1.0,
               label="s = saccade onset  (|v| first > 30 °/s)")
    land_start = t[onset]
    land_end = t[min(onset + DEFAULT_W_LAND_SAMPLES, len(target) - 1)]
    ax.axvspan(land_start, land_end, color="#aee", alpha=0.35,
               label="150 ms landing window")
    ax.set_ylabel("position (°)", fontsize=10)
    ax.set_title("(a) Target jump and eye response", fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(linewidth=0.3, alpha=0.5)

    ax = axes[1]
    ax.plot(t[sl], vel[sl], color="#d62728", linewidth=1.4,
            label="eye velocity (smoothed)")
    ax.axhline(DEFAULT_V_ONSET, color="grey", linestyle=":", linewidth=0.8,
               label=f"onset threshold = {DEFAULT_V_ONSET:.0f} °/s")
    ax.axvline(t[onset], color="#2ca02c", linestyle="--", linewidth=1.0)
    ax.axvspan(land_start, land_end, color="#aee", alpha=0.35)
    # peak velocity marker inside landing window
    land_hi = min(onset + DEFAULT_W_LAND_SAMPLES, len(vel) - 1)
    land_vel = vel[onset:land_hi + 1]
    if land_vel.size:
        peak_rel = int(np.argmax(np.abs(land_vel)))
        peak_idx = onset + peak_rel
        ax.plot(t[peak_idx], vel[peak_idx], "o", color="#ff7f0e",
                markersize=9, markeredgecolor="black",
                label=f"peak velocity  V_p ≈ {vel[peak_idx]:.0f} °/s")
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("velocity (°/s)", fontsize=10)
    ax.set_title("(b) Velocity profile", fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(linewidth=0.3, alpha=0.5)

    fig.suptitle(
        "Saccade anatomy: one trial, target step, onset, 150 ms landing window, "
        "peak velocity",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(DOCS_DIR, "04_saccade_anatomy.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 05: DC offset distribution across all CSVs
# ---------------------------------------------------------------------------
def figure_dc_offset() -> str:
    raw_medians: list[float] = []
    centered_means: list[float] = []
    per_group_raw: dict[str, list[float]] = {}

    for folder, label in GROUP_ORDER:
        for visit, csv in list_vertical_1hz_csvs(folder):
            df = load_csv(csv)
            if df is None or "LV" not in df.columns or "TargetV" not in df.columns:
                continue
            lv = df["LV"].values.astype(float)
            target = df["TargetV"].values.astype(float)
            raw_medians.append(float(np.nanmedian(lv)))
            centered_means.append(
                float(np.nanmean(lv - fixation_offset(lv, target)))
            )
            per_group_raw.setdefault(label, []).append(
                float(np.nanmedian(lv))
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(-20, 20, 41)
    ax = axes[0]
    ax.hist(raw_medians, bins=bins, color="#888888", edgecolor="black",
            linewidth=0.4, label="raw LV median (°)")
    ax.hist(centered_means, bins=bins, color="#1f77b4", edgecolor="black",
            linewidth=0.4, alpha=0.8,
            label="LV mean after fixation-dwell centring (°)")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("left-eye vertical position (°)", fontsize=10)
    ax.set_ylabel("number of CSVs", fontsize=10)
    ax.set_title(
        "(a) Before vs after centering "
        f"(N = {len(raw_medians)} vertical 1 Hz CSVs)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(linewidth=0.3, alpha=0.5)

    ax = axes[1]
    group_labels = list(per_group_raw.keys())
    data = [per_group_raw[g] for g in group_labels]
    positions = np.arange(len(group_labels))
    parts = ax.violinplot(
        data, positions=positions, widths=0.8, showmeans=False,
        showmedians=True, showextrema=False,
    )
    for pc, label in zip(parts["bodies"], group_labels):
        pc.set_facecolor(GROUP_COLORS.get(label, "#888888"))
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("raw LV median (°)", fontsize=10)
    ax.set_title("(b) DC offset spread by group (raw, uncentred)", fontsize=11)
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Raw eye-position DC offset varies widely across patients; "
        "fixation-dwell centring removes it",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(DOCS_DIR, "05_dc_offset.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 06: cross-group examples
# ---------------------------------------------------------------------------
def figure_cross_group() -> str:
    rng = random.Random(RNG_SEED)

    def _one_csv(folders: list[str]) -> Optional[tuple[str, str]]:
        pool: list[tuple[str, str]] = []
        for f in folders:
            pool.extend([(f, p) for _, p in list_vertical_1hz_csvs(f)])
        rng.shuffle(pool)
        for folder, path in pool:
            df = load_csv(path)
            if df is None or "LV" not in df.columns:
                continue
            return folder, path
        return None

    hc = _one_csv(["Healthy control"])
    mg = _one_csv(["Definite MG", "Probable MG"])
    cnp = _one_csv([
        "Non-MG diplopia (CNP, etc)/3rd",
        "Non-MG diplopia (CNP, etc)/4th",
        "Non-MG diplopia (CNP, etc)/6th",
    ])
    picks = [("HC", hc), ("MG", mg), ("CNP", cnp)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, (label, pick) in enumerate(picks):
        ax = axes[i]
        if pick is None:
            ax.axis("off")
            continue
        folder, csv = pick
        df = load_csv(csv)
        sig = preprocess_vertical(df)
        t = sig["t"]
        ax.plot(t, sig["target"], color="black", drawstyle="steps-post",
                linewidth=0.9, label="target", alpha=0.75)
        ax.plot(t, sig["eye"], color="#1f77b4", linewidth=0.8,
                label="eye position", alpha=0.85)
        ax2 = ax.twinx()
        ax2.plot(t, sig["vel"], color="#d62728", linewidth=0.6,
                 label="velocity", alpha=0.55)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-POS_YLIM, POS_YLIM)
        ax2.set_ylim(-VEL_YLIM, VEL_YLIM)
        ax.set_ylabel("position (°)", fontsize=10, color="#1f77b4")
        ax2.set_ylabel("velocity (°/s)", fontsize=10, color="#d62728")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.axhline(0, color="grey", linewidth=0.3, linestyle="--")
        visit = os.path.basename(os.path.dirname(csv))
        ax.set_title(
            f"{label}  |  group folder: {folder}  |  visit {visit_date(visit)}",
            fontsize=11, fontweight="bold", loc="left",
        )
        if i == 0:
            hp, lp = ax.get_legend_handles_labels()
            hv, lv = ax2.get_legend_handles_labels()
            ax.legend(hp + hv, lp + lv, loc="upper right",
                      fontsize=9, ncol=3, framealpha=0.9)
        if i == 2:
            ax.set_xlabel("time (s)", fontsize=10)

    fig.suptitle(
        "One recording from each clinical group (vertical 1 Hz, full session)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(DOCS_DIR, "06_cross_group_examples.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 07: kinematic distributions from the per-trial parquet
# ---------------------------------------------------------------------------
def figure_kinematic_distributions() -> Optional[str]:
    if not os.path.exists(PARQUET_PATH):
        print(f"  skipping 07: parquet not found at {PARQUET_PATH}")
        return None
    df = pd.read_parquet(PARQUET_PATH)
    # Trim outliers that only reflect detection failures.
    clip = {
        "gain": (0.0, 2.0),
        "peak_velocity": (0.0, 900.0),
        "latency": (0.0, 0.6),
        "duration": (0.0, 0.25),
    }
    group_order = ["HC", "MG", "CNP"]
    palette = {"HC": "#2ca02c", "MG": "#d62728", "CNP": "#1f77b4"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    kinematics = [
        ("gain", "Saccade gain  (amplitude / target step)"),
        ("peak_velocity", "Peak velocity (°/s)"),
        ("latency", "Latency  (s, target jump → saccade onset)"),
        ("duration", "Duration  (s, 2 × time to peak velocity)"),
    ]
    for (col, title), ax in zip(kinematics, axes.flat):
        data = []
        for g in group_order:
            lo, hi = clip[col]
            vals = df.loc[df["group_label"] == g, col].dropna().values
            vals = vals[(vals >= lo) & (vals <= hi)]
            data.append(vals)
        positions = np.arange(len(group_order))
        parts = ax.violinplot(
            data, positions=positions, widths=0.8,
            showmeans=False, showmedians=True, showextrema=False,
        )
        for pc, g in zip(parts["bodies"], group_order):
            pc.set_facecolor(palette[g])
            pc.set_edgecolor("black")
            pc.set_alpha(0.75)
        counts = [len(d) for d in data]
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{g}\n(n = {c})" for g, c in zip(group_order, counts)],
            fontsize=10,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Per-trial kinematic distributions by group  "
        f"(N = {len(df):,} trials, Vertical 1 Hz, upward jumps)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(DOCS_DIR, "07_kinematic_distributions.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Writing onboarding figures to", DOCS_DIR)
    for name, fn in [
        ("01 group counts", figure_group_counts),
        ("02 single recording", figure_single_recording),
        ("03 stimulus conditions", figure_stimulus_conditions),
        ("04 saccade anatomy", figure_saccade_anatomy),
        ("05 DC offset", figure_dc_offset),
        ("06 cross-group examples", figure_cross_group),
        ("07 kinematic distributions", figure_kinematic_distributions),
    ]:
        print(f"  {name} ...")
        out = fn()
        if out:
            print(f"    wrote {out}")
    print("done.")


if __name__ == "__main__":
    main()
