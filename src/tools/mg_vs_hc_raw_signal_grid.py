"""Raw-signal visualization: 4 random HC + 4 random MG patients.

Two figures on vertical 1 Hz saccade CSVs, written to docs/figures/:

  1. 08_random_hc_vs_mg_overview.png  (4 rows x 2 cols)
     Left col = HC, right col = MG. Each cell overlays eye position,
     target, and eye velocity on twin-y axes, all zero-centred, over
     the full ~150 s recording.

  2. 09_random_hc_vs_mg_zoom.png  (4 rows x 4 cols)
     Each patient gets two side-by-side panels: first few saccades (left)
     and last few saccades (right). Column order: HC-start, HC-end,
     MG-start, MG-end.

Patients are picked uniformly at random from the HC and MG (Definite +
Probable) pools. No classifier is used.

Run from project root:
    ./env/bin/python src/tools/mg_vs_hc_raw_signal_grid.py
"""

from __future__ import annotations

import os
import random
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.saccade_kinematics import (  # noqa: E402
    DEFAULT_SAMPLE_RATE, _smoothed_velocity, detect_target_jumps,
)

MG_FOLDERS = ["Definite MG", "Probable MG"]
HC_FOLDERS = ["Healthy control"]

DATA_DIR = os.path.abspath("./data")
OUT_DIR = os.path.abspath("./docs/figures")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # 120 Hz
RNG_SEED = 7
N_PER_GROUP = 4
N_SACCADES_ZOOM = 5  # how many target jumps to show at each end

POS_YLIM = 25.0   # degrees, clip eye position to hide blink spikes
VEL_YLIM = 700.0  # deg/s,  clip velocity for the same reason


# ---------------------------------------------------------------------------
# Discover patients and pick a CSV
# ---------------------------------------------------------------------------
def list_patients(folders: list[str]) -> list[tuple[str, str]]:
    """Return [(group_folder, visit_folder), ...] for every patient visit
    containing a Vertical 1 Hz CSV."""
    out: list[tuple[str, str]] = []
    for folder_name in folders:
        gdir = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(gdir):
            continue
        for visit in sorted(os.listdir(gdir)):
            vdir = os.path.join(gdir, visit)
            if not os.path.isdir(vdir):
                continue
            if not re.match(r"^\d{4}-\d{2}-\d{2}\s+.+$", visit):
                continue
            for fn in os.listdir(vdir):
                low = fn.lower()
                if fn.endswith(".csv") and "vertical" in low and "1hz" in low:
                    out.append((folder_name, visit))
                    break
    return out


def find_vertical_1hz_csv(folder_name: str, visit_folder: str) -> str | None:
    vdir = os.path.join(DATA_DIR, folder_name, visit_folder)
    best = None
    best_size = -1
    for fn in os.listdir(vdir):
        low = fn.lower()
        if not fn.endswith(".csv"):
            continue
        if "vertical" in low and "1hz" in low:
            p = os.path.join(vdir, fn)
            s = os.path.getsize(p)
            if s > best_size:
                best, best_size = p, s
    return best


# ---------------------------------------------------------------------------
# Load + centre
# ---------------------------------------------------------------------------
def load_sequence(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, encoding="utf-16-le", sep=",")
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    need = ["LV", "RV", "TargetV"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return df.reset_index(drop=True)


def fixation_offset(eye: np.ndarray, target: np.ndarray) -> float:
    """Offset to subtract from eye so the centre-of-fixation sits at 0.

    Primary: mean of eye samples where |target| < 0.5 (the target is at the
    centre fixation dwell). Fallback for older sequences with no dwell:
    the mean of outlier-trimmed eye position.
    """
    mask = (np.abs(target) < 0.5) & np.isfinite(eye)
    if mask.sum() >= 10:
        return float(np.mean(eye[mask]))
    keep = np.isfinite(eye) & (np.abs(eye) < 50.0)
    return float(np.mean(eye[keep])) if keep.any() else 0.0


def preprocess(df: pd.DataFrame) -> dict:
    target = df["TargetV"].values.astype(float)
    lv = df["LV"].values.astype(float)
    rv = df["RV"].values.astype(float)
    # Target is nominally symmetric around 0; subtract its own mean in case of drift.
    target_c = target - float(np.mean(target))
    lv_c = lv - fixation_offset(lv, target)
    rv_c = rv - fixation_offset(rv, target)
    eye = 0.5 * (lv_c + rv_c)
    vel = _smoothed_velocity(eye, SAMPLE_RATE)
    t = np.arange(len(df)) / SAMPLE_RATE
    return {"t": t, "target": target_c, "eye": eye, "vel": vel,
            "lv": lv_c, "rv": rv_c}


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------
def _twin_axes(ax: plt.Axes) -> plt.Axes:
    ax2 = ax.twinx()
    return ax2


def _plot_window(
    ax_pos: plt.Axes, sig: dict, t_lo: float, t_hi: float,
    show_xlabel: bool = True, show_ylabels: bool = True,
) -> plt.Axes:
    """Plot target + eye position on ax_pos and velocity on a twin-y.
    Returns the twin (velocity) axis."""
    t = sig["t"]
    sl = (t >= t_lo) & (t <= t_hi)

    ax_vel = _twin_axes(ax_pos)

    ax_pos.plot(
        t[sl], sig["target"][sl], color="black",
        linewidth=1.3, drawstyle="steps-post",
        label="target", alpha=0.85,
    )
    ax_pos.plot(
        t[sl], sig["eye"][sl], color="#1f77b4",
        linewidth=1.1, alpha=0.95, label="amplitude (eye)",
    )
    ax_vel.plot(
        t[sl], sig["vel"][sl], color="#d62728",
        linewidth=0.9, alpha=0.75, label="velocity",
    )

    ax_pos.set_xlim(t_lo, t_hi)
    ax_pos.set_ylim(-POS_YLIM, POS_YLIM)
    ax_vel.set_ylim(-VEL_YLIM, VEL_YLIM)
    ax_pos.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.7)

    if show_ylabels:
        ax_pos.set_ylabel("position (°)", fontsize=9, color="#1f77b4")
        ax_vel.set_ylabel("velocity (°/s)", fontsize=9, color="#d62728")
    else:
        ax_pos.set_ylabel("")
        ax_vel.set_ylabel("")
    ax_pos.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)
    ax_vel.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
    ax_pos.tick_params(axis="x", labelsize=8)
    if show_xlabel:
        ax_pos.set_xlabel("time (s)", fontsize=9)
    else:
        ax_pos.set_xlabel("")
    return ax_vel


def saccade_windows(target: np.ndarray, n: int) -> tuple[tuple[float, float],
                                                          tuple[float, float]]:
    """Return (first_n_window, last_n_window) in seconds, each framing `n`
    target jumps with a small pre/post pad."""
    jumps = detect_target_jumps(target, direction="both", threshold=5.0)
    if len(jumps) < 2:
        total = len(target) / SAMPLE_RATE
        half = total / 2
        return (0.0, half), (total - half, total)

    pad_pre = 0.4   # seconds before first jump of the window
    pad_post = 1.2  # seconds after last jump of the window (saccade + return)

    first_start = jumps[0]
    first_end = jumps[min(n - 1, len(jumps) - 1)]
    last_start = jumps[max(0, len(jumps) - n)]
    last_end = jumps[-1]

    t_first_lo = max(0.0, first_start / SAMPLE_RATE - pad_pre)
    t_first_hi = first_end / SAMPLE_RATE + pad_post
    t_last_lo = max(0.0, last_start / SAMPLE_RATE - pad_pre)
    t_last_hi = last_end / SAMPLE_RATE + pad_post
    total = len(target) / SAMPLE_RATE
    t_first_hi = min(t_first_hi, total)
    t_last_hi = min(t_last_hi, total)
    return (t_first_lo, t_first_hi), (t_last_lo, t_last_hi)


# ---------------------------------------------------------------------------
# Sampling patients
# ---------------------------------------------------------------------------
def sample_patients(
    rng: random.Random,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Return (HC picks, MG picks). Each pick is (group_folder, visit, csv_path)."""
    def _pick(folders: list[str], n: int) -> list[tuple[str, str, str]]:
        visits = list_patients(folders)
        rng.shuffle(visits)
        out: list[tuple[str, str, str]] = []
        for group, visit in visits:
            csv = find_vertical_1hz_csv(group, visit)
            if csv is None:
                continue
            df = load_sequence(csv)
            if df is None or len(df) < 2000:
                continue
            out.append((group, visit, csv))
            if len(out) >= n:
                break
        return out

    return _pick(HC_FOLDERS, N_PER_GROUP), _pick(MG_FOLDERS, N_PER_GROUP)


def patient_title(group: str, visit: str, row_idx: int) -> str:
    """Build a compact header. Visit dates are informative; names are not
    transliterated."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", visit)
    date = m.group(1) if m else visit[:10]
    return f"{group} #{row_idx+1}  ({date})"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def render_overview(hc: list, mg: list) -> str:
    fig, axes = plt.subplots(
        nrows=N_PER_GROUP, ncols=2,
        figsize=(18, 3.1 * N_PER_GROUP),
        squeeze=False,
    )
    for col, (group_label, picks) in enumerate([("HC", hc), ("MG", mg)]):
        for row in range(N_PER_GROUP):
            ax = axes[row][col]
            if row >= len(picks):
                ax.axis("off")
                continue
            group_folder, visit, csv = picks[row]
            df = load_sequence(csv)
            sig = preprocess(df)
            ax_vel = _plot_window(
                ax, sig, sig["t"][0], sig["t"][-1],
                show_xlabel=(row == N_PER_GROUP - 1),
                show_ylabels=True,
            )
            ax.set_title(
                patient_title(group_label, visit, row),
                fontsize=11, fontweight="bold",
            )
            if row == 0:
                handles_pos, labels_pos = ax.get_legend_handles_labels()
                handles_vel, labels_vel = ax_vel.get_legend_handles_labels()
                ax.legend(
                    handles_pos + handles_vel, labels_pos + labels_vel,
                    loc="upper right", fontsize=7, framealpha=0.85, ncol=3,
                )

    fig.suptitle(
        "Random 4 HC vs 4 MG, full-session vertical 1 Hz saccades.  "
        "Position & target on left axis (°); velocity on right axis (°/s).",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "08_random_hc_vs_mg_overview.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def render_zoom(hc: list, mg: list) -> str:
    """4 rows x 4 cols. Column layout:
       col 0 = HC first saccades    col 1 = HC last saccades
       col 2 = MG first saccades    col 3 = MG last saccades
    """
    fig, axes = plt.subplots(
        nrows=N_PER_GROUP, ncols=4,
        figsize=(22, 3.2 * N_PER_GROUP),
        squeeze=False,
    )

    for group_idx, (group_label, picks) in enumerate([("HC", hc), ("MG", mg)]):
        col_first = 2 * group_idx
        col_last = 2 * group_idx + 1
        for row in range(N_PER_GROUP):
            ax_a = axes[row][col_first]
            ax_b = axes[row][col_last]
            if row >= len(picks):
                ax_a.axis("off")
                ax_b.axis("off")
                continue
            _, visit, csv = picks[row]
            df = load_sequence(csv)
            sig = preprocess(df)

            (t0a, t1a), (t0b, t1b) = saccade_windows(
                sig["target"], n=N_SACCADES_ZOOM,
            )
            ax_vel_a = _plot_window(
                ax_a, sig, t0a, t1a,
                show_xlabel=(row == N_PER_GROUP - 1),
                show_ylabels=True,
            )
            ax_vel_b = _plot_window(
                ax_b, sig, t0b, t1b,
                show_xlabel=(row == N_PER_GROUP - 1),
                show_ylabels=False,
            )

            ax_a.set_title(
                f"{patient_title(group_label, visit, row)}  |first "
                f"{N_SACCADES_ZOOM} saccades",
                fontsize=10, fontweight="bold",
            )
            ax_b.set_title(
                f"{patient_title(group_label, visit, row)}  |last "
                f"{N_SACCADES_ZOOM} saccades",
                fontsize=10, fontweight="bold",
            )

            if row == 0 and group_idx == 0:
                hp, lp = ax_a.get_legend_handles_labels()
                hv, lv = ax_vel_a.get_legend_handles_labels()
                ax_a.legend(hp + hv, lp + lv, loc="upper right",
                            fontsize=7, framealpha=0.85, ncol=3)

    fig.suptitle(
        "Zoomed comparison: first vs last saccades for the same 4 HC and "
        "4 MG patients (vertical 1 Hz).  "
        "If MG shows amplitude-velocity dissociation, the blue position "
        "trace should shrink left→right while red velocity stays tall.",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(OUT_DIR, "09_random_hc_vs_mg_zoom.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = random.Random(RNG_SEED)
    hc, mg = sample_patients(rng)
    print(f"HC picks ({len(hc)}):")
    for g, v, _ in hc:
        print(f"  [{g}] {v}")
    print(f"MG picks ({len(mg)}):")
    for g, v, _ in mg:
        print(f"  [{g}] {v}")

    p1 = render_overview(hc, mg)
    print(f"wrote {p1}")
    p2 = render_zoom(hc, mg)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
