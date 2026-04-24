"""MG vs HC sanity check: train a simple classifier on patient-level
fatigue features, get out-of-fold predictions with stratified k-fold, then
plot raw normalized gain(t) and peak-velocity(t) curves for representative
patients in each of the four cells of the confusion matrix.

The goal is to let a human visually inspect whether the amplitude-vs-
velocity dissociation the clinical partner describes is present in the raw
trial-indexed time series --- especially in the patients our classifier
correctly calls MG or HC vs those it gets wrong.

Outputs land in:
    results/sanity_check_mg_vs_hc/
        predictions.csv
        confusion_matrix.txt
        time_series_grid.png        (one-page overview, 4 rows x 4 cols)
        patient_panels/             (individual full-size panels, one per patient)

Run from project root:
    ./env/bin/python src/tools/mg_vs_hc_sanity_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.fatigue_models import normalize_series  # noqa: E402

EXP22_DIR = os.path.abspath("./results/exp_22_dynamic_fatigability")
OUT_DIR = os.path.abspath("./results/sanity_check_mg_vs_hc")
PANEL_DIR = os.path.join(OUT_DIR, "patient_panels")
os.makedirs(PANEL_DIR, exist_ok=True)

K_BASE = 10
ELL_TAIL = 10
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Build patient-level features directly from the per-trial parquet
# ---------------------------------------------------------------------------
def _per_eye_summary(grp: pd.DataFrame, k: int = 10, ell: int = 10) -> dict:
    g = grp["gain"].values.astype(float)
    vp = grp["peak_velocity"].values.astype(float)
    tt = grp["latency"].values.astype(float)
    dur = grp["duration"].values.astype(float)
    amp = grp["amplitude"].values.astype(float)
    return dict(
        mean_gain=np.nanmean(g),
        mean_vp=np.nanmean(vp),
        mean_amp=np.nanmean(amp),
        mean_tt=np.nanmean(tt),
        mean_dur=np.nanmean(dur),
        baseline_gain=np.nanmean(g[:k]),
        baseline_vp=np.nanmean(vp[:k]),
        tail_gain=np.nanmean(g[-ell:]),
        tail_vp=np.nanmean(vp[-ell:]),
        pd_gain=1.0 - np.nanmean(g[-ell:]) / np.nanmean(g[:k]),
        pd_vp=1.0 - np.nanmean(vp[-ell:]) / np.nanmean(vp[:k]),
        std_gain=np.nanstd(g),
        std_vp=np.nanstd(vp),
        n_trials=len(grp),
    )


def build_patient_features(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trial kinematics to one row per patient (MG + HC only).

    Emits three families of features:
      - per-eye means / baseline / tail / PD (mean across eyes, then visits)
      - between-eye asymmetry: |L - R| per trial, averaged
      - trial-count and variability summaries
    """
    sub = trials[trials["group_label"].isin(["MG", "HC"])].copy()

    # First aggregate per (patient, visit, eye) → summary scalars
    eye_rows = []
    for (pk, visit, eye), grp in sub.groupby(
        ["patient_key", "visit_folder", "eye"], sort=False
    ):
        grp = grp.sort_values("sequential_idx")
        if len(grp) < 20:
            continue
        row = dict(
            patient_key=pk,
            visit=visit,
            eye=eye,
            group_label=grp["group_label"].iloc[0],
            patient_name=grp["patient_name"].iloc[0],
        )
        row.update(_per_eye_summary(grp))
        eye_rows.append(row)
    eye_df = pd.DataFrame(eye_rows)

    # Between-eye asymmetry per visit (|LV - RV| per trial, averaged)
    asym_rows = []
    for (pk, visit), grp in sub.groupby(
        ["patient_key", "visit_folder"], sort=False
    ):
        pivot_g = grp.pivot_table(
            index="sequential_idx", columns="eye", values="gain"
        )
        pivot_v = grp.pivot_table(
            index="sequential_idx", columns="eye", values="peak_velocity"
        )
        if pivot_g.shape[1] < 2 or pivot_v.shape[1] < 2:
            continue
        diff_g = (pivot_g.iloc[:, 0] - pivot_g.iloc[:, 1]).abs()
        diff_v = (pivot_v.iloc[:, 0] - pivot_v.iloc[:, 1]).abs()
        asym_rows.append(dict(
            patient_key=pk,
            visit=visit,
            asym_gain=float(np.nanmean(diff_g)),
            asym_vp=float(np.nanmean(diff_v)),
            asym_gain_std=float(np.nanstd(diff_g)),
            asym_vp_std=float(np.nanstd(diff_v)),
        ))
    asym_df = pd.DataFrame(asym_rows)

    # Collapse eye → patient via mean
    meta = ["patient_key", "group_label", "patient_name"]
    numeric = [c for c in eye_df.columns if eye_df[c].dtype.kind in "fi"]
    eye_agg = (eye_df.groupby(meta, sort=False, as_index=False)[numeric]
               .mean(numeric_only=True))

    # Collapse asymmetry visit → patient via mean
    asym_numeric = [c for c in asym_df.columns if asym_df[c].dtype.kind in "fi"]
    asym_agg = (asym_df.groupby("patient_key", sort=False, as_index=False)
                [asym_numeric].mean(numeric_only=True))

    merged = eye_agg.merge(asym_agg, on="patient_key", how="left")
    merged["label"] = (merged["group_label"] == "MG").astype(int)
    return merged


def pick_features(df: pd.DataFrame) -> list[str]:
    meta = {
        "patient_key", "group_label", "patient_name", "label",
        "n_trials",
    }
    feats = [c for c in df.columns if c not in meta
             and df[c].dtype.kind in "fi"]
    return feats


def oof_predictions(
    df: pd.DataFrame, feats: list[str], n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (oof_pred_label, oof_pred_proba, oof_true_label). Logistic
    regression with median imputation + z-score. Logistic because it is
    less prone to overfitting at n≈200."""
    X = df[feats].values
    y = df["label"].values

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=600, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    oof_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    return oof_pred, oof_proba, y


# ---------------------------------------------------------------------------
# 2. For each patient, pull one trial-indexed (gain, V_p) series
# ---------------------------------------------------------------------------
def load_trials() -> pd.DataFrame:
    path = os.path.join(EXP22_DIR, "kinematic_features_per_trial.parquet")
    return pd.read_parquet(path)


def richest_visit_eye(trials: pd.DataFrame, patient_key: str
                      ) -> tuple[pd.DataFrame, str, str]:
    """Return trials for the (visit, eye) combination with the most trials
    for this patient. Returns (df, visit_folder, eye)."""
    sub = trials[trials["patient_key"] == patient_key]
    if sub.empty:
        return sub, "", ""
    counts = (sub.groupby(["visit_folder", "eye"]).size()
              .sort_values(ascending=False))
    visit_folder, eye = counts.index[0]
    one = sub[(sub["visit_folder"] == visit_folder) & (sub["eye"] == eye)]
    one = one.sort_values("sequential_idx").reset_index(drop=True)
    return one, visit_folder, eye


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def plot_patient_panel(
    ax_raw: plt.Axes, ax_norm: plt.Axes, ax_diff: plt.Axes,
    trials: pd.DataFrame, label: str, pred: int,
    patient_id: str, visit_folder: str, eye: str,
    proba: float,
) -> None:
    """Three stacked subplots: raw gain+Vp, normalized overlay, and the
    normalized difference G̃(t) - Ṽ_p(t). The difference panel is what should
    clearly diverge from zero toward the tail if the amplitude-vs-velocity
    dissociation exists in this patient."""
    pred_label = "MG" if pred == 1 else "HC"
    verdict = "correct" if (pred == int(label == "MG")) else "wrong"
    title_base = (f"{label} (pred={pred_label}, {verdict}) — {patient_id} "
                  f"[{eye}, n={len(trials)}, p={proba:.2f}]")

    if len(trials) < K_BASE + ELL_TAIL:
        ax_raw.set_title(title_base + " — too few trials", fontsize=8)
        for a in (ax_raw, ax_norm, ax_diff):
            a.axis("off")
        return

    t = trials["sequential_idx"].values
    g = trials["gain"].values.astype(float)
    vp = trials["peak_velocity"].values.astype(float)

    # --- Raw axes (twin y for velocity) ---
    ax_raw.plot(t, g, "o-", color="#fb8500", markersize=2.0,
                linewidth=0.8, alpha=0.85)
    ax_raw.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax_raw.set_ylabel("gain", color="#fb8500", fontsize=7)
    ax_raw.tick_params(axis="y", colors="#fb8500", labelsize=7)
    ax_raw.tick_params(axis="x", labelsize=7)

    axv = ax_raw.twinx()
    axv.plot(t, vp, "o-", color="#219ebc", markersize=2.0,
             linewidth=0.8, alpha=0.85)
    axv.set_ylabel("V_p (°/s)", color="#219ebc", fontsize=7)
    axv.tick_params(axis="y", colors="#219ebc", labelsize=7)

    ax_raw.set_title(title_base, fontsize=8)

    # --- Normalized overlay ---
    _, g_tilde = normalize_series(g, k=K_BASE, kind="ratio")
    _, vp_tilde = normalize_series(vp, k=K_BASE, kind="ratio")
    ax_norm.plot(t, g_tilde, "o-", color="#fb8500", markersize=2.0,
                 linewidth=0.8, alpha=0.85, label=r"$\tilde{G}$")
    ax_norm.plot(t, vp_tilde, "o-", color="#219ebc", markersize=2.0,
                 linewidth=0.8, alpha=0.85, label=r"$\tilde{V}_p$")
    ax_norm.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax_norm.axvspan(-0.5, K_BASE - 0.5, color="#b0b0b0", alpha=0.2)
    ax_norm.axvspan(len(t) - ELL_TAIL - 0.5, len(t) - 0.5,
                    color="#ffd166", alpha=0.3)
    ax_norm.set_ylim(0, 2.0)
    ax_norm.set_ylabel("normalized", fontsize=7)
    ax_norm.tick_params(axis="both", labelsize=7)
    ax_norm.legend(loc="lower left", fontsize=6, framealpha=0.7)

    # --- Dissociation-over-time panel ---
    # Delta(t) = G̃(t) - Ṽ_p(t); positive = gain fell more than velocity
    # (the predicted pattern); rolling mean of 5 to smooth.
    delta = g_tilde - vp_tilde
    ax_diff.plot(t, delta, "o-", color="#555555", markersize=2.0,
                 linewidth=0.8, alpha=0.8, label=r"$\tilde{G}-\tilde{V}_p$")
    # Rolling 5-trial mean for trend visibility
    w = 5
    if len(delta) >= w:
        kernel = np.ones(w) / w
        rolling = np.convolve(np.nan_to_num(delta, nan=0.0), kernel,
                              mode="same")
        ax_diff.plot(t, rolling, "-", color="#d62828", linewidth=1.5,
                     alpha=0.85, label="5-trial mean")
    ax_diff.axhline(0.0, color="grey", linestyle="--", linewidth=0.5)
    ax_diff.axvspan(-0.5, K_BASE - 0.5, color="#b0b0b0", alpha=0.2)
    ax_diff.axvspan(len(t) - ELL_TAIL - 0.5, len(t) - 0.5,
                    color="#ffd166", alpha=0.3)
    ax_diff.set_ylim(-1.0, 1.0)
    ax_diff.set_ylabel(r"$\tilde{G}-\tilde{V}_p$", fontsize=7)
    ax_diff.set_xlabel("trial", fontsize=7)
    ax_diff.tick_params(axis="both", labelsize=7)
    ax_diff.legend(loc="lower left", fontsize=6, framealpha=0.7)


def pick_examples(
    df: pd.DataFrame, label: str, pred: int, n: int, seed: int = 0
) -> pd.DataFrame:
    """Pick up to n example patients from the given (label, pred) cell."""
    target = int(label == "MG")
    cell = df[(df["label"] == target) & (df["oof_pred"] == pred)]
    if cell.empty:
        return cell
    cell = cell.copy()
    cell["certainty"] = np.abs(cell["oof_proba"] - 0.5)
    # For correct cells pick most-certain; for wrong cells pick least-certain
    if pred == target:
        cell = cell.sort_values("certainty", ascending=False)
    else:
        cell = cell.sort_values("certainty", ascending=True)
    return cell.head(n)


def make_overview_grid(
    df: pd.DataFrame, trials: pd.DataFrame, n_per_cell: int = 2,
) -> None:
    """4 columns (cells) x 3*n_per_cell rows (raw, norm, diff per patient)."""
    fig, axes = plt.subplots(
        nrows=3 * n_per_cell, ncols=4,
        figsize=(20, 2.3 * 3 * n_per_cell),
        squeeze=False,
    )
    cells = [
        ("MG", 1, "MG — correct"),
        ("MG", 0, "MG — wrong"),
        ("HC", 0, "HC — correct"),
        ("HC", 1, "HC — wrong"),
    ]
    for col, (label, pred, title) in enumerate(cells):
        picks = pick_examples(df, label, pred, n_per_cell)
        for row, (_, patient) in enumerate(picks.iterrows()):
            one, visit, eye = richest_visit_eye(trials, patient["patient_key"])
            ax_raw = axes[3 * row][col]
            ax_norm = axes[3 * row + 1][col]
            ax_diff = axes[3 * row + 2][col]
            pid = f"{label}-{row+1}"
            plot_patient_panel(
                ax_raw, ax_norm, ax_diff, one, label,
                int(patient["oof_pred"]),
                pid, visit, eye, float(patient["oof_proba"]),
            )
        # Column title via the top axis
        axes[0][col].set_title(
            title + "\n" + axes[0][col].get_title(),
            fontsize=10, fontweight="bold",
        )
    fig.suptitle(
        "MG vs HC sanity check --- raw gain + V_p (row 1), normalized overlay "
        "(row 2), and dissociation $\\tilde{G}-\\tilde{V}_p$ (row 3, red = "
        "rolling mean). If the clinical dissociation is present, the red "
        "curve should rise away from zero toward the orange tail window.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "time_series_grid.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _patient_normalized_series(trials: pd.DataFrame, patient_key: str
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool all this patient's (visit, eye) trial-series into one set of
    per-trial-index mean normalized curves. Returns (t, gtilde_mean, vtilde_mean).
    Each (visit, eye) series is independently normalized, then pooled
    by trial index using nanmean so artefact trials don't pull the mean."""
    sub = trials[trials["patient_key"] == patient_key]
    if sub.empty:
        return np.array([]), np.array([]), np.array([])
    series_g = []
    series_v = []
    max_len = 0
    for (_, _), grp in sub.groupby(["visit_folder", "eye"], sort=False):
        grp = grp.sort_values("sequential_idx")
        if len(grp) < K_BASE + ELL_TAIL:
            continue
        g = grp["gain"].values.astype(float)
        vp = grp["peak_velocity"].values.astype(float)
        _, g_t = normalize_series(g, k=K_BASE, kind="ratio")
        _, v_t = normalize_series(vp, k=K_BASE, kind="ratio")
        series_g.append(g_t)
        series_v.append(v_t)
        max_len = max(max_len, len(g_t))
    if not series_g:
        return np.array([]), np.array([]), np.array([])
    # Pad to max_len with NaN and take nanmean across sequences
    def _pad(s, n):
        out = np.full(n, np.nan)
        out[:len(s)] = s
        return out
    G = np.vstack([_pad(s, max_len) for s in series_g])
    V = np.vstack([_pad(s, max_len) for s in series_v])
    return (
        np.arange(max_len),
        np.nanmean(G, axis=0),
        np.nanmean(V, axis=0),
    )


def make_mean_dissociation_figure(
    df: pd.DataFrame, trials: pd.DataFrame,
) -> None:
    """One figure showing mean G̃(t) - Ṽ_p(t) per cell, with CI bands.

    If the clinical dissociation is real, the MG-correct curve (and
    arguably MG-wrong) should rise above zero toward the tail; HC-correct
    should stay at zero."""
    cells = [
        ("MG correct",  "MG", 1, "#fb8500"),
        ("MG wrong",    "MG", 0, "#d62828"),
        ("HC correct",  "HC", 0, "#219ebc"),
        ("HC wrong",    "HC", 1, "#6a4c93"),
    ]
    max_len = 75
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-cell aggregates for left panel; per-group aggregates for right panel.
    group_all = {"MG": {"g": [], "v": []}, "HC": {"g": [], "v": []}}

    for title, label, pred, color in cells:
        target = int(label == "MG")
        cell_df = df[(df["label"] == target) & (df["oof_pred"] == pred)]
        deltas = []
        for _, row in cell_df.iterrows():
            t, g_t, v_t = _patient_normalized_series(
                trials, row["patient_key"]
            )
            if len(t) < K_BASE + ELL_TAIL:
                continue
            d = g_t - v_t
            padded_d = np.full(max_len, np.nan)
            padded_d[:min(len(d), max_len)] = d[:max_len]
            deltas.append(padded_d)
            # Accumulate for group-level right panel
            padded_g = np.full(max_len, np.nan)
            padded_g[:min(len(g_t), max_len)] = g_t[:max_len]
            padded_v = np.full(max_len, np.nan)
            padded_v[:min(len(v_t), max_len)] = v_t[:max_len]
            group_all[label]["g"].append(padded_g)
            group_all[label]["v"].append(padded_v)

        if not deltas:
            continue
        D = np.vstack(deltas)
        n_per_trial = np.sum(~np.isnan(D), axis=0)
        mean = np.nanmean(D, axis=0)
        sem = np.nanstd(D, axis=0) / np.sqrt(np.maximum(n_per_trial, 1))
        t_axis = np.arange(max_len)
        axes[0].plot(t_axis, mean, "-", color=color,
                     linewidth=2.2, label=f"{title} (n={len(deltas)})")
        axes[0].fill_between(
            t_axis, mean - 1.96 * sem, mean + 1.96 * sem,
            color=color, alpha=0.15,
        )

    # Right panel: MG vs HC group means of G̃ and Ṽ
    t_axis = np.arange(max_len)
    for label, style in (("MG", "-"), ("HC", "--")):
        if not group_all[label]["g"]:
            continue
        G = np.vstack(group_all[label]["g"])
        V = np.vstack(group_all[label]["v"])
        g_mean = np.nanmean(G, axis=0)
        v_mean = np.nanmean(V, axis=0)
        g_sem = np.nanstd(G, axis=0) / np.sqrt(
            np.maximum(np.sum(~np.isnan(G), axis=0), 1)
        )
        v_sem = np.nanstd(V, axis=0) / np.sqrt(
            np.maximum(np.sum(~np.isnan(V), axis=0), 1)
        )
        n_group = G.shape[0]
        axes[1].plot(t_axis, g_mean, style, color="#fb8500",
                     linewidth=2.0,
                     label=f"gain — {label} (n={n_group})")
        axes[1].fill_between(
            t_axis, g_mean - 1.96 * g_sem, g_mean + 1.96 * g_sem,
            color="#fb8500", alpha=0.10,
        )
        axes[1].plot(t_axis, v_mean, style, color="#219ebc",
                     linewidth=2.0,
                     label=f"V_p  — {label} (n={n_group})")
        axes[1].fill_between(
            t_axis, v_mean - 1.96 * v_sem, v_mean + 1.96 * v_sem,
            color="#219ebc", alpha=0.10,
        )

    # ax[0]: dissociation per cell
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axvspan(-0.5, K_BASE - 0.5, color="#b0b0b0", alpha=0.2)
    axes[0].axvspan(max_len - ELL_TAIL - 0.5, max_len - 0.5,
                    color="#ffd166", alpha=0.3,
                    label="tail window")
    axes[0].set_xlabel("trial index")
    axes[0].set_ylabel(r"mean $\tilde{G}(t) - \tilde{V}_p(t)$")
    axes[0].set_title(
        "Mean dissociation per cell (shaded = 95% CI).\n"
        "Prediction: MG curves should rise positive into the tail; "
        "HC curves should stay near 0."
    )
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.2)

    # ax[1]: normalized gain and Vp means for MG correct / wrong
    axes[1].axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    axes[1].axvspan(-0.5, K_BASE - 0.5, color="#b0b0b0", alpha=0.2)
    axes[1].axvspan(max_len - ELL_TAIL - 0.5, max_len - 0.5,
                    color="#ffd166", alpha=0.3)
    axes[1].set_xlabel("trial index")
    axes[1].set_ylabel(r"mean $\tilde{G}(t)$ or $\tilde{V}_p(t)$")
    axes[1].set_title(
        "Mean normalized gain vs V_p by group (solid=MG, dashed=HC).\n"
        "Prediction: MG orange gain should fall below MG blue V_p by the tail."
    )
    axes[1].set_ylim(0.7, 1.3)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "mean_dissociation_per_cell.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_individual_panels(
    df: pd.DataFrame, trials: pd.DataFrame, n_per_cell: int = 6,
) -> None:
    cells = [
        ("MG", 1, "mg_correct"),
        ("MG", 0, "mg_wrong"),
        ("HC", 0, "hc_correct"),
        ("HC", 1, "hc_wrong"),
    ]
    for label, pred, tag in cells:
        picks = pick_examples(df, label, pred, n_per_cell)
        for i, (_, patient) in enumerate(picks.iterrows()):
            one, visit, eye = richest_visit_eye(trials, patient["patient_key"])
            fig, axes = plt.subplots(3, 1, figsize=(10, 7),
                                     gridspec_kw={"height_ratios": [3, 3, 2]})
            pid = f"{label}-{i+1:02d}"
            plot_patient_panel(
                axes[0], axes[1], axes[2], one, label,
                int(patient["oof_pred"]),
                pid, visit, eye, float(patient["oof_proba"]),
            )
            fig.suptitle(
                f"{tag} — patient {pid} [{eye}] — proba={patient['oof_proba']:.2f}",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out = os.path.join(PANEL_DIR, f"{tag}__{i+1:02d}.png")
            fig.savefig(out, dpi=140)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading per-trial parquet…")
    trials = load_trials()
    print(f"  {len(trials)} trials total")

    print("Building per-patient features (mean aggregation)…")
    df = build_patient_features(trials)
    feats = pick_features(df)
    print(f"  {len(df)} patients (MG={int((df['label']==1).sum())}, "
          f"HC={int((df['label']==0).sum())}), {len(feats)} features")
    print(f"  features: {feats}")

    print("Running 5-fold stratified CV with Logistic Regression…")
    oof_pred, oof_proba, y_true = oof_predictions(df, feats)
    df["oof_pred"] = oof_pred
    df["oof_proba"] = oof_proba

    acc = accuracy_score(y_true, oof_pred)
    auc = roc_auc_score(y_true, oof_proba)
    cm = confusion_matrix(y_true, oof_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  accuracy = {acc:.3f}   AUROC = {auc:.3f}")
    print(f"  confusion matrix (rows=true, cols=pred, order [HC, MG]):")
    print(cm)
    print(f"  HC correct={tn}, HC→MG wrong={fp}, MG→HC wrong={fn}, MG correct={tp}")

    # Write predictions.csv
    out_pred = df[[
        "patient_key", "group_label", "patient_name", "label",
        "oof_pred", "oof_proba",
    ]].copy()
    out_pred.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "confusion_matrix.txt"), "w") as f:
        f.write(f"accuracy = {acc:.3f}\n")
        f.write(f"AUROC    = {auc:.3f}\n")
        f.write(f"\nrows=true, cols=pred, order [HC, MG]\n")
        f.write(f"                pred=HC    pred=MG\n")
        f.write(f"true=HC       {tn:8d}  {fp:8d}\n")
        f.write(f"true=MG       {fn:8d}  {tp:8d}\n")

    print("Rendering overview grid…")
    make_overview_grid(df, trials, n_per_cell=2)

    print("Rendering mean-dissociation-per-cell summary…")
    make_mean_dissociation_figure(df, trials)

    print("Rendering individual patient panels…")
    make_individual_panels(df, trials, n_per_cell=6)

    print(f"Done. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
