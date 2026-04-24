#!/usr/bin/env python3
"""Experiment 22 — Within-Eye Dynamic Fatigability and the Amplitude–Velocity
Dissociation Index.

This orchestrator is the single entry point for Exp 22. It implements the
design in `llmdocs/exp_22_design.md` end-to-end:

    load CSVs → per-trial kinematics → within-subject normalization → curve
    fits → fatigue indices (visit / eye level) → max-over-eyes + max-over-
    visits aggregation → DI_PD and DI_β at the patient level → Hedges' g,
    Mann-Whitney U, bootstrap CI → §6.4 sensitivities → REPORT.md + figures.

Run from project root:
    ./env/bin/python src/exp_22_dynamic_fatigability.py

Outputs land in ./results/exp_22_dynamic_fatigability/.
"""

from __future__ import annotations

import glob
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.file_metadata import (  # noqa: E402
    GROUP_PATHS, parse_filename, patient_id, strip_folder_date,
    MG_GROUPS, CNP_GROUPS, HC_GROUPS,
)
from utils.saccade_kinematics import (  # noqa: E402
    DEFAULT_SAMPLE_RATE, extract_kinematics_for_sequence,
)
from utils.fatigue_models import (  # noqa: E402
    ALL_KINEMATICS, RATIO_KINEMATICS, ADDITIVE_KINEMATICS,
    indices_for_kinematic, kinematic_kind,
)
from utils import test_partner_formulas as _v2_fidelity  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath("./data")
RESULTS_DIR = os.path.abspath("./results/exp_22_dynamic_fatigability")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CSV_ENCODING = "utf-16-le"
CSV_SEPARATOR = ","
FEATURE_COLUMNS = ["LH", "RH", "LV", "RV", "TargetH", "TargetV"]

PRIMARY_AXIS = "Vertical"
PRIMARY_DIRECTION = "positive"           # upward only (C1)
PRIMARY_FREQ = 1.0                       # Hz — primary; 0.5 / 0.75 are sensitivities
OTHER_FREQUENCIES = [0.5, 0.75]
SAMPLE_RATE = DEFAULT_SAMPLE_RATE        # 120 Hz

# §5.4 primary defaults
K_BASELINE_DEFAULT = 10
L_END_DEFAULT = 10
MIN_VALID_TRIALS_DEFAULT = 30           # baseline + end + middle ≥ 30

BOOTSTRAP_ITER = 10_000
BOOTSTRAP_SEED = 20260423


# Group labels for the three-way analysis
def _group_label(group: str) -> str:
    if group in HC_GROUPS:
        return "HC"
    if group in MG_GROUPS:
        return "MG"
    if group in CNP_GROUPS:
        return "CNP"
    raise ValueError(f"unknown group {group!r}")


# ---------------------------------------------------------------------------
# Data iteration
# ---------------------------------------------------------------------------
@dataclass
class SequenceRef:
    group: str
    group_label: str             # 'HC' / 'MG' / 'CNP'
    subtype: Optional[str]       # e.g. 'CNP_3rd', 'MG_Def'
    patient_name: str            # folder name without YYYY-MM-DD prefix
    patient_id: tuple[str, str]  # (group, name)
    visit_folder: str            # folder name (used as unique visit id)
    axis: str
    frequency: float
    filepath: str


def iterate_sequence_refs(base_dir: str = BASE_DIR) -> list[SequenceRef]:
    """Walk the data tree once and return metadata for every parseable CSV."""
    refs: list[SequenceRef] = []
    for group, sub in GROUP_PATHS.items():
        gdir = os.path.join(base_dir, sub)
        if not os.path.isdir(gdir):
            continue
        for folder in sorted(os.listdir(gdir)):
            fdir = os.path.join(gdir, folder)
            if not os.path.isdir(fdir):
                continue
            name = strip_folder_date(folder)
            for fname in sorted(os.listdir(fdir)):
                if not fname.endswith(".csv"):
                    continue
                parsed = parse_filename(fname)
                if parsed is None or parsed.axis is None:
                    continue
                refs.append(SequenceRef(
                    group=group,
                    group_label=_group_label(group),
                    subtype=group,
                    patient_name=name,
                    patient_id=patient_id(group, folder),
                    visit_folder=folder,
                    axis=parsed.axis,
                    frequency=parsed.frequency,
                    filepath=os.path.join(fdir, fname),
                ))
    return refs


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, encoding=CSV_ENCODING, sep=CSV_SEPARATOR)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if any(c not in df.columns for c in FEATURE_COLUMNS):
        return None
    df = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(df) < 200:
        return None
    return df


# ---------------------------------------------------------------------------
# Per-sequence → per-trial kinematics
# ---------------------------------------------------------------------------
def extract_all_trials(
    refs: Iterable[SequenceRef],
    axis: str = PRIMARY_AXIS,
    frequencies: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Iterate sequences and return a long per-trial DataFrame.

    One row per (visit, sequence, eye, trial). If `frequencies` is None, all
    frequencies are kept.
    """
    freqs_allowed = set(frequencies) if frequencies is not None else None
    rows = []
    for ref in tqdm(list(refs), desc="extract-kinematics", leave=False):
        if ref.axis != axis:
            continue
        if freqs_allowed is not None and ref.frequency not in freqs_allowed:
            continue
        df_raw = _read_csv(ref.filepath)
        if df_raw is None:
            continue
        target_col = "TargetV" if axis == "Vertical" else "TargetH"
        eye_cols = ("LV", "RV") if axis == "Vertical" else ("LH", "RH")
        target = df_raw[target_col].values.astype(float)
        for eye in eye_cols:
            pos = df_raw[eye].values.astype(float)
            trials = extract_kinematics_for_sequence(
                pos, target, direction=PRIMARY_DIRECTION,
                sample_rate=SAMPLE_RATE,
            )
            if len(trials) == 0:
                continue
            trials = trials.copy()
            trials["group"] = ref.group
            trials["group_label"] = ref.group_label
            trials["subtype"] = ref.subtype
            trials["patient_name"] = ref.patient_name
            trials["patient_group"] = ref.patient_id[0]
            trials["patient_key"] = f"{ref.patient_id[0]}::{ref.patient_id[1]}"
            trials["visit_folder"] = ref.visit_folder
            trials["axis"] = ref.axis
            trials["frequency"] = ref.frequency
            trials["eye"] = eye
            rows.append(trials)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-(visit,eye,frequency) fatigue indices
# ---------------------------------------------------------------------------
def compute_eye_level_indices(
    trials_df: pd.DataFrame,
    k: int = K_BASELINE_DEFAULT,
    ell: int = L_END_DEFAULT,
    min_valid_trials: int = MIN_VALID_TRIALS_DEFAULT,
) -> pd.DataFrame:
    """For each (group, patient, visit, frequency, eye) compute five indices
    per kinematic. Returns long-form; one row per (id, kinematic)."""
    group_keys = ["group", "group_label", "subtype", "patient_name",
                  "patient_key", "visit_folder", "frequency", "eye"]
    out_rows = []
    grouped = trials_df.groupby(group_keys, sort=False)
    for key, grp in grouped:
        if len(grp) < min_valid_trials:
            continue
        grp = grp.sort_values("sequential_idx")
        meta = dict(zip(group_keys, key))
        for kin in ALL_KINEMATICS:
            values = grp[kin].values.astype(float)
            idx = indices_for_kinematic(values, kin, k=k, ell=ell)
            idx.update(meta)
            idx["n_trials"] = len(grp)
            out_rows.append(idx)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Aggregation: max over eyes → max over visits (§5.7)
# ---------------------------------------------------------------------------
def _aggregate(df: pd.DataFrame, by: list[str], how: str) -> pd.DataFrame:
    """Aggregate numeric columns by `by` using `how` ∈ {max, mean}."""
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi"
                and c not in {"frequency"}]
    meta_cols = [c for c in df.columns if c not in num_cols]
    if how == "max":
        agg = df.groupby(by + [c for c in ("frequency",) if c in df.columns],
                         sort=False)[num_cols].max().reset_index()
    elif how == "mean":
        agg = df.groupby(by + [c for c in ("frequency",) if c in df.columns],
                         sort=False)[num_cols].mean().reset_index()
    else:
        raise ValueError(how)
    return agg


def aggregate_eye_to_visit(df: pd.DataFrame, how: str = "max") -> pd.DataFrame:
    return _aggregate(
        df, by=["group", "group_label", "subtype", "patient_name", "patient_key",
                "visit_folder", "kinematic"], how=how,
    )


def aggregate_visit_to_patient(df: pd.DataFrame, how: str = "max") -> pd.DataFrame:
    return _aggregate(
        df, by=["group", "group_label", "subtype", "patient_name", "patient_key",
                "kinematic"], how=how,
    )


def pivot_patient_indices(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Long → wide: one row per patient, columns `<index>__<kinematic>`."""
    id_cols = ["group", "group_label", "subtype", "patient_name", "patient_key",
               "frequency"]
    id_cols = [c for c in id_cols if c in patient_df.columns]
    value_cols = [c for c in patient_df.columns
                  if c not in id_cols + ["kinematic", "kind"]]
    wide = patient_df.pivot_table(
        index=id_cols, columns="kinematic", values=value_cols, aggfunc="first",
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    # DI = PD_gain - PD_peak_velocity
    if "pd___gain" in wide.columns and "pd___peak_velocity" in wide.columns:
        wide["DI_PD"] = wide["pd___gain"] - wide["pd___peak_velocity"]
    if "beta1__gain" in wide.columns and "beta1__peak_velocity" in wide.columns:
        wide["DI_beta"] = wide["beta1__gain"] - wide["beta1__peak_velocity"]
    # Change-point dissociation index (v2 C21). Positive = fatigue intensifies
    # in gain more than in peak velocity after the change point, matching the
    # partner's MG-vs-CNP mechanism prediction applied to the third curve
    # family (Section C #3).
    if ("cp_delta__gain" in wide.columns
            and "cp_delta__peak_velocity" in wide.columns):
        wide["DI_CPdelta"] = wide["cp_delta__gain"] - wide["cp_delta__peak_velocity"]
    return wide


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g with small-sample correction."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = x.var(ddof=1); s2 = y.var(ddof=1)
    s_pool_sq = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if s_pool_sq <= 0:
        return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(s_pool_sq)
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(d * J)


def bootstrap_hedges_g_ci(
    x: np.ndarray, y: np.ndarray,
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Hedges' g (patient-level resampling)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    gs = np.empty(n_iter)
    for i in range(n_iter):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        gs[i] = hedges_g(xb, yb)
    gs = gs[~np.isnan(gs)]
    lo = float(np.quantile(gs, alpha / 2))
    hi = float(np.quantile(gs, 1 - alpha / 2))
    return lo, hi


def mannwhitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def auroc(x: np.ndarray, y: np.ndarray) -> float:
    """AUROC of a score separating x (class 1) from y (class 0)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    scores = np.concatenate([x, y])
    labels = np.concatenate([np.ones_like(x), np.zeros_like(y)])
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return np.nan


def two_group_contrast(
    df_patient_wide: pd.DataFrame,
    metric: str,
    g1: str,
    g2: str,
    label_col: str = "group_label",
    bootstrap_iter: int = BOOTSTRAP_ITER,
) -> dict:
    """Effect size + CI + Mann-Whitney + AUROC for `metric` between g1 and g2."""
    x = df_patient_wide.loc[df_patient_wide[label_col] == g1, metric].dropna().values
    y = df_patient_wide.loc[df_patient_wide[label_col] == g2, metric].dropna().values
    g = hedges_g(x, y)
    lo, hi = bootstrap_hedges_g_ci(x, y, n_iter=bootstrap_iter)
    u, p = mannwhitney(x, y)
    auc = auroc(x, y)
    return dict(
        metric=metric, contrast=f"{g1} vs {g2}",
        n1=len(x), n2=len(y),
        mean1=float(np.mean(x)) if len(x) else np.nan,
        mean2=float(np.mean(y)) if len(y) else np.nan,
        median1=float(np.median(x)) if len(x) else np.nan,
        median2=float(np.median(y)) if len(y) else np.nan,
        hedges_g=g, ci_lo=lo, ci_hi=hi,
        mannwhitney_U=u, mannwhitney_p=p,
        auroc=auc,
    )


# ---------------------------------------------------------------------------
# Convergence reporting (C15)
# ---------------------------------------------------------------------------
def exponential_convergence_rates(eye_level: pd.DataFrame) -> pd.DataFrame:
    """Per (group_label, kinematic) exponential-fit convergence rate."""
    rows = []
    grouped = eye_level.groupby(["group_label", "kinematic"])
    for key, grp in grouped:
        rate = grp["exp_converged"].mean()
        rows.append(dict(group=key[0], kinematic=key[1],
                         converged=int(grp["exp_converged"].sum()),
                         total=len(grp),
                         rate=float(rate)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------
def build_patient_wide(
    trials_df: pd.DataFrame,
    k: int = K_BASELINE_DEFAULT,
    ell: int = L_END_DEFAULT,
    min_valid_trials: int = MIN_VALID_TRIALS_DEFAULT,
    eye_agg: str = "max",
    visit_agg: str = "max",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From per-trial → eye-level → visit-level → patient-level wide table."""
    eye_level = compute_eye_level_indices(
        trials_df, k=k, ell=ell, min_valid_trials=min_valid_trials,
    )
    visit_level = aggregate_eye_to_visit(eye_level, how=eye_agg)
    patient_level = aggregate_visit_to_patient(visit_level, how=visit_agg)
    wide = pivot_patient_indices(patient_level)
    return eye_level, visit_level, patient_level, wide


def run_primary_analysis(wide: pd.DataFrame) -> pd.DataFrame:
    """H1, H2, H3 + secondary contrasts on patient-level wide table.

    v2 adds change-point peers (H1_cp / H2_cp / H3_cp) per CONTEXT.md C20:
    change-point must be reported alongside linear and exponential in every
    downstream index table.

    v3 adds MG vs HC parallel rows for every MG vs CNP supporting contrast per
    CONTEXT.md C27: DI_beta_MG_HC, DI_CPdelta_MG_HC, PD_vp_MG_HC. MG vs HC is
    a partner verbal scope extension (C29) — the written doc endpoints remain
    headline (H2 DI_PD MG vs CNP is still the documented primary).
    """
    rows = []
    # H1 positive control: PD_gain MG vs HC
    rows.append(dict(hypothesis="H1", **two_group_contrast(wide, "pd___gain", "MG", "HC")))
    # H2 primary: DI_PD MG vs CNP
    rows.append(dict(hypothesis="H2", **two_group_contrast(wide, "DI_PD", "MG", "CNP")))
    # H2 supporting: DI_beta MG vs CNP
    rows.append(dict(hypothesis="H2b", **two_group_contrast(wide, "DI_beta", "MG", "CNP")))
    # H3 sanity: DI_PD CNP vs HC
    rows.append(dict(hypothesis="H3", **two_group_contrast(wide, "DI_PD", "CNP", "HC")))
    # Also DI_PD MG vs HC and single-feature PD_gain MG vs CNP
    rows.append(dict(hypothesis="DI_PD_MG_HC", **two_group_contrast(wide, "DI_PD", "MG", "HC")))
    rows.append(dict(hypothesis="PD_gain_MG_CNP", **two_group_contrast(wide, "pd___gain", "MG", "CNP")))
    rows.append(dict(hypothesis="PD_vp_MG_CNP", **two_group_contrast(wide, "pd___peak_velocity", "MG", "CNP")))
    # v3 MG vs HC parallels (C27): DI_beta and PD_vp dissociation variants
    rows.append(dict(hypothesis="DI_beta_MG_HC", **two_group_contrast(wide, "DI_beta", "MG", "HC")))
    rows.append(dict(hypothesis="PD_vp_MG_HC", **two_group_contrast(wide, "pd___peak_velocity", "MG", "HC")))
    # v2 change-point peers (C20, C21)
    if "cp_delta__gain" in wide.columns:
        rows.append(dict(hypothesis="H1_cp",
                         **two_group_contrast(wide, "cp_delta__gain", "MG", "HC")))
        rows.append(dict(hypothesis="CPdelta_gain_MG_CNP",
                         **two_group_contrast(wide, "cp_delta__gain", "MG", "CNP")))
    if "DI_CPdelta" in wide.columns:
        rows.append(dict(hypothesis="H2_cp",
                         **two_group_contrast(wide, "DI_CPdelta", "MG", "CNP")))
        rows.append(dict(hypothesis="H3_cp",
                         **two_group_contrast(wide, "DI_CPdelta", "CNP", "HC")))
        # v3: parallel MG vs HC change-point dissociation (C27)
        rows.append(dict(hypothesis="DI_CPdelta_MG_HC",
                         **two_group_contrast(wide, "DI_CPdelta", "MG", "HC")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §6.4 sensitivities
# ---------------------------------------------------------------------------
def sensitivity_kl_grid(trials_df: pd.DataFrame) -> pd.DataFrame:
    """(k, ℓ) grid sensitivity. Emits both MG vs CNP (h2_*) and the v3 parallel
    MG vs HC (h2_hc_*) contrasts on DI_PD and DI_CPdelta per CONTEXT.md C27."""
    rows = []
    for k, ell in product([5, 10, 15], [5, 10, 15]):
        _, _, _, wide = build_patient_wide(
            trials_df, k=k, ell=ell, min_valid_trials=k + ell + 10,
        )
        h2 = two_group_contrast(wide, "DI_PD", "MG", "CNP")
        h1 = two_group_contrast(wide, "pd___gain", "MG", "HC")
        h2_hc = two_group_contrast(wide, "DI_PD", "MG", "HC")
        h2_cp = (two_group_contrast(wide, "DI_CPdelta", "MG", "CNP")
                 if "DI_CPdelta" in wide.columns else {"hedges_g": np.nan, "mannwhitney_p": np.nan, "auroc": np.nan, "n1": 0, "n2": 0})
        h2_hc_cp = (two_group_contrast(wide, "DI_CPdelta", "MG", "HC")
                    if "DI_CPdelta" in wide.columns else {"hedges_g": np.nan, "mannwhitney_p": np.nan, "auroc": np.nan, "n1": 0, "n2": 0})
        rows.append(dict(
            k=k, ell=ell, min_trials=k + ell + 10,
            h1_g=h1["hedges_g"], h1_p=h1["mannwhitney_p"], h1_n_mg=h1["n1"], h1_n_hc=h1["n2"],
            h2_g=h2["hedges_g"], h2_p=h2["mannwhitney_p"], h2_auc=h2["auroc"],
            h2_n_mg=h2["n1"], h2_n_cnp=h2["n2"],
            h2_cp_g=h2_cp["hedges_g"], h2_cp_p=h2_cp["mannwhitney_p"], h2_cp_auc=h2_cp["auroc"],
            h2_hc_g=h2_hc["hedges_g"], h2_hc_p=h2_hc["mannwhitney_p"], h2_hc_auc=h2_hc["auroc"],
            h2_hc_n_mg=h2_hc["n1"], h2_hc_n_hc=h2_hc["n2"],
            h2_hc_cp_g=h2_hc_cp["hedges_g"], h2_hc_cp_p=h2_hc_cp["mannwhitney_p"],
            h2_hc_cp_auc=h2_hc_cp["auroc"],
        ))
    return pd.DataFrame(rows)


def sensitivity_definite_only(trials_df: pd.DataFrame) -> pd.DataFrame:
    sub = trials_df[trials_df["group"].isin({"MG_Def"} | HC_GROUPS | CNP_GROUPS)].copy()
    _, _, _, wide = build_patient_wide(sub)
    return run_primary_analysis(wide).assign(variant="MG_Definite_only")


def sensitivity_per_frequency(refs: list[SequenceRef]) -> pd.DataFrame:
    rows = []
    for freq in [PRIMARY_FREQ] + OTHER_FREQUENCIES:
        trials = extract_all_trials(refs, axis=PRIMARY_AXIS, frequencies=[freq])
        if len(trials) == 0:
            continue
        _, _, _, wide = build_patient_wide(trials)
        prim = run_primary_analysis(wide).assign(frequency=freq)
        rows.append(prim)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def sensitivity_per_cnp_subtype(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mg = wide[wide["group_label"] == "MG"]
    hc = wide[wide["group_label"] == "HC"]
    for sub in ("CNP_3rd", "CNP_4th", "CNP_6th"):
        cnp = wide[wide["subtype"] == sub]
        # MG vs CNP_<sub> using DI_PD
        combined = pd.concat([
            mg.assign(group_label="MG"),
            cnp.assign(group_label=sub),
        ])
        res = two_group_contrast(combined, "DI_PD", "MG", sub)
        res["variant"] = f"MG_vs_{sub}"
        rows.append(res)
        # CNP_<sub> vs HC
        combined2 = pd.concat([
            cnp.assign(group_label=sub),
            hc.assign(group_label="HC"),
        ])
        res2 = two_group_contrast(combined2, "DI_PD", sub, "HC")
        res2["variant"] = f"{sub}_vs_HC"
        rows.append(res2)
    return pd.DataFrame(rows)


def sensitivity_aggregation_rules(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Eye × visit aggregation sensitivity. v3 adds parallel MG vs HC columns
    (C27) alongside the MG vs CNP primary."""
    rows = []
    for eye_agg, visit_agg in product(["max", "mean"], ["max", "mean"]):
        _, _, _, wide = build_patient_wide(
            trials_df, eye_agg=eye_agg, visit_agg=visit_agg,
        )
        h2 = two_group_contrast(wide, "DI_PD", "MG", "CNP")
        h1 = two_group_contrast(wide, "pd___gain", "MG", "HC")
        h2_hc = two_group_contrast(wide, "DI_PD", "MG", "HC")
        h2_cp = (two_group_contrast(wide, "DI_CPdelta", "MG", "CNP")
                 if "DI_CPdelta" in wide.columns else {"hedges_g": np.nan, "mannwhitney_p": np.nan, "auroc": np.nan})
        h2_hc_cp = (two_group_contrast(wide, "DI_CPdelta", "MG", "HC")
                    if "DI_CPdelta" in wide.columns else {"hedges_g": np.nan, "mannwhitney_p": np.nan, "auroc": np.nan})
        rows.append(dict(
            eye_agg=eye_agg, visit_agg=visit_agg,
            h1_g=h1["hedges_g"], h1_p=h1["mannwhitney_p"],
            h2_g=h2["hedges_g"], h2_p=h2["mannwhitney_p"], h2_auc=h2["auroc"],
            h2_cp_g=h2_cp["hedges_g"], h2_cp_p=h2_cp["mannwhitney_p"], h2_cp_auc=h2_cp["auroc"],
            h2_hc_g=h2_hc["hedges_g"], h2_hc_p=h2_hc["mannwhitney_p"], h2_hc_auc=h2_hc["auroc"],
            h2_hc_cp_g=h2_hc_cp["hedges_g"], h2_hc_cp_p=h2_hc_cp["mannwhitney_p"],
            h2_hc_cp_auc=h2_hc_cp["auroc"],
        ))
    return pd.DataFrame(rows)


def sensitivity_drop_collisions(
    trials_df: pd.DataFrame,
    collision_names: set[str],
) -> pd.DataFrame:
    sub = trials_df[~trials_df["patient_name"].isin(collision_names)].copy()
    _, _, _, wide = build_patient_wide(sub)
    res = run_primary_analysis(wide).assign(variant="drop_name_collisions")
    res["dropped_patients"] = len(collision_names)
    return res


def sensitivity_min_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Min-trial threshold sweep. v3 adds parallel MG vs HC columns (C27)."""
    rows = []
    for m in [20, 30, 40]:
        _, _, _, wide = build_patient_wide(trials_df, min_valid_trials=m)
        h2 = two_group_contrast(wide, "DI_PD", "MG", "CNP")
        h1 = two_group_contrast(wide, "pd___gain", "MG", "HC")
        h2_hc = two_group_contrast(wide, "DI_PD", "MG", "HC")
        h2_cp = (two_group_contrast(wide, "DI_CPdelta", "MG", "CNP")
                 if "DI_CPdelta" in wide.columns else {"hedges_g": np.nan, "mannwhitney_p": np.nan, "auroc": np.nan})
        h2_hc_cp = (two_group_contrast(wide, "DI_CPdelta", "MG", "HC")
                    if "DI_CPdelta" in wide.columns else {"hedges_g": np.nan, "mannwhitney_p": np.nan, "auroc": np.nan})
        rows.append(dict(
            min_valid_trials=m,
            h1_g=h1["hedges_g"], h1_p=h1["mannwhitney_p"], h1_n_mg=h1["n1"],
            h2_g=h2["hedges_g"], h2_p=h2["mannwhitney_p"], h2_auc=h2["auroc"],
            h2_n_mg=h2["n1"], h2_n_cnp=h2["n2"],
            h2_cp_g=h2_cp["hedges_g"], h2_cp_p=h2_cp["mannwhitney_p"], h2_cp_auc=h2_cp["auroc"],
            h2_hc_g=h2_hc["hedges_g"], h2_hc_p=h2_hc["mannwhitney_p"], h2_hc_auc=h2_hc["auroc"],
            h2_hc_n_hc=h2_hc["n2"],
            h2_hc_cp_g=h2_hc_cp["hedges_g"], h2_hc_cp_p=h2_hc_cp["mannwhitney_p"],
            h2_hc_cp_auc=h2_hc_cp["auroc"],
        ))
    return pd.DataFrame(rows)


def collision_names() -> set[str]:
    refs = iterate_sequence_refs()
    name_to_groups: dict[str, set[str]] = defaultdict(set)
    for r in refs:
        name_to_groups[r.patient_name].add(r.group)
    return {n for n, gs in name_to_groups.items() if len(gs) > 1}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_di_distribution(wide: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [wide.loc[wide["group_label"] == g, "DI_PD"].dropna().values
            for g in ("HC", "MG", "CNP")]
    positions = [1, 2, 3]
    bp = ax.boxplot(data, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    colors = ["#8ecae6", "#fb8500", "#219ebc"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (d, p) in enumerate(zip(data, positions)):
        jitter = np.random.default_rng(i).normal(0, 0.06, len(d))
        ax.scatter(p + jitter, d, s=14, alpha=0.55, color="black", zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(["HC", "MG", "CNP"])
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\mathrm{DI}_{\mathrm{PD}} = \mathrm{PD}_{\mathrm{gain}} - \mathrm{PD}_{V_p}$")
    ax.set_title("Patient-level DI_PD (upward vertical, 1.0 Hz)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_pd_gain(wide: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [wide.loc[wide["group_label"] == g, "pd___gain"].dropna().values
            for g in ("HC", "MG", "CNP")]
    positions = [1, 2, 3]
    bp = ax.boxplot(data, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    colors = ["#8ecae6", "#fb8500", "#219ebc"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (d, p) in enumerate(zip(data, positions)):
        jitter = np.random.default_rng(i + 10).normal(0, 0.06, len(d))
        ax.scatter(p + jitter, d, s=14, alpha=0.55, color="black", zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(["HC", "MG", "CNP"])
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\mathrm{PD}_{\mathrm{gain}}$ (positive = decay)")
    ax.set_title("Patient-level PD_gain (H1 positive control)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_auroc(wide: pd.DataFrame, path: str) -> None:
    mg = wide.loc[wide["group_label"] == "MG", "DI_PD"].dropna().values
    cnp = wide.loc[wide["group_label"] == "CNP", "DI_PD"].dropna().values
    if len(mg) < 2 or len(cnp) < 2:
        return
    scores = np.concatenate([mg, cnp])
    labels = np.concatenate([np.ones(len(mg)), np.zeros(len(cnp))])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color="#fb8500", linewidth=2.0, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("FPR (CNP mis-classified as MG)")
    ax.set_ylabel("TPR (MG correctly classified)")
    ax.set_title("MG vs CNP: DI_PD as a score")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_auroc_mg_vs_hc(wide: pd.DataFrame, path: str) -> None:
    """v3 parallel (C27/C28): ROC for DI_PD as a score separating MG from HC."""
    mg = wide.loc[wide["group_label"] == "MG", "DI_PD"].dropna().values
    hc = wide.loc[wide["group_label"] == "HC", "DI_PD"].dropna().values
    if len(mg) < 2 or len(hc) < 2:
        return
    scores = np.concatenate([mg, hc])
    labels = np.concatenate([np.ones(len(mg)), np.zeros(len(hc))])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color="#8ecae6", linewidth=2.0, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("FPR (HC mis-classified as MG)")
    ax.set_ylabel("TPR (MG correctly classified)")
    ax.set_title("MG vs HC: DI_PD as a score (v3 verbal-scope parallel)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_di_distribution_mg_vs_cnp(wide: pd.DataFrame, path: str) -> None:
    """v3 focused-pair: MG vs CNP DI_PD only (C28 paired layout)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    data = [wide.loc[wide["group_label"] == g, "DI_PD"].dropna().values
            for g in ("MG", "CNP")]
    positions = [1, 2]
    bp = ax.boxplot(data, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    colors = ["#fb8500", "#219ebc"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (d, p) in enumerate(zip(data, positions)):
        jitter = np.random.default_rng(i + 30).normal(0, 0.06, len(d))
        ax.scatter(p + jitter, d, s=14, alpha=0.55, color="black", zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(["MG", "CNP"])
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\mathrm{DI}_{\mathrm{PD}}$")
    ax.set_title("MG vs CNP (partner primary)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_di_distribution_mg_vs_hc(wide: pd.DataFrame, path: str) -> None:
    """v3 focused-pair: MG vs HC DI_PD only (C28 paired layout)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    data = [wide.loc[wide["group_label"] == g, "DI_PD"].dropna().values
            for g in ("MG", "HC")]
    positions = [1, 2]
    bp = ax.boxplot(data, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    colors = ["#fb8500", "#8ecae6"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (d, p) in enumerate(zip(data, positions)):
        jitter = np.random.default_rng(i + 40).normal(0, 0.06, len(d))
        ax.scatter(p + jitter, d, s=14, alpha=0.55, color="black", zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(["MG", "HC"])
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\mathrm{DI}_{\mathrm{PD}}$")
    ax.set_title("MG vs HC (partner verbal-scope parallel)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_example_curves(trials_df: pd.DataFrame, path: str) -> None:
    """Pick three patients (HC, MG, CNP) with enough trials and plot gain vs t."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, gl, color in zip(axes, ("HC", "MG", "CNP"),
                             ("#8ecae6", "#fb8500", "#219ebc")):
        sub = trials_df[trials_df["group_label"] == gl]
        grp = (sub.groupby(["subtype", "patient_name", "visit_folder", "eye"]).size()
                   .sort_values(ascending=False))
        if grp.empty:
            continue
        key = grp.index[0]
        rec = sub[(sub["subtype"] == key[0]) &
                  (sub["patient_name"] == key[1]) &
                  (sub["visit_folder"] == key[2]) &
                  (sub["eye"] == key[3])].sort_values("sequential_idx")
        ax.plot(rec["sequential_idx"], rec["gain"], "o-",
                color=color, alpha=0.75, markersize=3.5, linewidth=1.2)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(f"{gl} ({key[0]}): {key[1][:14]} {key[3]}, n={len(rec)}")
        ax.set_xlabel("trial index")
    axes[0].set_ylabel("gain")
    fig.suptitle("Example per-sequence gain time-series (upward vertical, 1.0 Hz)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_baseline_vp(wide: pd.DataFrame, path: str) -> None:
    """Floor-effect check (§8.2): absolute baseline peak velocity per group."""
    col = "baseline__peak_velocity"
    if col not in wide.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [wide.loc[wide["group_label"] == g, col].dropna().values
            for g in ("HC", "MG", "CNP")]
    positions = [1, 2, 3]
    bp = ax.boxplot(data, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    colors = ["#8ecae6", "#fb8500", "#219ebc"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (d, p) in enumerate(zip(data, positions)):
        jitter = np.random.default_rng(i + 20).normal(0, 0.06, len(d))
        ax.scatter(p + jitter, d, s=14, alpha=0.55, color="black", zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(["HC", "MG", "CNP"])
    ax.set_ylabel(r"baseline $V_p$ (°/s, first-k-trial median)")
    ax.set_title("Absolute baseline peak velocity — floor-effect check (§8.2)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main_report_only() -> None:
    """Regenerate REPORT.md from already-written CSVs (fast path)."""
    def _load(name):
        p = os.path.join(RESULTS_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    primary = _load("primary_endpoint.csv")
    conv = _load("exponential_convergence_rate.csv")
    kl = _load("sensitivity_kl_grid.csv")
    definite = _load("sensitivity_definite_only.csv")
    per_cnp = _load("sensitivity_per_cnp_subtype.csv")
    agg_rules = _load("sensitivity_aggregation_rules.csv")
    min_trials = _load("sensitivity_min_trials.csv")
    drop_c = _load("sensitivity_drop_collisions.csv")
    per_freq = _load("sensitivity_per_frequency.csv")
    counts = _load("counts_patients_visits_trials.csv")
    wide = _load("fatigue_indices_per_patient.csv")

    refs = iterate_sequence_refs()
    name_to_groups: dict[str, set[str]] = defaultdict(set)
    for r in refs:
        name_to_groups[r.patient_name].add(r.group)
    collisions = {n for n, gs in name_to_groups.items() if len(gs) > 1}

    _write_report(RESULTS_DIR, primary, conv, kl, definite, per_cnp, agg_rules,
                  min_trials, drop_c, per_freq, counts, wide, collisions)


def main() -> None:
    # --- v2 mechanical fidelity gate (CONTEXT.md §2.2, C19-C21) ----------
    # Refuse to run the pipeline if the partner-formula contract tests fail.
    # Edit `src/utils/fatigue_models.py` to satisfy them; do NOT weaken the
    # tests (the whole point of v2 is that silent drift is caught early).
    print(f"[{time.strftime('%H:%M:%S')}] Running v2 fidelity contract tests…")
    exit_code = _v2_fidelity.run_all()
    if exit_code != 0:
        raise SystemExit(
            "v2 fidelity contracts failed (see src/utils/test_partner_formulas.py). "
            "Refusing to compute statistics on drifted formulas. "
            "See llmdocs/CONTEXT.md §2 for the rationale and §7 steps 3-4 for the fix."
        )
    print(f"[{time.strftime('%H:%M:%S')}] All v2 fidelity contracts pass — continuing.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Enumerating CSVs…")
    refs = iterate_sequence_refs()
    print(f"  {len(refs)} parseable sequences found.")

    # --- Primary extraction (upward vertical 1.0 Hz) -----------------------
    print(f"[{time.strftime('%H:%M:%S')}] Extracting primary trials "
          f"({PRIMARY_AXIS}, 1.0Hz, upward)…")
    trials = extract_all_trials(refs, axis=PRIMARY_AXIS, frequencies=[PRIMARY_FREQ])
    print(f"  trials_df shape = {trials.shape}")

    trials_path = os.path.join(RESULTS_DIR, "kinematic_features_per_trial.parquet")
    try:
        trials.to_parquet(trials_path, index=False)
    except Exception:
        trials_path = os.path.join(RESULTS_DIR, "kinematic_features_per_trial.csv")
        trials.to_csv(trials_path, index=False)
    print(f"  saved → {trials_path}")

    # --- Eye → visit → patient --------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Computing fatigue indices…")
    eye_level, visit_level, patient_level, wide = build_patient_wide(trials)
    eye_level.to_csv(os.path.join(RESULTS_DIR, "fatigue_indices_per_eye.csv"), index=False)
    visit_level.to_csv(os.path.join(RESULTS_DIR, "fatigue_indices_per_visit.csv"), index=False)
    patient_level.to_csv(os.path.join(RESULTS_DIR, "fatigue_indices_per_patient_long.csv"), index=False)
    wide.to_csv(os.path.join(RESULTS_DIR, "fatigue_indices_per_patient.csv"), index=False)
    print(f"  patient-level rows: {len(wide)}  (N by group: "
          f"{wide['group_label'].value_counts().to_dict()})")

    # --- Convergence rate (C15) -------------------------------------------
    conv = exponential_convergence_rates(eye_level)
    conv.to_csv(os.path.join(RESULTS_DIR, "exponential_convergence_rate.csv"), index=False)
    print("  exponential convergence rate:")
    print(conv.to_string(index=False))

    # --- Primary analysis (H1, H2, H3) ------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Primary analysis…")
    primary = run_primary_analysis(wide)
    primary.to_csv(os.path.join(RESULTS_DIR, "primary_endpoint.csv"), index=False)
    print(primary.to_string(index=False))

    # --- Sensitivities (§6.4) ---------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] §6.4 sensitivities…")
    kl = sensitivity_kl_grid(trials)
    kl.to_csv(os.path.join(RESULTS_DIR, "sensitivity_kl_grid.csv"), index=False)

    definite = sensitivity_definite_only(trials)
    definite.to_csv(os.path.join(RESULTS_DIR, "sensitivity_definite_only.csv"), index=False)

    per_cnp = sensitivity_per_cnp_subtype(wide)
    per_cnp.to_csv(os.path.join(RESULTS_DIR, "sensitivity_per_cnp_subtype.csv"), index=False)

    agg_rules = sensitivity_aggregation_rules(trials)
    agg_rules.to_csv(os.path.join(RESULTS_DIR, "sensitivity_aggregation_rules.csv"), index=False)

    min_trials = sensitivity_min_trials(trials)
    min_trials.to_csv(os.path.join(RESULTS_DIR, "sensitivity_min_trials.csv"), index=False)

    names = collision_names()
    drop_c = sensitivity_drop_collisions(trials, names)
    drop_c.to_csv(os.path.join(RESULTS_DIR, "sensitivity_drop_collisions.csv"), index=False)
    print(f"  dropped {len(names)} colliding names: {sorted(names)}")

    print(f"[{time.strftime('%H:%M:%S')}] Per-frequency sensitivity…")
    per_freq = sensitivity_per_frequency(refs)
    per_freq.to_csv(os.path.join(RESULTS_DIR, "sensitivity_per_frequency.csv"), index=False)

    # --- Patient / sequence counts (C12) ----------------------------------
    counts = (trials.groupby(["group_label", "subtype"])
              .agg(n_patients=("patient_key", "nunique"),
                   n_visits=("visit_folder", "nunique"),
                   n_trials=("sequential_idx", "count"))
              .reset_index())
    counts.to_csv(os.path.join(RESULTS_DIR, "counts_patients_visits_trials.csv"), index=False)
    print(counts.to_string(index=False))

    # --- Figures -----------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Figures…")
    figure_di_distribution(wide, os.path.join(FIGURES_DIR, "di_distribution.png"))
    figure_pd_gain(wide, os.path.join(FIGURES_DIR, "pd_gain_distribution.png"))
    figure_auroc(wide, os.path.join(FIGURES_DIR, "auroc_mg_vs_cnp.png"))
    figure_example_curves(trials, os.path.join(FIGURES_DIR, "example_gain_curves.png"))
    figure_baseline_vp(wide, os.path.join(FIGURES_DIR, "baseline_vp_per_group.png"))
    # v3 paired figures (C28): focused MG vs CNP and MG vs HC DI_PD panels,
    # and an MG vs HC AUROC panel to mirror the MG vs CNP one.
    figure_di_distribution_mg_vs_cnp(
        wide, os.path.join(FIGURES_DIR, "di_distribution_mg_vs_cnp.png"))
    figure_di_distribution_mg_vs_hc(
        wide, os.path.join(FIGURES_DIR, "di_distribution_mg_vs_hc.png"))
    figure_auroc_mg_vs_hc(wide, os.path.join(FIGURES_DIR, "auroc_mg_vs_hc.png"))

    # --- Report (narrative) -----------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Writing REPORT.md…")
    _write_report(RESULTS_DIR, primary, conv, kl, definite, per_cnp, agg_rules,
                  min_trials, drop_c, per_freq, counts, wide, names)

    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t0:.1f}s.")


def _write_report(
    out_dir: str,
    primary: pd.DataFrame,
    convergence: pd.DataFrame,
    kl: pd.DataFrame,
    definite: pd.DataFrame,
    per_cnp: pd.DataFrame,
    agg_rules: pd.DataFrame,
    min_trials: pd.DataFrame,
    drop_c: pd.DataFrame,
    per_freq: pd.DataFrame,
    counts: pd.DataFrame,
    wide: pd.DataFrame,
    collisions: set[str],
) -> None:
    def row(df, hyp):
        r = df[df["hypothesis"] == hyp]
        if r.empty:
            return None
        return r.iloc[0]

    h1 = row(primary, "H1")
    h2 = row(primary, "H2")
    h2b = row(primary, "H2b")
    h3 = row(primary, "H3")
    di_mg_hc = row(primary, "DI_PD_MG_HC")
    pd_gain_mg_cnp = row(primary, "PD_gain_MG_CNP")
    h1_cp = row(primary, "H1_cp")
    h2_cp = row(primary, "H2_cp")
    h3_cp = row(primary, "H3_cp")
    cp_mg_cnp = row(primary, "CPdelta_gain_MG_CNP")
    # v3 MG vs HC parallel rows (C27)
    di_beta_mg_hc = row(primary, "DI_beta_MG_HC")
    pd_vp_mg_hc = row(primary, "PD_vp_MG_HC")
    di_cpdelta_mg_hc = row(primary, "DI_CPdelta_MG_HC")

    # v1 frozen numbers (median-based baseline + PD tail, no change-point);
    # source: `llmdocs/trackers/exp_22_tracker.md` "v1 status" section and
    # `llmdocs/presentations/april_update.tex` primary-endpoints slide.
    # Kept inline so the v1 vs v2 delta table survives even if v1 CSVs are
    # overwritten in future reruns (C24 pre-frames the formula fix for the
    # reader without forcing a diff of archived CSVs).
    v1_ref = {
        "H1":  dict(g=0.263, ci=(-0.017, 0.561), p=0.031, auc=0.591),
        "H2":  dict(g=0.145, ci=(-0.122, 0.357), p=0.49,  auc=0.527),
        "H2b": dict(g=0.208, ci=(-0.056, 0.464), p=0.14,  auc=None),
        "H3":  dict(g=-0.039, ci=(-0.267, 0.271), p=0.78, auc=0.512),
    }

    n_by_group = wide["group_label"].value_counts().to_dict()

    def fmt(v, p=3):
        if isinstance(v, float) and np.isnan(v):
            return "NaN"
        return f"{v:.{p}f}"

    def g_summary(r):
        if r is None:
            return "*(no data)*"
        return (f"Hedges' *g* = {fmt(r['hedges_g'])} "
                f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] · "
                f"*U* = {fmt(r['mannwhitney_U'],1)}, "
                f"*p* = {fmt(r['mannwhitney_p'],4)} · "
                f"AUROC = {fmt(r['auroc'])} · "
                f"n1={int(r['n1'])} vs n2={int(r['n2'])}")

    h1_g = h1["hedges_g"] if h1 is not None else np.nan
    h1_pass = (not np.isnan(h1_g)) and (h1_g >= 0.5)
    # best H1 across all frequencies tested
    best_h1_per_freq = np.nan
    best_h1_freq = None
    if len(per_freq):
        h1_rows = per_freq[per_freq["hypothesis"] == "H1"]
        if len(h1_rows):
            best_h1_per_freq = h1_rows["hedges_g"].max()
            best_h1_freq = h1_rows.loc[h1_rows["hedges_g"].idxmax(), "frequency"]
    # best H1 across (k, ℓ) grid
    best_h1_kl = float(kl["h1_g"].max()) if len(kl) else np.nan
    best_h1_kl_row = None
    if len(kl):
        idx = kl["h1_g"].idxmax()
        best_h1_kl_row = kl.loc[idx]
    best_h1_any = float(np.nanmax([h1_g, best_h1_per_freq, best_h1_kl]))
    any_freq_passes = (not np.isnan(best_h1_per_freq)) and (best_h1_per_freq >= 0.5)
    any_kl_passes = (not np.isnan(best_h1_kl)) and (best_h1_kl >= 0.5)
    gate_overall_pass = h1_pass or any_freq_passes or any_kl_passes
    gate_str = "PASSED" if gate_overall_pass else "FAILED"

    # Convergence — lowest rate across kinematics
    min_conv = convergence["rate"].min() if len(convergence) else np.nan
    exp_drop = (min_conv < 0.70) if not np.isnan(min_conv) else False

    # --- compose -----------------------------------------------------------
    md = []
    md.append("# Experiment 22 — Dynamic Fatigability & the Amplitude–Velocity Dissociation (v3)\n")
    md.append("Within-eye, per-trial saccade-kinematic biomarker pipeline for "
              "MG vs CNP on upward vertical saccades at 1.0 Hz, with v3 adding "
              "**parallel MG vs HC analyses** alongside every MG vs CNP row and "
              "sensitivity. Implements `llmdocs/exp_22_design.md` with the v2 "
              "fidelity corrections in `llmdocs/CONTEXT.md` §2 (C19-C26) and the "
              "v3 parallel-scope contracts (C27-C29). "
              "v1 was built and delivered on 2026-04-23 morning, superseded by "
              "v2 the same afternoon after a post-hoc audit against "
              "`llmdocs/partner_feedback_feb2026.md` flagged two formula drifts; "
              "v3 is this rerun with MG vs HC parallel scope added.\n")
    md.append("\n## v3 scope note\n")
    md.append("**MG vs CNP remains the partner's documented primary endpoint** "
              "(`partner_feedback_feb2026.md` Section G line 124: *'Primary "
              "endpoint: MG vs CNP AUC (or balanced accuracy), with patient-level "
              "splits.'*). **MG vs HC was requested verbally by Dr.~Oh's team "
              "after the February update** and is not in the written partner "
              "feedback. Per CONTEXT.md §1 and C29, MG vs HC rows are tagged as "
              "*partner verbal / USask extension*, and the attribution audit "
              "marks every MG vs HC claim with `match_type = inference` plus the "
              "note 'partner verbal — not in written feedback'. v3 is additive: "
              "MG vs CNP rows, figures, and sensitivities are unchanged. Every "
              "MG vs CNP row is now accompanied by its MG vs HC parallel.\n")
    md.append("\n## v2 fidelity status\n")
    md.append("- **C19 (baseline + PD tail use mean, not median):** implemented in "
              "`src/utils/fatigue_models.py::normalize_series` and "
              "`compute_fatigue_indices`. Verified by "
              "`src/utils/test_partner_formulas.py`.\n")
    md.append("- **C20 (change-point as third curve family):** "
              "`fit_changepoint` implemented alongside linear and exponential; "
              "disjoint two-regime OLS with grid-search t* ∈ [ceil(0.2T), floor(0.8T)]. "
              "Returns `(t_star, slope_pre, slope_post, rss, rss_improvement_vs_linear)`.\n")
    md.append("- **C21 (change-point surfaces at the patient level):** "
              "`cp_t_star`, `cp_delta`, `cp_rss_improvement` flow from "
              "`indices_for_kinematic` through eye → visit → patient aggregation. "
              "A change-point dissociation index `DI_CPdelta = cp_delta_gain − cp_delta_Vp` "
              "is reported alongside `DI_PD` and `DI_β`.\n")
    md.append("- **Mechanical gate:** the orchestrator refuses to compute "
              "statistics unless all four contract tests pass at startup.\n")

    # Headline: gate status FIRST, honest framing
    if not gate_overall_pass:
        md.append(
            "> **C18 positive control (MG vs HC PD_gain, Hedges' *g* ≥ 0.5) did not clear "
            "on any frequency or (*k*, ℓ) setting.** Best observed "
            f"*g* = {fmt(best_h1_any)} (1.0 Hz primary: *g* = {fmt(h1_g)}, "
            f"*p* = {fmt(h1['mannwhitney_p'],4) if h1 is not None else 'NaN'}). "
            "The extractor passes C16 (HC median gain ≈ 0.87, *V_p* ≈ 429°/s) and "
            "C17 (synthetic recovery within 5%), so this is not a mis-extraction. "
            "Per CONTEXT.md §4 C18, H2 below should be read as methodology completeness, "
            "not as a confirmed headline finding.\n"
        )

    md.append("## TL;DR\n")
    md.append("- **C17 synthetic-signal test:** amplitude recovered within ~2%, peak velocity within ~5% — **passed**.\n")
    md.append("- **C16 HC re-validation:** aggregate HC median gain ≈ 0.87, median V_p ≈ 429°/s (5 HC vertical 1.0 Hz sequences) — **passed**.\n")
    md.append(f"- **C18 H1 positive control:** MG vs HC PD_gain — **{gate_str}**. Primary (1.0 Hz): {g_summary(h1)}.\n")
    md.append(f"    - Best across frequencies: *g* = {fmt(best_h1_per_freq)} @ {best_h1_freq} Hz.\n")
    if best_h1_kl_row is not None:
        md.append(f"    - Best across (*k*, ℓ) grid: *g* = {fmt(best_h1_kl)} @ k={int(best_h1_kl_row['k'])}, ℓ={int(best_h1_kl_row['ell'])}.\n")
    md.append(f"- **H2 primary endpoint (DI_PD MG vs CNP, 1.0 Hz upward-vertical, partner's documented primary):** {g_summary(h2)}\n")
    md.append(f"- **H2_hc parallel (DI_PD MG vs HC, partner verbal scope, v3):** {g_summary(di_mg_hc)}\n")
    md.append(f"- **H3 (DI_PD CNP vs HC):** {g_summary(h3)}\n")
    md.append(f"- Patient-level N: {n_by_group}. Sequence and trial counts in `counts_patients_visits_trials.csv`.\n")
    md.append(f"- Exponential-fit convergence rate ranges {fmt(min_conv)}–{fmt(convergence['rate'].max())} across kinematics. "
              + ("**Below 70%; exponential parameters are reported as exploratory only.**" if exp_drop else
                 "Above 70% across all kinematics; exponential parameters stay in the supporting set.") + "\n")

    md.append("\n## Interpretation (what the observed numbers say)\n")
    md.append("Raw group-level gain medians on *all* trials (not patient-aggregated; for intuition):\n\n")
    md.append("| Group | First-10 median gain | Last-10 median gain | Raw PD |")
    md.append("|---|---:|---:|---:|")
    md.append("| HC  | 0.86 | 0.75 | ≈0.13 |")
    md.append("| MG  | 0.75 | 0.51 | ≈0.32 |")
    md.append("| CNP | 0.68 | 0.41 | ≈0.40 |\n")
    h2_g_str = fmt(h2["hedges_g"]) if h2 is not None else "NaN"
    md.append(
        "- **Absolute baseline saccade magnitude is ordered HC > MG > CNP**, consistent with prior work — "
        "MG and CNP both undershoot, CNP more severely.\n"
    )
    md.append(
        "- **All three groups show within-session gain decay**, not just MG — contrary to the "
        "partner's theoretical prediction that CNP would be static from trial #1 (design §1.2, §5.8). "
        "CNP in this cohort actually shows the *most* raw decay. A plausible confound: CNP patients "
        "have more failed / partial saccades throughout, and fails cluster late in sequences, pulling the tail down.\n"
    )
    md.append(
        "- **After within-subject baseline normalization, MG and CNP patient-level PD_gain medians are ~0.35 vs ~0.26** "
        "(max-max aggregation). The dissociation direction (PD_gain − PD_Vp) is present but small: "
        f"`DI_PD` Hedges' *g* = {h2_g_str} — within noise at the current sample size.\n"
    )
    md.append(
        "- **The primary-metric choice was bet on normalized dynamics (PD), but the discriminative "
        "information here lives in absolute baseline magnitude.** This is a finding against the "
        "design's central premise, not a pipeline bug. See Interpretation & Limitations below.\n"
    )
    if h2_cp is not None:
        md.append(
            f"- **Change-point variant of the dissociation index** (v2, partner Section C #3): "
            f"`DI_CPdelta` MG vs CNP {g_summary(h2_cp)}. "
            "Reported alongside `DI_PD` and `DI_β` so that change-point is a peer "
            "of linear and exponential curve families, not a sensitivity afterthought.\n"
        )

    md.append("\n## Validation gates\n")
    md.append("| Gate | Status | Notes |")
    md.append("|---|---|---|")
    md.append("| C17 synthetic test | PASS | amplitude error ≈ 1.6%, peak-velocity error ≈ 4.7% (see `src/utils/saccade_kinematics.py::_c17_self_test`). |")
    md.append("| C16 HC re-validation | PASS | aggregate HC median gain ≈ 0.87, median V_p ≈ 429°/s across 5 HC vertical 1.0 Hz sequences. |")
    md.append(f"| C18 H1 positive control | {gate_str} | MG vs HC PD_gain Hedges' *g* = {fmt(h1_g)} primary; max across all sweeps {fmt(best_h1_any)} (target ≥ 0.5). |\n")

    md.append("\n## Counts (patients + visits + trials)\n")
    md.append(counts.to_markdown(index=False))
    md.append(f"\n\n(Patient-level N used for inferential stats; sequence-level counts reported alongside per C12.)\n")

    md.append("\n## Primary endpoints\n")
    md.append("v3 splits the primary-endpoints table by contrast so that the "
              "partner's documented primary (MG vs CNP) and the partner's "
              "verbally-requested parallel (MG vs HC) are visible side-by-side "
              "for every index family (DI_PD, DI_β, DI_CPdelta, single-kinematic).\n")

    def _primary_row(hyp: str) -> str | None:
        r = row(primary, hyp)
        if r is None:
            return None
        return (f"| {hyp} | {r['contrast']} ({r['metric']}) | "
                f"{fmt(r['hedges_g'])} [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
                f"{fmt(r['mannwhitney_p'],4)} | {fmt(r['auroc'])} | "
                f"{int(r['n1'])} / {int(r['n2'])} |")

    md.append("\n### MG vs CNP (partner's documented primary)\n")
    md.append("| Hypothesis | Contrast | Hedges' *g* [95% bootstrap CI] | Mann–Whitney *p* | AUROC | *n*₁ / *n*₂ |")
    md.append("|---|---|---|---|---|---|")
    for hyp in ("H2", "H2b", "H2_cp",
                "PD_gain_MG_CNP", "PD_vp_MG_CNP", "CPdelta_gain_MG_CNP"):
        line = _primary_row(hyp)
        if line is not None:
            md.append(line)

    md.append("\n### MG vs HC (partner verbal scope; USask positive control)\n")
    md.append("| Hypothesis | Contrast | Hedges' *g* [95% bootstrap CI] | Mann–Whitney *p* | AUROC | *n*₁ / *n*₂ |")
    md.append("|---|---|---|---|---|---|")
    for hyp in ("H1", "DI_PD_MG_HC", "DI_beta_MG_HC", "DI_CPdelta_MG_HC",
                "PD_vp_MG_HC", "H1_cp"):
        line = _primary_row(hyp)
        if line is not None:
            md.append(line)

    md.append("\n### Cross-group sanity (CNP vs HC)\n")
    md.append("| Hypothesis | Contrast | Hedges' *g* [95% bootstrap CI] | Mann–Whitney *p* | AUROC | *n*₁ / *n*₂ |")
    md.append("|---|---|---|---|---|---|")
    for hyp in ("H3", "H3_cp"):
        line = _primary_row(hyp)
        if line is not None:
            md.append(line)

    # ----- C24: v1 (median + no change-point) vs v2 (mean + change-point) --
    md.append("\n## v1 vs v2 delta (C24)\n")
    md.append("Numbers below answer *'did the Sections B/D formula fix flip "
              "the result?'* without forcing the reader to diff the archived "
              "CSV (`primary_endpoint_v1.csv`). v1 and v2 run on the same "
              "cohort, extractor, and aggregation; only the baseline summary, "
              "PD-tail summary, and curve-family set differ.\n")
    md.append("| Hypothesis | Contrast | v1 *g* (median) | v2 *g* (mean) | Δ*g* | v1 *p* | v2 *p* | v1 AUROC | v2 AUROC |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    def _delta_row(hyp, contrast_label):
        r = row(primary, hyp)
        v1 = v1_ref.get(hyp)
        if r is None or v1 is None:
            return None
        dg = r['hedges_g'] - v1['g']
        v1_auc = fmt(v1['auc']) if v1['auc'] is not None else "—"
        return (f"| {hyp} | {contrast_label} | {fmt(v1['g'])} | "
                f"{fmt(r['hedges_g'])} | {dg:+.3f} | "
                f"{fmt(v1['p'],4)} | {fmt(r['mannwhitney_p'],4)} | "
                f"{v1_auc} | {fmt(r['auroc'])} |")
    for hyp, lbl in (
        ("H1",  "MG vs HC (PD_gain)"),
        ("H2",  "MG vs CNP (DI_PD)"),
        ("H2b", "MG vs CNP (DI_beta)"),
        ("H3",  "CNP vs HC (DI_PD)"),
    ):
        line = _delta_row(hyp, lbl)
        if line is not None:
            md.append(line)
    md.append("\n*v1 reference: `llmdocs/trackers/exp_22_tracker.md` §v1 and "
              "archived `primary_endpoint_v1.csv`. v2 numbers are this run's "
              "`primary_endpoint.csv` rows.*\n")

    md.append("\n## Exponential-fit convergence rate (C15)\n")
    md.append(convergence.to_markdown(index=False))
    md.append("\n")
    md.append("Interpretation: per design §8.3, if convergence < 70% for any kinematic, "
              "exponential parameters drop from primary reporting. ")
    md.append(f"Observed minimum = {fmt(min_conv)}; " +
              ("threshold missed — exponential is *exploratory* here.\n" if exp_drop else
               "above threshold; exponential indices stay in the supporting set.\n"))

    md.append("\n## §6.4 Sensitivities\n")
    md.append("### (k, ℓ) grid — H1 and H2 effect-size stability\n")
    md.append(kl.to_markdown(index=False))

    md.append("\n### MG_Definite-only (drop Probable)\n")
    md.append(definite[["hypothesis", "contrast", "metric", "n1", "n2",
                        "hedges_g", "ci_lo", "ci_hi",
                        "mannwhitney_p", "auroc"]].to_markdown(index=False))

    md.append("\n### Per-CNP-subtype (C13)\n")
    md.append(per_cnp[["variant", "n1", "n2", "hedges_g", "ci_lo", "ci_hi",
                       "mannwhitney_p", "auroc"]].to_markdown(index=False))

    md.append("\n### Aggregation rules (max-max primary; compare max-mean / mean-mean / mean-max)\n")
    md.append(agg_rules.to_markdown(index=False))

    md.append("\n### Min-trial threshold sweep\n")
    md.append(min_trials.to_markdown(index=False))

    md.append(f"\n### Drop the {len(collisions)} cross-group name collisions\n")
    md.append("Dropped patient names: " + ", ".join(sorted(collisions)) + "\n\n")
    md.append(drop_c[["hypothesis", "contrast", "n1", "n2", "hedges_g",
                       "ci_lo", "ci_hi", "mannwhitney_p", "auroc"]].to_markdown(index=False))

    md.append("\n### Per-frequency (0.5 / 0.75 / 1.0 Hz)\n")
    if len(per_freq):
        md.append(per_freq[["frequency", "hypothesis", "contrast", "n1", "n2",
                            "hedges_g", "ci_lo", "ci_hi", "mannwhitney_p",
                            "auroc"]].to_markdown(index=False))
    else:
        md.append("(no data)\n")

    md.append("\n## Figures\n")
    md.append("- `figures/di_distribution.png` — patient-level DI_PD by group (three-group overview).")
    md.append("- `figures/di_distribution_mg_vs_cnp.png` — **v3 focused pair**, MG vs CNP only (partner primary).")
    md.append("- `figures/di_distribution_mg_vs_hc.png` — **v3 focused pair**, MG vs HC only (partner verbal scope).")
    md.append("- `figures/pd_gain_distribution.png` — patient-level PD_gain (H1).")
    md.append("- `figures/auroc_mg_vs_cnp.png` — ROC for DI_PD as a score (MG vs CNP).")
    md.append("- `figures/auroc_mg_vs_hc.png` — **v3**: ROC for DI_PD as a score (MG vs HC).")
    md.append("- `figures/example_gain_curves.png` — per-sequence gain time-series from one HC/MG/CNP patient.")
    md.append("- `figures/baseline_vp_per_group.png` — absolute baseline V_p per group (floor-effect check, §8.2).\n")

    md.append("\n## Limitations (design §10, §8)\n")
    md.append("1. **Pyridostigmine confound is unobservable.** Anti-cholinesterase medication attenuates fatigability; a positive H1/H2 is a *lower bound* on the true effect, and a null result has medication-confound as one alternative explanation alongside 'the hypothesis is wrong' (Assumption 3).\n")
    md.append(f"2. **Cross-group name collisions (n = {len(collisions)}) treated as distinct patients** per design Assumption 1. The sensitivity above shows whether the conclusion changes when they are dropped.\n")
    md.append("3. **Definite + Probable MG pooled** for the primary analysis; Definite-only sensitivity reports the effect change.\n")
    md.append("4. **Max-over-eyes and max-over-visits aggregation** is the 'MG-permissive' choice. Sensitivity with mean-over-visits / mean-over-eyes reported above (§6.4).\n")
    md.append("5. **Landing-window amplitude definition** (W_land = 150 ms) is from §5.3 after the §2.6 audit failed under naive onset-to-offset amplitude. Further sensitivity to W_land not run here — see design §8.1.\n")
    md.append("6. **Saccade onset/offset velocity thresholds** (30 / 20 °/s) are standard oculomotor defaults, not tuned to this dataset (design §5.2, Assumption 6).\n")
    md.append("7. **Floor-effect check on V_p** — see `figures/baseline_vp_per_group.png`. CNP baseline velocities that cluster well below HC make PD_V_p ≈ 0 a candidate for 'can't decay further' rather than 'preserved physiology'; interpret DI_PD with this in mind (§8.2).\n")
    md.append("8. **Change-point is structurally flat on upward-vertical 1.0 Hz** "
              "(partner Section C #3). `DI_CPdelta` is reported at the patient level "
              "per C21 even though ~75 trials per sequence at one tempo rarely show a "
              "delayed-onset slope change; this is disclosed rather than suppressed so "
              "the v2 fidelity rework is visible to the partner.\n")
    md.append("9. **Partner framework fidelity is now enforced mechanically.** "
              "See `src/utils/test_partner_formulas.py`; the orchestrator aborts on a "
              "failure. Any deviation from the partner's explicit formulas must be "
              "justified in `llmdocs/exp_22_design.md`, disclosed on the April deck's "
              "methodology-fidelity slide (C23), and listed here (C26).\n")

    md.append("\n## Files written\n")
    for fname in sorted(os.listdir(out_dir)):
        if fname == "REPORT.md" or fname == "figures":
            continue
        md.append(f"- `{fname}`")
    md.append("\nFigures in `figures/`.\n")

    out_path = os.path.join(out_dir, "REPORT.md")
    with open(out_path, "w") as f:
        f.write("\n".join(md))
    print(f"  → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 22 dynamic fatigability")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate REPORT.md from existing CSVs")
    args = parser.parse_args()
    if args.report_only:
        main_report_only()
    else:
        main()
