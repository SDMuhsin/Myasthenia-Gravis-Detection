"""Within-subject normalization + curve fits + fatigue indices (Exp 22 v2).

Implements §5.4–5.6 of `llmdocs/exp_22_design.md` with the v2 fidelity
corrections captured in `llmdocs/CONTEXT.md` §§2–3:

  - C19: baseline normalization and PD tail both use **mean** (partner
    Sections B and D), not median. The v1 median justification in design
    §5.4 is superseded.
  - C20/C21: the partner's third curve family (piecewise / change-point,
    Section C #3) is implemented as `fit_changepoint` and flows through
    `indices_for_kinematic` as `cp_t_star`, `cp_delta`,
    `cp_rss_improvement`.

Sign convention throughout: **positive = worsening**. Ratio kinematics (gain,
peak velocity) have a multiplicative baseline; additive kinematics (latency,
duration) have an additive baseline. The six indices per (series, kinematic)
are (PD, AUC_def, β1, a, 1/τ, cp_delta). NaN propagates through curve-fit
failures (C10) — *do not silently replace with zero*.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Kinematic classification (ratio vs additive)
# ---------------------------------------------------------------------------
RATIO_KINEMATICS = ("gain", "peak_velocity")
ADDITIVE_KINEMATICS = ("latency", "duration")
ALL_KINEMATICS = RATIO_KINEMATICS + ADDITIVE_KINEMATICS


def kinematic_kind(name: str) -> str:
    if name in RATIO_KINEMATICS:
        return "ratio"
    if name in ADDITIVE_KINEMATICS:
        return "additive"
    raise ValueError(f"unknown kinematic {name!r}")


# ---------------------------------------------------------------------------
# §5.4 — baseline normalization
# ---------------------------------------------------------------------------
def normalize_series(
    values: np.ndarray,
    kind: str,
    k: int = 10,
) -> tuple[float, np.ndarray]:
    """Baseline-normalize a raw kinematic series.

    Returns (baseline, X̃). Baseline is the **mean** of the first k non-NaN
    values, per partner feedback Section B:

        G̃_e(t) = G_e(t) / mean(G_e(1..k))

    (C19.) If no finite values exist in the first k, returns (NaN, all-NaN).

    Ratio kind  → X̃ = X / baseline
    Additive    → X̃ = X − baseline
    """
    arr = np.asarray(values, dtype=float)
    head = arr[:k]
    head = head[~np.isnan(head)]
    if head.size == 0:
        return np.nan, np.full_like(arr, np.nan)
    baseline = float(np.mean(head))
    if kind == "ratio":
        if baseline == 0 or not np.isfinite(baseline):
            return baseline, np.full_like(arr, np.nan)
        return baseline, arr / baseline
    if kind == "additive":
        return baseline, arr - baseline
    raise ValueError(f"kind must be ratio|additive; got {kind!r}")


# ---------------------------------------------------------------------------
# §5.5 — curve fits
# ---------------------------------------------------------------------------
def fit_linear_theilsen(y: np.ndarray) -> tuple[float, float]:
    """Theil–Sen slope + intercept on a y-series indexed by t = 0,1,…,T−1.

    Returns (β1, β0). NaN on failure (< 2 finite points)."""
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    x = np.arange(len(y), dtype=float)
    # scipy.stats.theilslopes: returns (slope, intercept, low_slope, high_slope)
    res = _stats.theilslopes(y[mask], x[mask])
    return float(res[0]), float(res[1])


def _exp_ratio(t, a, tau):
    return 1.0 - a * (1.0 - np.exp(-t / tau))


def _exp_additive(t, a, tau):
    return a * (1.0 - np.exp(-t / tau))


def fit_exponential_decay(
    y: np.ndarray,
    kind: str,
    *,
    min_points: int = 8,
    maxfev: int = 2000,
) -> tuple[float, float, bool]:
    """Non-linear LS fit of X̃(t) = 1 − a(1 − exp(−t/τ))  (ratio) or
    a(1 − exp(−t/τ))  (additive).

    Returns (a, τ, converged). Non-convergent → (NaN, NaN, False).
    Bounds: a ∈ [−0.5, 1], τ ∈ [1, 100] (trials). Finite data required.
    """
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < min_points:
        return np.nan, np.nan, False

    x = np.arange(len(y), dtype=float)[mask]
    yv = y[mask]

    if kind == "ratio":
        f = _exp_ratio
        p0 = (0.1, 20.0)
    elif kind == "additive":
        f = _exp_additive
        p0 = (0.05, 20.0)
    else:
        raise ValueError(f"kind must be ratio|additive; got {kind!r}")

    try:
        popt, _ = curve_fit(
            f, x, yv, p0=p0,
            bounds=([-0.5, 1.0], [1.0, 100.0]),
            maxfev=maxfev,
        )
        a, tau = float(popt[0]), float(popt[1])
        if not (np.isfinite(a) and np.isfinite(tau)):
            return np.nan, np.nan, False
        return a, tau, True
    except (RuntimeError, ValueError):
        return np.nan, np.nan, False


# ---------------------------------------------------------------------------
# Change-point / piecewise-linear fit (partner Section C #3; C20)
# ---------------------------------------------------------------------------
_CP_NAN = {
    "t_star": np.nan,
    "slope_pre": np.nan,
    "slope_post": np.nan,
    "rss": np.nan,
    "rss_improvement_vs_linear": np.nan,
}


def fit_changepoint(
    y: np.ndarray,
    *,
    min_points: int = 10,
    t_star_lo_frac: float = 0.2,
    t_star_hi_frac: float = 0.8,
) -> dict:
    """Piecewise-linear fit with one change point t* (C20).

    Two disjoint OLS lines are fit, one on each side of the candidate change
    point:

        t <  t*:  y = β0_pre  + β1_pre · t
        t >= t*:  y = β0_post + β1_post · t

    t* is grid-searched over ``[ceil(t_star_lo_frac·T), floor(t_star_hi_frac·T)]``
    (defaults 0.2 / 0.8 per CONTEXT.md C20). For each candidate t* both
    regimes are OLS-fit independently; the winning t* is the one with lowest
    combined residual sum of squares. The two lines do not share parameters
    so a sharp level change at t* is reproduced exactly — important for the
    "delayed-onset" step test (CONTEXT.md §2.1 test #3).

    Returns a dict with:

        t_star                    — winning change-point index (int)
        slope_pre                 — β1_pre  (slope for t < t*)
        slope_post                — β1_post (slope for t >= t*)
        rss                       — combined RSS at winning t*
        rss_improvement_vs_linear — RSS(linear) − RSS(change-point); ≥ 0

    All NaN if fewer than `min_points` finite samples or the search range is
    empty.
    """
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < max(min_points, 4):
        return dict(_CP_NAN)

    T = len(y)
    t_all = np.arange(T, dtype=float)
    t = t_all[mask]
    y_v = y[mask]

    # linear reference fit for the improvement score
    X_lin = np.column_stack([np.ones_like(t), t])
    try:
        beta_lin, *_ = np.linalg.lstsq(X_lin, y_v, rcond=None)
    except np.linalg.LinAlgError:
        return dict(_CP_NAN)
    rss_lin = float(np.sum((X_lin @ beta_lin - y_v) ** 2))

    t_lo = int(np.ceil(t_star_lo_frac * T))
    t_hi = int(np.floor(t_star_hi_frac * T))
    if t_hi <= t_lo:
        return dict(_CP_NAN)

    best: Optional[dict] = None
    best_rss = np.inf
    for t_star in range(t_lo, t_hi + 1):
        # pre regime: finite samples with t < t_star
        pre_mask = mask & (t_all < t_star)
        post_mask = mask & (t_all >= t_star)
        if int(pre_mask.sum()) < 2 or int(post_mask.sum()) < 2:
            continue

        t_pre = t_all[pre_mask]
        y_pre = y[pre_mask]
        X_pre = np.column_stack([np.ones_like(t_pre), t_pre])
        t_post = t_all[post_mask]
        y_post = y[post_mask]
        X_post = np.column_stack([np.ones_like(t_post), t_post])

        try:
            beta_pre, *_ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
            beta_post, *_ = np.linalg.lstsq(X_post, y_post, rcond=None)
        except np.linalg.LinAlgError:
            continue

        rss_pre = float(np.sum((X_pre @ beta_pre - y_pre) ** 2))
        rss_post = float(np.sum((X_post @ beta_post - y_post) ** 2))
        rss = rss_pre + rss_post
        if rss < best_rss:
            best_rss = rss
            best = {
                "t_star": int(t_star),
                "slope_pre": float(beta_pre[1]),
                "slope_post": float(beta_post[1]),
                "rss": rss,
                "rss_improvement_vs_linear": rss_lin - rss,
            }

    return best if best is not None else dict(_CP_NAN)


# ---------------------------------------------------------------------------
# §5.6 — fatigue indices (positive = worsening)
# ---------------------------------------------------------------------------
@dataclass
class FatigueIndices:
    pd_: float           # PD percent-decrement       (positive = worsening)
    auc_def: float       # cumulative deficit         (positive = worsening)
    beta1: float         # Theil-Sen slope, decay-oriented (positive = worsening)
    beta1_raw: float     # raw slope for debugging
    exp_a: float         # exponential asymptotic magnitude (positive = worsening)
    exp_rate: float      # 1/τ                        (larger = faster onset)
    exp_tau: float       # τ in trials                (for debugging)
    exp_converged: bool
    # Change-point (partner Section C #3; C20/C21) — positive cp_delta means
    # fatigue intensifies after t*. Signed to match the beta1 convention:
    #   ratio   : cp_delta = slope_pre − slope_post   (post more negative = worse)
    #   additive: cp_delta = slope_post − slope_pre   (post more positive = worse)
    cp_t_star: float                    # winning change-point index
    cp_delta: float                     # kinematic-oriented slope jump (+ = worse)
    cp_rss_improvement: float           # RSS(linear) − RSS(change-point); ≥ 0
    cp_slope_pre_raw: float             # raw fitted slope pre-break (debug)
    cp_slope_post_raw: float            # raw fitted slope post-break (debug)
    n_valid: int         # # finite points in X̃
    baseline: float


def compute_fatigue_indices(
    x_tilde: np.ndarray,
    kind: str,
    k: int = 10,
    ell: int = 10,
    baseline: float = np.nan,
) -> FatigueIndices:
    """Compute the fatigue indices from a normalized series.

    PD (percent decrement) and the baseline window both use **mean** per
    partner feedback Sections B and D (C19). The change-point fit from
    `fit_changepoint` is computed here so that patient-level aggregation
    can surface `cp_t_star` / `cp_delta` / `cp_rss_improvement` alongside
    PD, β1, and the exponential parameters (C21).
    """
    arr = np.asarray(x_tilde, dtype=float)
    T_total = int(np.isfinite(arr).sum())
    T = len(arr)

    # ----- PD: tail-vs-baseline percent decrement (partner Section D) ------
    #       PD_G = 1 − mean(G̃(T − ℓ + 1..T))                             (C19)
    if T < ell:
        pd_ = np.nan
    else:
        tail = arr[-ell:]
        tail = tail[np.isfinite(tail)]
        if tail.size == 0:
            pd_ = np.nan
        else:
            tail_mean = float(np.mean(tail))
            if kind == "ratio":
                pd_ = 1.0 - tail_mean
            else:  # additive
                pd_ = tail_mean

    # ----- AUC deficit -----------------------------------------------------
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() < 2:
        auc_def = np.nan
    else:
        x = np.arange(T, dtype=float)[finite_mask]
        y_ = arr[finite_mask]
        if kind == "ratio":
            auc_def = float(np.trapezoid(1.0 - y_, x))
        else:
            auc_def = float(np.trapezoid(y_, x))

    # ----- β1 (Theil–Sen), decay-oriented ----------------------------------
    beta1_raw, _ = fit_linear_theilsen(arr)
    if np.isnan(beta1_raw):
        beta1 = np.nan
    elif kind == "ratio":
        beta1 = -beta1_raw   # negative raw = decay → flip so positive = worsening
    else:
        beta1 = beta1_raw    # additive: positive slope = worsening

    # ----- Exponential fit -------------------------------------------------
    a, tau, conv = fit_exponential_decay(arr, kind)
    exp_rate = 1.0 / tau if (conv and tau > 0) else np.nan

    # ----- Change-point fit (partner Section C #3) -------------------------
    cp = fit_changepoint(arr)
    cp_t_star = cp["t_star"]
    slope_pre_raw = cp["slope_pre"]
    slope_post_raw = cp["slope_post"]
    if not (np.isfinite(slope_pre_raw) and np.isfinite(slope_post_raw)):
        cp_delta = np.nan
    elif kind == "ratio":
        # decay-oriented: post more negative than pre → positive cp_delta
        cp_delta = float(slope_pre_raw - slope_post_raw)
    else:
        # additive worsening-oriented: post more positive than pre → positive cp_delta
        cp_delta = float(slope_post_raw - slope_pre_raw)
    cp_rss_improvement = cp["rss_improvement_vs_linear"]

    return FatigueIndices(
        pd_=pd_, auc_def=auc_def, beta1=beta1, beta1_raw=beta1_raw,
        exp_a=a, exp_rate=exp_rate, exp_tau=tau, exp_converged=conv,
        cp_t_star=float(cp_t_star) if np.isfinite(cp_t_star) else np.nan,
        cp_delta=cp_delta,
        cp_rss_improvement=(float(cp_rss_improvement)
                            if np.isfinite(cp_rss_improvement) else np.nan),
        cp_slope_pre_raw=(float(slope_pre_raw)
                          if np.isfinite(slope_pre_raw) else np.nan),
        cp_slope_post_raw=(float(slope_post_raw)
                           if np.isfinite(slope_post_raw) else np.nan),
        n_valid=T_total, baseline=float(baseline),
    )


# ---------------------------------------------------------------------------
# High-level wrapper: from raw kinematic DataFrame → fatigue indices
# ---------------------------------------------------------------------------
def indices_for_kinematic(
    raw_series: np.ndarray,
    kinematic: str,
    k: int = 10,
    ell: int = 10,
) -> dict:
    """Normalize a raw kinematic series then compute all fatigue indices.

    `raw_series` is a 1-D ordered-by-trial numpy array. Returns a plain dict
    suitable for adding as columns in a long-form DataFrame.
    """
    kind = kinematic_kind(kinematic)
    baseline, x_tilde = normalize_series(raw_series, kind, k=k)
    fi = compute_fatigue_indices(x_tilde, kind, k=k, ell=ell, baseline=baseline)
    out = asdict(fi)
    out["kinematic"] = kinematic
    out["kind"] = kind
    # also report a compact N_post-baseline count for debugging
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Quick sanity: known fatigue curve, exponential, check that indices
    have the right sign and magnitude."""
    rng = np.random.default_rng(0)

    # --- gain with 15% decay over 60 trials (true exp with a=0.15, τ=20) ---
    T = 60
    t = np.arange(T)
    true_gain = 1.0 - 0.15 * (1.0 - np.exp(-t / 20.0))
    obs_gain = true_gain * (1.0 + rng.normal(0, 0.02, size=T))
    out = indices_for_kinematic(obs_gain, "gain")
    print("gain (simulated 15% decay, τ=20):")
    print(f"  PD={out['pd_']:.3f}  β1={out['beta1']:.4f}  "
          f"a={out['exp_a']:.3f}  τ={out['exp_tau']:.1f}  "
          f"converged={out['exp_converged']}  "
          f"cp_t*={out['cp_t_star']}  cp_delta={out['cp_delta']:+.4f}")
    assert out["pd_"] > 0.1, f"PD too small: {out['pd_']:.3f}"
    assert out["beta1"] > 0, f"β1 sign wrong: {out['beta1']:.4f}"

    # --- velocity flat (no fatigue) --------------------------------------
    obs_vp = np.full(T, 500.0) * (1.0 + rng.normal(0, 0.02, size=T))
    out_v = indices_for_kinematic(obs_vp, "peak_velocity")
    print("\npeak_velocity (no fatigue):")
    print(f"  PD={out_v['pd_']:.4f}  β1={out_v['beta1']:.4f}  "
          f"a={out_v['exp_a']:.3f}  converged={out_v['exp_converged']}")
    assert abs(out_v["pd_"]) < 0.05, f"PD should be ~0: {out_v['pd_']:.3f}"

    # --- latency additive decay ------------------------------------------
    true_lat = 0.10 + 0.03 * (1.0 - np.exp(-t / 15.0))
    obs_lat = true_lat + rng.normal(0, 0.003, size=T)
    out_l = indices_for_kinematic(obs_lat, "latency")
    print("\nlatency (additive +30ms):")
    print(f"  PD={out_l['pd_']:.4f}  β1={out_l['beta1']:.5f}  "
          f"a={out_l['exp_a']:.3f}  converged={out_l['exp_converged']}")
    assert out_l["pd_"] > 0.01, f"PD should be positive: {out_l['pd_']:.4f}"

    # --- DI = PD_gain − PD_velocity on these toy series ------------------
    di_pd = out["pd_"] - out_v["pd_"]
    print(f"\nDI_PD (toy MG-like): gain_PD={out['pd_']:.3f} − "
          f"vp_PD={out_v['pd_']:.3f} = {di_pd:+.3f}")
    assert di_pd > 0.05

    print("\nPASS")


if __name__ == "__main__":
    _self_test()
