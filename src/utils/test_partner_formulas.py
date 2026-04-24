"""Mechanical fidelity tests for the Exp 22 v2 rework.

These tests exist because Exp 22 v1 drifted on two specific formulas:
  (1) baseline and PD-tail summaries use the MEDIAN in v1 code, but the
      partner's primary feedback (`llmdocs/partner_feedback_feb2026.md`
      Sections B and D) writes the formulas with MEAN.
  (2) Partner Section C lists three curve families: linear, exponential,
      and change-point / piecewise. v1 implements only the first two.

This file enforces those two contracts (C19, C20 in CONTEXT.md v2) as unit
tests. The tests are designed so they FAIL on v1 code and only PASS when
the next agent has switched to mean and added `fit_changepoint`. Do not
weaken the tests; edit the implementation instead.

Run from project root:

    ./env/bin/python src/utils/test_partner_formulas.py

Exit code 0 = all pass. Any other exit code = at least one contract failed.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# allow running from project root without installing the package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# C19 — baseline + PD tail must use MEAN, not median
# ---------------------------------------------------------------------------
def test_baseline_uses_mean() -> None:
    """Baseline normalization must be mean of first k, per partner Section B:
        G~_e(t) = G_e(t) / mean(G_e(1..k))
    A deliberately skewed input distinguishes mean from median.
    """
    from utils.fatigue_models import normalize_series

    # first 5 values are [1, 2, 3, 4, 100] — mean = 22.0, median = 3.0
    y = np.array(
        [1.0, 2.0, 3.0, 4.0, 100.0,
         6.0, 7.0, 8.0, 9.0, 10.0,
         1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )
    baseline, _ = normalize_series(y, kind="ratio", k=5)
    expected = float(np.mean(y[:5]))  # 22.0
    unexpected = float(np.median(y[:5]))  # 3.0

    assert abs(baseline - expected) < 1e-9, (
        f"C19 violation: baseline = {baseline} (expected mean {expected}). "
        f"Median would be {unexpected}. Partner Section B formula uses mean."
    )
    print(f"  [PASS] test_baseline_uses_mean (baseline={baseline}, mean={expected})")


def test_pd_uses_mean_tail() -> None:
    """PD must be 1 − mean(tail) for ratio kinematics per partner Section D:
        PD_G = 1 − mean(G~(T − ℓ + 1..T))
    The input below has a skewed tail that differentiates mean from median.
    """
    from utils.fatigue_models import indices_for_kinematic

    # Constant baseline of 1.0 so normalized series equals the raw series.
    head = np.ones(10, dtype=float)
    middle = 0.9 * np.ones(10, dtype=float)
    tail = np.array([0.3, 0.4, 0.4, 0.5, 100.0], dtype=float)  # ℓ=5, skewed
    y = np.concatenate([head, middle, tail])

    out = indices_for_kinematic(y, "gain", k=10, ell=5)
    expected_pd = 1.0 - float(np.mean(tail))          # mean-based partner formula
    unexpected_pd = 1.0 - float(np.median(tail))      # median (v1 behavior)

    assert abs(out["pd_"] - expected_pd) < 1e-6, (
        f"C19 violation: PD = {out['pd_']} (expected mean-based {expected_pd}). "
        f"Median-based would be {unexpected_pd}. Partner Section D uses mean."
    )
    print(f"  [PASS] test_pd_uses_mean_tail (PD={out['pd_']}, expected={expected_pd})")


# ---------------------------------------------------------------------------
# C20 — change-point / piecewise model is the third curve family, not optional
# ---------------------------------------------------------------------------
def test_fit_changepoint_exists_and_detects_known_step() -> None:
    """Partner Section C lists three model families and calls out change-point
    explicitly ("detects delayed fatigue onset"). A `fit_changepoint` function
    must be present in fatigue_models and recover a known change point within
    ±3 samples on a synthetic delayed-onset gain trace.
    """
    try:
        from utils.fatigue_models import fit_changepoint
    except ImportError as e:
        raise AssertionError(
            "C20 violation: `fit_changepoint` is missing from "
            "src/utils/fatigue_models.py. Partner Section C requires "
            "a piecewise / change-point model as the third curve family."
        ) from e

    rng = np.random.default_rng(42)
    T = 60
    cp_true = 30
    # gain stays near 1.0, then drops to 0.7 at t=30 — classic delayed-onset
    signal = np.concatenate([np.ones(cp_true), 0.7 * np.ones(T - cp_true)])
    noise = rng.normal(0, 0.01, T)
    y = signal + noise

    result = fit_changepoint(y)
    # contract: result is a dict with at least these keys
    for key in ("t_star", "slope_pre", "slope_post"):
        assert key in result, (
            f"C20 violation: fit_changepoint must return a dict containing "
            f"key {key!r}. Got keys: {sorted(result)}."
        )
    cp_est = int(result["t_star"])
    assert abs(cp_est - cp_true) <= 3, (
        f"C20 violation: fit_changepoint recovered t_star = {cp_est} for "
        f"synthetic step at t = {cp_true}; tolerance is ±3. Likely a search "
        f"range or objective-function bug."
    )
    # sanity: the post-break slope should be shallower (mean ~ 0.7) than pre-break (mean ~ 1.0);
    # slope pre ≈ 0, slope post ≈ 0 since both halves are flat. Require |slopes| small
    assert abs(result["slope_pre"]) < 0.02, (
        f"pre-break slope should be near zero for this synthetic input; "
        f"got {result['slope_pre']}"
    )
    assert abs(result["slope_post"]) < 0.02, (
        f"post-break slope should be near zero for this synthetic input; "
        f"got {result['slope_post']}"
    )
    print(f"  [PASS] test_fit_changepoint (t_star={cp_est}, slopes="
          f"{result['slope_pre']:.4f}/{result['slope_post']:.4f})")


def test_changepoint_surfaces_in_indices() -> None:
    """The change-point fit must flow into the fatigue-index dict returned by
    `indices_for_kinematic`, under keys at least `cp_t_star` and `cp_delta`
    (delta defined so positive = fatigue intensifies after t*). Without this,
    change-point results cannot be aggregated patient-level and the partner's
    Step C prescription is silently dropped after fitting.
    """
    from utils.fatigue_models import indices_for_kinematic

    rng = np.random.default_rng(7)
    T = 60
    # gain clean at 1.0, then decays linearly after t=30
    y = np.ones(T)
    y[30:] = np.linspace(1.0, 0.5, T - 30)
    y = y + rng.normal(0, 0.005, T)

    out = indices_for_kinematic(y, "gain", k=10, ell=5)
    for key in ("cp_t_star", "cp_delta"):
        assert key in out, (
            f"C20 violation: `indices_for_kinematic` output does not contain "
            f"{key!r}. Partner-prescribed change-point results must be exposed "
            f"at the patient/visit aggregation layer, not only inside the fit."
        )
    assert np.isfinite(out["cp_t_star"]), (
        f"cp_t_star is NaN on a series with a clear change point. "
        f"If the fit failed on a clean synthetic, the search range is too narrow."
    )
    print(f"  [PASS] test_changepoint_surfaces_in_indices (cp_t_star="
          f"{out['cp_t_star']}, cp_delta={out['cp_delta']:+.4f})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_all() -> int:
    """Run all C19/C20 contract tests. Return 0 on pass, non-zero on failure."""
    tests = [
        ("C19: baseline uses mean",       test_baseline_uses_mean),
        ("C19: PD tail uses mean",        test_pd_uses_mean_tail),
        ("C20: fit_changepoint exists",   test_fit_changepoint_exists_and_detects_known_step),
        ("C20: cp indices surface",       test_changepoint_surfaces_in_indices),
    ]
    failed: list[tuple[str, str]] = []
    for name, fn in tests:
        print(f"  {name} ...")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append((name, str(e)))
        except Exception as e:  # import errors etc
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))

    print()
    if not failed:
        print("All v2 fidelity contracts pass.")
        return 0
    print(f"{len(failed)} contract(s) failed:")
    for name, reason in failed:
        print(f"  - {name}: {reason}")
    print()
    print("These tests enforce partner-framework fidelity. Do not weaken them.")
    print("Edit `src/utils/fatigue_models.py` to satisfy them.")
    return 1


if __name__ == "__main__":
    sys.exit(run_all())
