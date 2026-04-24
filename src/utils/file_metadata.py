"""File- and folder-name parsing utilities for Exp 22.

This module is a **new** robust parser alongside `data_loading.py` (C5: do not
refactor the old one mid-experiment). It resolves the three pieces of metadata
we need per CSV:

    (patient_id, axis, frequency)

where:
    patient_id = (group, folder_name_stripped_of_YYYY-MM-DD_prefix)
    axis       = 'Horizontal' | 'Vertical' | None   (None = filename does not
                 specify; caller decides to drop)
    frequency  = float, 0.5 / 0.75 / 1.0 (Hz)

The regex covers all 12 filename variants present across the 2,056 CSVs. It is
intentionally permissive about whitespace between the patient name and the
`MG_` marker (some filenames have no space, e.g. `김성리MG_Horizontal ...`).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Group -> subpath map (single source of truth for "where the data lives")
# ---------------------------------------------------------------------------
GROUP_PATHS: dict[str, str] = {
    "HC": "Healthy control",
    "MG_Def": "Definite MG",
    "MG_Prob": "Probable MG",
    "CNP_3rd": "Non-MG diplopia (CNP, etc)/3rd",
    "CNP_4th": "Non-MG diplopia (CNP, etc)/4th",
    "CNP_6th": "Non-MG diplopia (CNP, etc)/6th",
    # CNP_TAO is explicitly out of scope (§4 of design).
}

# MG groupings for the analysis (Definite + Probable pooled by default; §4).
MG_GROUPS = {"MG_Def", "MG_Prob"}
CNP_GROUPS = {"CNP_3rd", "CNP_4th", "CNP_6th"}
HC_GROUPS = {"HC"}


# ---------------------------------------------------------------------------
# Filename regex
# ---------------------------------------------------------------------------
# Examples the regex must handle (all 6 variants per patient-folder):
#   송나리 MG_Horizontal Saccade  B (0.5Hz).csv
#   송나리 MG_Vertical Saccade  B (1Hz)_000.csv
#   김성리MG_Horizontal Saccade  B (0.75Hz).csv            # no space
#   Huang Li MG_Horizontal Saccade  B (0.5Hz).csv          # two-word name
#   <name> MG VOG - Horizontal Saccade B (0.5Hz).csv       # alt marker style
# Note the (very common) double-space before B.

_FILENAME_RE = re.compile(
    r"""
    ^(?P<name>.+?)                                # patient name (non-greedy)
    \s*                                           # optional whitespace
    (?:(?:MG\s+VOG\s+-_?)|(?:VOG\s+-_?)|(?:MG_))? # optional marker
    \s*
    (?P<axis>Horizontal|Vertical)?                # axis (optional in filename)
    \s*Saccade\s+B\s*                             # literal 'Saccade B'
    \(\s*(?P<freq>\d+(?:\.\d+)?)\s*Hz\)           # (X[.Y]Hz)
    (?:_\d+)?                                     # optional _000 suffix
    (?:\.hea)?                                    # optional .hea tag
    \.csv$                                        # literal .csv
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Folder names are `YYYY-MM-DD <name>`. We keep the regex tolerant to trailing
# whitespace and allow leading `YYYYMMDD` (no dashes) just in case.
_FOLDER_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedFilename:
    name: str        # patient name, trimmed
    axis: Optional[str]   # 'Horizontal' | 'Vertical' | None
    frequency: float      # 0.5 / 0.75 / 1.0


def parse_filename(fname: str) -> Optional[ParsedFilename]:
    """Parse a CSV filename into (name, axis, frequency).

    Returns None if the filename does not match the recognised patterns.
    """
    m = _FILENAME_RE.match(fname)
    if m is None:
        return None
    name = m.group("name").strip()
    axis = m.group("axis")
    if axis is not None:
        axis = axis.capitalize()  # normalise case
    freq = float(m.group("freq"))
    return ParsedFilename(name=name, axis=axis, frequency=freq)


def strip_folder_date(folder_name: str) -> str:
    """Strip a leading `YYYY-MM-DD ` from a patient-folder name, if present."""
    return _FOLDER_DATE_RE.sub("", folder_name).strip()


def patient_id(group: str, folder_name: str) -> tuple[str, str]:
    """The canonical (group, name) patient_id used for all aggregation and CV.

    See design §2.1. Cross-group name collisions are treated as different
    patients (§2.2); that is why `group` is part of the tuple.
    """
    return (group, strip_folder_date(folder_name))


# ---------------------------------------------------------------------------
# Self-test (CLI / smoke)
# ---------------------------------------------------------------------------
def _self_test(base_dir: str) -> None:
    """Walk the data tree and print parse stats. Intended for ad-hoc use."""
    import collections

    total = 0
    parsed = 0
    axis_missing = collections.Counter()
    by_group_freq_axis = collections.Counter()
    unique_names_by_group: dict[str, set[str]] = collections.defaultdict(set)
    visit_folders_by_group: dict[str, set[str]] = collections.defaultdict(set)
    examples_missing: dict[str, list[str]] = collections.defaultdict(list)

    for group, sub in GROUP_PATHS.items():
        group_dir = os.path.join(base_dir, sub)
        if not os.path.isdir(group_dir):
            print(f"[warn] missing group dir: {group_dir}")
            continue
        for folder in sorted(os.listdir(group_dir)):
            fdir = os.path.join(group_dir, folder)
            if not os.path.isdir(fdir):
                continue
            visit_folders_by_group[group].add(folder)
            pid = patient_id(group, folder)
            unique_names_by_group[group].add(pid[1])
            for f in os.listdir(fdir):
                if not f.endswith(".csv"):
                    continue
                total += 1
                p = parse_filename(f)
                if p is None:
                    if len(examples_missing[group]) < 3:
                        examples_missing[group].append(f)
                    continue
                parsed += 1
                if p.axis is None:
                    axis_missing[group] += 1
                by_group_freq_axis[(group, p.frequency, p.axis or "NoAxis")] += 1

    print(f"Total CSVs: {total}, parsed: {parsed}")
    print(f"Axis missing in filename: {sum(axis_missing.values())} (by group: {dict(axis_missing)})")
    for group in GROUP_PATHS:
        folders = len(visit_folders_by_group[group])
        names = len(unique_names_by_group[group])
        print(f"  {group:8s} visits={folders:3d} unique_names={names:3d}")
    print("Sample counts by (group, freq, axis):")
    for k, v in sorted(by_group_freq_axis.items()):
        print(f"  {k}: {v}")
    for g, exs in examples_missing.items():
        for e in exs:
            print(f"  UNPARSED {g}: {e}")


if __name__ == "__main__":
    import sys

    default_base = "/workspace/Myasthenia-Gravis-Detection/data"
    _self_test(sys.argv[1] if len(sys.argv) > 1 else default_base)
