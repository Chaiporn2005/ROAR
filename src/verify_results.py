"""Independent sanity checks on the results written by roar_experiment.py.

Does not re-fit any model. It re-derives what can be re-derived arithmetically
from the saved CSVs/JSON and flags anything inconsistent, in the same spirit
as verify_results.py in the parent Forecast repository.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
DATA = Path("data")
FAILS = []


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILS.append(label)


def main():
    protocol = json.load(open(RESULTS / "protocol.json"))

    # 1. Input file hashes match what the run recorded (data has not silently changed)
    for fname, recorded in protocol["input_file_sha256"].items():
        h = hashlib.sha256((DATA / fname).read_bytes()).hexdigest()
        check(f"input hash unchanged: {fname}", h == recorded)

    # 2. Sample counts match the documented MTBLS242 preop/12-month split
    check("n_preop == 106", protocol["n_preop"] == 106)
    check("n_12month == 71", protocol["n_12month"] == 71)
    check("n_features == 21", protocol["n_features"] == 21)

    # 3. baseline_shap_ranking.csv is sorted by mean_abs_shap descending and rank is contiguous
    ranking = pd.read_csv(RESULTS / "baseline_shap_ranking.csv")
    check("shap ranking sorted descending",
          (ranking["mean_abs_shap"].values == np.sort(ranking["mean_abs_shap"].values)[::-1]).all())
    check("shap ranking rank column contiguous 1..21",
          list(ranking["rank"]) == list(range(1, len(ranking) + 1)))

    known_artifacts = set(protocol["known_artifacts"])
    flagged = set(ranking.loc[ranking["is_known_artifact"], "metabolite"])
    check("is_known_artifact flags match protocol's known_artifacts list", flagged == known_artifacts)

    # 4. roar_curve.csv: informed removal must equal the mean of raw_fold_scores for that k
    curve = pd.read_csv(RESULTS / "roar_curve.csv")
    raw = pd.read_csv(RESULTS / "raw_fold_scores.csv")
    for _, row in curve.iterrows():
        k = row["k"]
        recomputed = raw.loc[(raw["condition"] == "informed") & (raw["k"] == k), "pr_auc"].mean()
        check(f"k={k}: informed_pr_auc_mean matches recomputed mean of raw folds",
              abs(recomputed - row["informed_pr_auc_mean"]) < 1e-9)

    baseline_recomputed = raw.loc[raw["condition"] == "baseline", "pr_auc"].mean()
    check("baseline_pr_auc_mean matches recomputed mean of raw baseline folds",
          abs(baseline_recomputed - protocol["baseline_pr_auc_mean"]) < 1e-9)

    # 5. random_draws_raw.csv: exactly N_RANDOM_DRAWS per k, and stored percentile is reproducible
    draws = pd.read_csv(RESULTS / "random_draws_raw.csv")
    for _, row in curve.iterrows():
        k = row["k"]
        sub = draws.loc[draws["k"] == k, "mean_pr_auc"]
        check(f"k={k}: {protocol['n_random_draws_per_k']} random draws recorded",
              len(sub) == protocol["n_random_draws_per_k"])
        recomputed_pct = float((sub.values <= row["informed_pr_auc_mean"]).mean() * 100)
        check(f"k={k}: informed_percentile_in_random_dist reproduces from raw draws",
              abs(recomputed_pct - row["informed_percentile_in_random_dist"]) < 1e-6)
        recomputed_mean = sub.mean()
        check(f"k={k}: random_pr_auc_mean reproduces from raw draws",
              abs(recomputed_mean - row["random_pr_auc_mean"]) < 1e-9)

    # 6. No fold in raw_fold_scores.csv exceeds [0, 1] (PR-AUC / average precision is bounded;
    #    a small float-precision tolerance covers values like 1.0000000000000002)
    check("all recorded PR-AUC values lie in [0, 1] (float tolerance 1e-9)",
          raw["pr_auc"].between(-1e-9, 1 + 1e-9).all())

    # 7. Every "removed_features" list in roar_curve.csv is a superset relationship:
    #    the k=1 removed feature must be a subset of the k=3 removed features, etc.
    #    (top-k by SHAP is nested by construction; this catches a ranking/indexing bug.)
    removed_sets = {row["k"]: set(row["removed_features"].split(";")) for _, row in curve.iterrows()}
    ks_sorted = sorted(removed_sets)
    nested_ok = all(removed_sets[ks_sorted[i]] <= removed_sets[ks_sorted[i + 1]]
                     for i in range(len(ks_sorted) - 1))
    check("top-k removed-feature sets are nested (k=1 subset of k=3 subset of k=5 ...)", nested_ok)

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
