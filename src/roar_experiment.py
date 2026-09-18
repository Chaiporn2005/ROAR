"""ROAR (RemOve And Retrain) faithfulness experiment for the MTBLS242
obesity (preop vs 12-month) SHAP ranking documented in
Chaiporn2005/Forecast (Bridge_ai_summit_nmr_2026, Experiment A).

For each checkpoint k, the top-k SHAP-ranked metabolites are removed and the
model is retrained from scratch (never masked at inference time). PR-AUC
after informed removal is compared against a distribution of PR-AUC scores
from removing k *random* metabolites, so a genuine effect can be told apart
from ordinary variance. A retrained-model SHAP re-ranking after each removal
checks whether a correlated "proxy" metabolite takes over the top rank.

Run from the repository root:  python src/roar_experiment.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent))
from data import load_dataset, CONSTRAINT_DIRECTION, KNOWN_ARTIFACTS  # noqa: E402
from model import cv_pr_auc, fit_full, REPEAT_SEEDS  # noqa: E402

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

K_CHECKPOINTS = [1, 3, 5, 8, 11]
N_RANDOM_DRAWS = 30
RANDOM_CV_REPEATS = REPEAT_SEEDS[:3]  # lighter CV budget for the 30 control draws
RNG_BASE_SEED = 20260918

def shap_ranking(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def constraints_for_subset(feature_names):
    return tuple(CONSTRAINT_DIRECTION.get(name, 0) for name in feature_names)


def drop_features(X, feature_names, drop_set):
    keep_idx = [i for i, n in enumerate(feature_names) if n not in drop_set]
    kept_names = [feature_names[i] for i in keep_idx]
    return X[:, keep_idx], kept_names

def proxy_check(X_full, feature_names_full, removed_set, kept_X, kept_names):
    """Retrain on the reduced feature set, SHAP-rank it, and correlate the new
    top feature against every removed feature using the ORIGINAL (full,
    untransformed-by-removal) log1p values."""
    reduced_constraints = constraints_for_subset(kept_names)
    reduced_model = fit_full(kept_X, y_global, reduced_constraints)
    reduced_ranking = shap_ranking(reduced_model, kept_X, kept_names)
    new_top_feature = reduced_ranking[0][0]
    new_top_values = kept_X[:, kept_names.index(new_top_feature)]

    best_corr, best_removed = 0.0, None
    for rf in removed_set:
        removed_values = X_full[:, feature_names_full.index(rf)]
        r = float(np.corrcoef(new_top_values, removed_values)[0, 1])
        if abs(r) > abs(best_corr):
            best_corr, best_removed = r, rf
    return new_top_feature, best_removed, best_corr, reduced_ranking

def main():
    t0 = time.time()
    data = load_dataset()
    X, y, feature_names = data["X"], data["y"], data["feature_names"]
    global y_global
    y_global = y
    monotone = data["monotone_constraints"]

    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} metabolites "
          f"({int(y.sum())} preop / {int((1 - y).sum())} 12-month).")

    # ---------------------------------------------------------------
    # 1. Baseline: full-feature CV PR-AUC + full-data SHAP ranking
    # ---------------------------------------------------------------
    print("Baseline repeated CV ...")
    baseline_scores = cv_pr_auc(X, y, monotone)
    baseline_mean, baseline_std = baseline_scores.mean(), baseline_scores.std()
    print(f"  baseline PR-AUC = {baseline_mean:.4f} +/- {baseline_std:.4f} "
          f"({len(baseline_scores)} folds)")

    raw_fold_rows = [{"condition": "baseline", "k": 0, "fold_index": i, "pr_auc": s}
                      for i, s in enumerate(baseline_scores)]

    baseline_model = fit_full(X, y, monotone)
    baseline_ranking = shap_ranking(baseline_model, X, feature_names)

    ranking_rows = []
    for rank, (name, val) in enumerate(baseline_ranking, start=1):
        ranking_rows.append({
            "rank": rank,
            "metabolite": name,
            "mean_abs_shap": val,
            "constraint_direction": CONSTRAINT_DIRECTION.get(name, 0),
            "is_known_artifact": name in KNOWN_ARTIFACTS,
        })
    pd.DataFrame(ranking_rows).to_csv(RESULTS_DIR / "baseline_shap_ranking.csv", index=False)

    top3 = [r[0] for r in baseline_ranking[:3]]
    top3_is_artifact_trio = set(top3) == set(KNOWN_ARTIFACTS)
    print(f"  top-3 SHAP features: {top3}")
    print(f"  matches documented artifact trio {KNOWN_ARTIFACTS}: {top3_is_artifact_trio}")

    # ---------------------------------------------------------------
    # 2. ROAR checkpoints: informed removal vs random-removal control
    # ---------------------------------------------------------------
    curve_rows = []
    proxy_rows = []
    random_draw_rows = []
    random_dists = {}  # k -> np.array of 30 random-draw mean APs (kept for the contrast step)

    for k in K_CHECKPOINTS:
        informed_set = [name for name, _ in baseline_ranking[:k]]
        Xk, kept_names = drop_features(X, feature_names, set(informed_set))
        ck = constraints_for_subset(kept_names)

        informed_scores = cv_pr_auc(Xk, y, ck)
        informed_mean = informed_scores.mean()
        raw_fold_rows.extend({"condition": "informed", "k": k, "fold_index": i, "pr_auc": s}
                              for i, s in enumerate(informed_scores))

        rng = np.random.default_rng(RNG_BASE_SEED + k)
        random_means = []
        for draw in range(N_RANDOM_DRAWS):
            random_set = rng.choice(feature_names, size=k, replace=False).tolist()
            Xr, kept_r = drop_features(X, feature_names, set(random_set))
            cr = constraints_for_subset(kept_r)
            scores = cv_pr_auc(Xr, y, cr, repeat_seeds=RANDOM_CV_REPEATS)
            random_means.append(scores.mean())
            random_draw_rows.append({
                "k": k, "draw_index": draw,
                "removed_features": ";".join(sorted(random_set)),
                "mean_pr_auc": scores.mean(),
            })
        random_means = np.array(random_means)
        random_dists[k] = random_means

        percentile = float((random_means <= informed_mean).mean() * 100)

        curve_rows.append({
            "k": k,
            "removed_features": ";".join(informed_set),
            "informed_pr_auc_mean": informed_mean,
            "informed_pr_auc_std": informed_scores.std(),
            "random_pr_auc_mean": random_means.mean(),
            "random_pr_auc_std": random_means.std(),
            "random_pr_auc_p05": float(np.percentile(random_means, 5)),
            "random_pr_auc_p95": float(np.percentile(random_means, 95)),
            "informed_percentile_in_random_dist": percentile,
            "baseline_pr_auc_mean": baseline_mean,
        })
        print(f"  k={k:2d}  informed AP={informed_mean:.4f}  "
              f"random AP={random_means.mean():.4f}+/-{random_means.std():.4f}  "
              f"percentile={percentile:.1f}")

        new_top, best_removed, corr, _ = proxy_check(X, feature_names, set(informed_set), Xk, kept_names)
        proxy_rows.append({
            "k": k,
            "removed_features": ";".join(informed_set),
            "new_top_feature_after_retrain": new_top,
            "most_correlated_removed_feature": best_removed,
            "pearson_r": corr,
        })

    pd.DataFrame(curve_rows).to_csv(RESULTS_DIR / "roar_curve.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(RESULTS_DIR / "proxy_detection.csv", index=False)
    pd.DataFrame(random_draw_rows).to_csv(RESULTS_DIR / "random_draws_raw.csv", index=False)

    # ---------------------------------------------------------------
    # 3. Contrast: known artifact trio vs. top literature-grounded trio
    # ---------------------------------------------------------------
    constrained_ranking = [(n, v) for n, v in baseline_ranking if n in CONSTRAINT_DIRECTION]
    biology_top3 = [n for n, _ in constrained_ranking[:3]]

    def eval_group(drop_set, label):
        Xg, kept_g = drop_features(X, feature_names, set(drop_set))
        cg = constraints_for_subset(kept_g)
        scores = cv_pr_auc(Xg, y, cg)
        raw_fold_rows.extend({"condition": label, "k": 3, "fold_index": i, "pr_auc": s}
                              for i, s in enumerate(scores))
        return scores.mean(), scores.std()

    artifact_mean, artifact_std = eval_group(KNOWN_ARTIFACTS, "contrast_artifact_trio")
    biology_mean, biology_std = eval_group(biology_top3, "contrast_biology_trio")
    rand3 = random_dists[3]
    contrast_rows = [
        {
            "group": "known_artifact_trio",
            "features": ";".join(KNOWN_ARTIFACTS),
            "pr_auc_mean": artifact_mean,
            "pr_auc_std": artifact_std,
            "percentile_in_random_k3_dist": float((rand3 <= artifact_mean).mean() * 100),
        },
        {
            "group": "top_literature_biology_trio",
            "features": ";".join(biology_top3),
            "pr_auc_mean": biology_mean,
            "pr_auc_std": biology_std,
            "percentile_in_random_k3_dist": float((rand3 <= biology_mean).mean() * 100),
        },
    ]
    pd.DataFrame(contrast_rows).to_csv(RESULTS_DIR / "contrast_artifact_vs_biology.csv", index=False)
    print("Contrast (k=3):")
    for row in contrast_rows:
        print(f"  {row['group']:26s} AP={row['pr_auc_mean']:.4f}  "
              f"percentile-in-random={row['percentile_in_random_k3_dist']:.1f}")

    # ---------------------------------------------------------------
    # 4. Protocol / provenance record
    # ---------------------------------------------------------------
    import sklearn
    import xgboost

    def sha256(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    protocol = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_preop": int(y.sum()),
        "n_12month": int((1 - y).sum()),
        "k_checkpoints": K_CHECKPOINTS,
        "n_random_draws_per_k": N_RANDOM_DRAWS,
        "cv_repeat_seeds_informed_and_baseline": REPEAT_SEEDS,
        "cv_repeat_seeds_random_control": RANDOM_CV_REPEATS,
        "n_splits": 5,
        "rng_base_seed": RNG_BASE_SEED,
        "baseline_pr_auc_mean": float(baseline_mean),
        "baseline_pr_auc_std": float(baseline_std),
        "baseline_top3_shap": top3,
        "baseline_top3_matches_known_artifact_trio": top3_is_artifact_trio,
        "known_artifacts": KNOWN_ARTIFACTS,
        "constraint_direction": CONSTRAINT_DIRECTION,
        "package_versions": {
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "input_file_sha256": {
            "m_MTBLS242_v2_maf.tsv": sha256("data/m_MTBLS242_v2_maf.tsv"),
            "s_MTBLS242.txt": sha256("data/s_MTBLS242.txt"),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS_DIR / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    pd.DataFrame(raw_fold_rows).to_csv(RESULTS_DIR / "raw_fold_scores.csv", index=False)

    print(f"Done in {protocol['runtime_seconds']}s. Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
