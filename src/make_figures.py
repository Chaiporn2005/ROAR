"""Render the two summary figures from the CSVs written by roar_experiment.py."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

INK = "#16261f"
MUTED = "#6b7a70"
INFORMED = "#a13a2f"
RANDOM = "#0f6e64"
GRID = "#d9e0db"


def fig_roar_curve():
    df = pd.read_csv(RESULTS / "roar_curve.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(df["baseline_pr_auc_mean"].iloc[0], color=MUTED, ls=":", lw=1.3,
               label="baseline (all 21 metabolites)")

    ax.fill_between(df["k"], df["random_pr_auc_p05"], df["random_pr_auc_p95"],
                     color=RANDOM, alpha=0.15, label="random-k removal, 5th-95th pct (30 draws)")
    ax.plot(df["k"], df["random_pr_auc_mean"], color=RANDOM, marker="o", lw=1.8,
             label="random-k removal, mean")

    ax.errorbar(df["k"], df["informed_pr_auc_mean"], yerr=df["informed_pr_auc_std"],
                color=INFORMED, marker="D", lw=2, capsize=4,
                label="informed removal (top-k SHAP)")

    for _, row in df.iterrows():
        ax.annotate(f"{row['informed_percentile_in_random_dist']:.0f}%ile",
                     (row["k"], row["informed_pr_auc_mean"]),
                     textcoords="offset points", xytext=(8, -14), fontsize=8.5, color=INFORMED)

    ax.set_xlabel("k metabolites removed (top-k by SHAP)")
    ax.set_ylabel("PR-AUC (average precision), repeated 5-fold CV")
    ax.set_title("ROAR: informed vs. random feature removal\nMTBLS242 preop-vs-12-month, XGBoost + SHAP")
    ax.set_xticks(df["k"])
    ax.grid(color=GRID, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "roar_curve.png", dpi=200)
    plt.close(fig)


def fig_baseline_shap():
    df = pd.read_csv(RESULTS / "baseline_shap_ranking.csv").sort_values("mean_abs_shap")

    def color(row):
        if row["is_known_artifact"]:
            return INFORMED
        if row["constraint_direction"] != 0:
            return RANDOM
        return MUTED

    colors = df.apply(color, axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.barh(df["metabolite"], df["mean_abs_shap"], color=colors)
    ax.set_xlabel("mean |SHAP value| (full-data fit)")
    ax.set_title("Baseline SHAP ranking\nred = known exogenous artifact, teal = literature-constrained")
    ax.grid(axis="x", color=GRID, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_shap_ranking.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_roar_curve()
    fig_baseline_shap()
    print("Wrote figures/roar_curve.png and figures/baseline_shap_ranking.png")
