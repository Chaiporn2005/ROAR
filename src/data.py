"""Load the MTBLS242 preop-vs-12-month obesity classification table.

Reuses the exact same public files and label definition documented in
Chaiporn2005/Forecast (Bridge_ai_summit_nmr_2026, Experiment A):
  obese  = preop            (label 1, n=106)
  healthy = 12 months after surgery (label 0, n=71)
21 metabolites, log1p-transformed.
"""
import csv
import numpy as np

MAF_PATH = "data/m_MTBLS242_v2_maf.tsv"
SAMPLE_SHEET_PATH = "data/s_MTBLS242.txt"

# Monotonic constraint direction relative to label=1 (obese/preop), taken
# verbatim from the constraint set documented in Bridge_ai_summit_nmr_2026's
# README for Experiment A (11 of 21 metabolites constrained from
# obesity/insulin-resistance metabolomics literature; Newgard 2009,
# Wang 2011 Nat Med, Wurtz et al.).
CONSTRAINT_DIRECTION = {
    "L-valine": 1,
    "L-leucine": 1,
    "L-allo-Isoleucine": 1,
    "L-tyrosine": 1,
    "D-phenylalanine": 1,
    "L-alanine": 1,
    "lipoproteins": 1,
    "L-Lactic acid": 1,
    "glycine": -1,
    "L-glutamine": -1,
    "histidine": -1,
}

# Metabolites flagged in the same README as exogenous / non-biological
# artifacts (isopropanol = surgical skin-prep contaminant), deliberately
# left unconstrained in the original work.
KNOWN_ARTIFACTS = ["isopropanol", "methanol", "Dimethyl sulfone"]


def _norm(name):
    return name.replace("-", "_")


def load_dataset():
    sample_tp = {}
    with open(SAMPLE_SHEET_PATH, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            sample_tp[_norm(row["Sample Name"])] = row["Factor Value[time point]"]

    with open(MAF_PATH, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        sample_cols = header[18:]
        metabolite_names = []
        rows = []
        for row in r:
            metabolite_names.append(row[4])
            rows.append([float(x) if x not in ("", "NA") else np.nan for x in row[18:]])

    values = np.array(rows, dtype=float)  # metabolites x samples

    keep_idx = []
    labels = []
    for i, col in enumerate(sample_cols):
        tp = sample_tp.get(_norm(col))
        if tp == "preop":
            keep_idx.append(i)
            labels.append(1)
        elif tp == "12 months after surgery":
            keep_idx.append(i)
            labels.append(0)

    X = values[:, keep_idx].T  # samples x metabolites
    y = np.array(labels, dtype=int)

    if np.isnan(X).any():
        raise ValueError("Unexpected missing values in the 21-metabolite panel for the "
                          "preop/12-month subset -- original project reports a complete panel.")

    X_log = np.log1p(X)

    assert X_log.shape == (177, 21), X_log.shape
    assert int(y.sum()) == 106 and int((1 - y).sum()) == 71, (y.sum(), (1 - y).sum())

    monotone = tuple(CONSTRAINT_DIRECTION.get(name, 0) for name in metabolite_names)

    return {
        "X": X_log,
        "y": y,
        "feature_names": metabolite_names,
        "monotone_constraints": monotone,
    }


if __name__ == "__main__":
    d = load_dataset()
    print("X", d["X"].shape, "y", d["y"].shape, "positives", d["y"].sum())
    print("features:", d["feature_names"])
    print("monotone:", d["monotone_constraints"])
