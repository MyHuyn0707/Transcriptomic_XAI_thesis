"""Generate self-contained charts used by Chapter 3, Section 4."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PERF_ROOT = ROOT / "outputs_LV" / "rules_extraction_results"
QUALITY = ROOT / "outputs_LV" / "rule_extraction_summary" / "rule_quality_train_test.csv"
OUT = ROOT / "CT552_TraMy_Nguyen" / "images" / "part-02" / "chap-03" / "sec-04"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["boruta", "mrmr_k50", "mrmr_k75"]
METHOD_LABELS = {"boruta": "Boruta", "mrmr_k50": "mRMR-50", "mrmr_k75": "mRMR-75"}
COLORS = {"boruta": "#1f77b4", "mrmr_k50": "#ff7f0e", "mrmr_k75": "#2ca02c"}

def style(ax, title, ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

for k in (4, 5):
    perf = pd.read_csv(PERF_ROOT / f"k{k}.csv")
    datasets = sorted(perf["dataset"].unique())
    for model in ("decisiontree", "rf"):
        d = perf[perf["model"] == model].copy()
        d["method"] = pd.Categorical(d["fs_method"], METHODS, ordered=True)
        d = d.sort_values(["method", "dataset"])
        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        for metric, ax, label in (("f1_macro", axes[0], "F1-macro"), ("accuracy", axes[1], "Accuracy")):
            for method in METHODS:
                z = d[d["fs_method"] == method].set_index("dataset").reindex(datasets)
                ax.plot(datasets, z[metric], marker="o", linewidth=1.8, label=METHOD_LABELS[method], color=COLORS[method])
            style(ax, f"{model.upper()} rule model performance (k={k}) — {label}", label)
        fig.tight_layout()
        fig.savefig(OUT / f"rule_model_performance_{model}_k{k}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

quality = pd.read_csv(QUALITY)
for k in (4, 5):
    for model in ("dt", "rf"):
        d = quality[(quality["k"] == k) & (quality["model"] == model)].copy()
        datasets = sorted(d["dataset"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
        panels = [
            ("mean_confidence_train", axes[0, 0], "Confidence — train"),
            ("mean_confidence_test", axes[0, 1], "Confidence — test"),
            ("mean_fidelity_train", axes[1, 0], "Fidelity — train"),
            ("mean_fidelity_test", axes[1, 1], "Fidelity — test"),
        ]
        for metric, ax, label in panels:
            for method in METHODS:
                z = d[d["fs_method"] == method].set_index("dataset").reindex(datasets)
                ax.plot(datasets, z[metric], marker="o", linewidth=1.6, label=METHOD_LABELS[method], color=COLORS[method])
            style(ax, label, "Value")
        fig.suptitle(f"Rule quality by feature-selection method — {model.upper()} (k={k})", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT / f"rule_quality_{model}_k{k}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

# Dedicated k=4 fidelity chart used immediately before the quality summary table.
d = quality[quality["k"] == 4]
fig, ax = plt.subplots(figsize=(15, 6))
for model, marker in (("dt", "o"), ("rf", "s")):
    z = d[d["model"] == model].groupby("dataset", as_index=False)["mean_fidelity_test"].mean().sort_values("dataset")
    ax.plot(z["dataset"], z["mean_fidelity_test"], marker=marker, linewidth=1.8, label=model.upper())
style(ax, "Mean per-rule test fidelity at k=4", "Mean fidelity")
fig.tight_layout()
fig.savefig(OUT / "rule_model_test_fidelity_k4.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"Generated charts in {OUT}")
