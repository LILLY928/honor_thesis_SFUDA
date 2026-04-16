"""
Aggregate per-seed results and run a Wilcoxon signed-rank test comparing the
JMDS-weighted method to the unweighted baseline on macro accuracy.
Run after all per-seed training jobs complete (see run_seed.py).
"""
import os, json, glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import wilcoxon

SEEDS   = [42, 123, 456, 789, 1024]
METHODS = ["unweighted", "jmds"]
LABELS  = ["O", "PERSON", "GOD", "NORP", "LOC"]

results = {m: [] for m in METHODS}

# Load all result files
missing = []
for method in METHODS:
    for seed in SEEDS:
        path = f"./results/results_{method}_seed{seed}.json"
        try:
            with open(path) as f:
                results[method].append(json.load(f))
        except FileNotFoundError:
            missing.append(path)

if missing:
    print("Missing result files:")
    for p in missing:
        print(f"  {p}")
    raise SystemExit(1)

# Build stats table
def stats(method, key):
    vals = [r[key] for r in results[method]]
    return np.mean(vals), np.std(vals)

print("\n" + "="*90)
print(f"{'Method':<16} | {'O':^14} | {'PERSON':^14} | {'GOD':^14} | {'NORP':^14} | {'LOC':^14} | {'Macro':^14}")
print("="*90)

for method in METHODS:
    row = f"{method:<16} |"
    for key in LABELS + ["macro"]:
        mu, sd = stats(method, key)
        row += f" {mu:.4f}+-{sd:.4f} |"
    print(row)

print("="*90)

# Wilcoxon signed-rank test on macro scores
macro_unweighted = [r["macro"] for r in results["unweighted"]]
macro_jmds       = [r["macro"] for r in results["jmds"]]

print(f"\nUnweighted macro scores : {[round(x,4) for x in macro_unweighted]}")
print(f"JMDS macro scores       : {[round(x,4) for x in macro_jmds]}")
print(f"Mean difference (JMDS - Unweighted): {np.mean(macro_jmds) - np.mean(macro_unweighted):+.4f}")

try:
    stat, p = wilcoxon(macro_jmds, macro_unweighted, alternative="greater")
    print(f"\nWilcoxon signed-rank test (one-sided, JMDS > Unweighted):")
    print(f"  statistic = {stat:.4f}")
    print(f"  p-value   = {p:.4f}")
    if p < 0.05:
        print("  -> Statistically significant improvement (p < 0.05)")
    elif p < 0.10:
        print("  -> Marginal improvement (p < 0.10)")
    else:
        print("  -> Not statistically significant (p >= 0.10)")
except Exception as e:
    print(f"Wilcoxon test could not run: {e}")

# Per-label breakdown
print("\n--- Per-label detail ---")
for key in LABELS + ["macro", "token_acc"]:
    mu_u, sd_u = stats("unweighted", key)
    mu_j, sd_j = stats("jmds", key)
    delta = mu_j - mu_u
    print(f"{key:12s}  unweighted={mu_u:.4f}+-{sd_u:.4f}  jmds={mu_j:.4f}+-{sd_j:.4f}  delta={delta:+.4f}")

# Save summary to JSON
summary = {}
for method in METHODS:
    summary[method] = {}
    for key in LABELS + ["macro", "token_acc"]:
        mu, sd = stats(method, key)
        summary[method][key] = {"mean": round(mu, 6), "std": round(sd, 6)}

os.makedirs("./results", exist_ok=True)
with open("./results/aggregated_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: results/aggregated_results.json")
