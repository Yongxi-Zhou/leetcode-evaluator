#!/usr/bin/env python3
"""
Generate Figure 5: Reasoning vs. Standard Models analysis.
- Scatter plot: RLPR vs Δ (gap), with reasoning models highlighted
- Linear regression + residual analysis
- Permutation test for residual difference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Data from Table 2 (D=Detailed, M=Minimal)
models = [
    # (name, RLPR_D, RLPR_M, PSR_D, PSR_M, AV_D, AV_M, is_reasoning)
    ("GPT-4.1",              59.6, 59.2, 47.5, 49.0, 4.36, 4.08, False),
    ("GPT-4.1-mini",         58.4, 59.2, 50.0, 47.0, 3.28, 4.48, False),
    ("o4-mini",              86.2, 85.6, 75.0, 69.0, 3.36, 5.20, True),
    ("Claude Sonnet 4.5",    52.8, 51.4, 43.0, 47.0, 3.36, 1.68, False),
    ("Claude Haiku 4.5",     41.2, 40.8, 35.0, 34.0, 2.40, 2.00, False),
    ("Gemini 3.1 Pro",       86.2, 85.8, 75.0, 74.0, 3.36, 3.92, False),
    ("Gemini 3 Flash",       80.8, 83.2, 67.0, 70.0, 5.28, 4.40, False),
    ("Gemini 3.1 Flash-Lite",69.2, 63.6, 53.0, 50.0, 5.28, 5.44, False),
    ("Gemini 2.5 Flash",     53.8, 48.2, 36.0, 33.0, 7.04, 5.84, False),
    ("Gemini 2.5 Flash+Think",71.6, 74.4, 58.0, 57.0, 4.64, 4.72, True),
    ("QwQ-Plus",             72.8, 72.6, 59.0, 58.0, 4.80, 4.72, True),
    ("Qwen-Plus",            63.4, 61.2, 52.0, 51.0, 4.72, 4.64, False),
    ("Qwen-Max",             30.6, 30.4, 26.0, 25.0, 2.24, 2.24, False),
    ("Qwen-Turbo",           34.4, 30.8, 32.0, 28.0, 0.88, 1.04, False),
    ("DeepSeek-R1",          84.2, 84.4, 74.0, 75.0, 3.76, 3.20, True),
    ("DeepSeek-V3.2",        48.4, 47.6, 36.0, 38.0, 5.52, 4.00, False),
]

# Compute per-model averages across both prompts
names = []
rlpr_mean = []
psr_mean = []
delta_mean = []
av_mean = []
is_reasoning = []

for m in models:
    name, rd, rm, pd, pm, ad, am, reasoning = m
    rlpr = (rd + rm) / 2
    psr = (pd + pm) / 2
    delta = rlpr - psr
    av = (ad + am) / 2
    names.append(name)
    rlpr_mean.append(rlpr)
    psr_mean.append(psr)
    delta_mean.append(delta)
    av_mean.append(av)
    is_reasoning.append(reasoning)

rlpr_mean = np.array(rlpr_mean)
delta_mean = np.array(delta_mean)
av_mean = np.array(av_mean)
is_reasoning = np.array(is_reasoning)

# Also compute per-configuration (D and M separately) for scatter with all 26 points
config_rlpr = []
config_delta = []
config_reasoning = []
config_labels = []
config_prompt = []

for m in models:
    name, rd, rm, pd, pm, ad, am, reasoning = m
    # Detailed
    config_rlpr.append(rd)
    config_delta.append(rd - pd)
    config_reasoning.append(reasoning)
    config_labels.append(name)
    config_prompt.append("Detailed")
    # Minimal
    config_rlpr.append(rm)
    config_delta.append(rm - pm)
    config_reasoning.append(reasoning)
    config_labels.append(name)
    config_prompt.append("Minimal")

config_rlpr = np.array(config_rlpr)
config_delta = np.array(config_delta)
config_reasoning = np.array(config_reasoning)

# ── Linear regression: Δ ~ RLPR (all 26 configurations) ──
slope, intercept, r_value, p_value, std_err = stats.linregress(config_rlpr, config_delta)
predicted = intercept + slope * config_rlpr
residuals = config_delta - predicted

# Separate residuals by type
reasoning_mask = config_reasoning
standard_mask = ~config_reasoning
resid_reasoning = residuals[reasoning_mask]
resid_standard = residuals[standard_mask]

n_configs = len(config_rlpr)
print("=" * 60)
print(f"LINEAR REGRESSION: Δ ~ RLPR ({n_configs} configurations)")
print(f"  slope = {slope:.4f}, intercept = {intercept:.4f}")
print(f"  R² = {r_value**2:.4f}, p = {p_value:.6f}")
print()
print("RESIDUAL ANALYSIS")
print(f"  Reasoning models (n={len(resid_reasoning)}): mean residual = {resid_reasoning.mean():.3f}")
print(f"  Standard models  (n={len(resid_standard)}):  mean residual = {resid_standard.mean():.3f}")
print(f"  Difference (reasoning - standard) = {resid_reasoning.mean() - resid_standard.mean():.3f}")
print()

# ── Permutation test ──
observed_diff = resid_reasoning.mean() - resid_standard.mean()
n_perms = 10000
n_reasoning = reasoning_mask.sum()
rng = np.random.default_rng(42)
count = 0
for _ in range(n_perms):
    perm = rng.permutation(len(residuals))
    perm_reasoning = residuals[perm[:n_reasoning]]
    perm_standard = residuals[perm[n_reasoning:]]
    if abs(perm_reasoning.mean() - perm_standard.mean()) >= abs(observed_diff):
        count += 1
perm_p = count / n_perms
print(f"PERMUTATION TEST (two-sided, {n_perms} permutations)")
print(f"  Observed difference = {observed_diff:.3f}")
print(f"  p-value = {perm_p:.4f}")
print()

# ── AV comparison ──
av_reasoning = av_mean[is_reasoning]
av_standard = av_mean[~is_reasoning]
print("AV COMPARISON (model-level means)")
print(f"  Reasoning models: mean AV = {av_reasoning.mean():.3f} (n={len(av_reasoning)})")
print(f"  Standard models:  mean AV = {av_standard.mean():.3f} (n={len(av_standard)})")
print()

# ── Per-model summary ──
print("PER-MODEL SUMMARY (averaged across D and M):")
print(f"{'Model':<25} {'RLPR':>6} {'PSR':>6} {'Δ':>6} {'AV':>6} {'Type':>10}")
for i in range(len(names)):
    typ = "reasoning" if is_reasoning[i] else "standard"
    print(f"{names[i]:<25} {rlpr_mean[i]:>6.1f} {psr_mean[i]:>6.1f} {delta_mean[i]:>6.1f} {av_mean[i]:>6.2f} {typ:>10}")
print()

# ── FIGURE: Scatter plot with regression line ──
fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Plot all 26 configurations
for i in range(len(config_rlpr)):
    if config_reasoning[i]:
        color = '#E63946'  # red for reasoning
        marker = 'D'  # diamond
        zorder = 5
    else:
        color = '#457B9D'  # blue for standard
        marker = 'o'
        zorder = 4

    # Use different edge for D vs M
    edge = 'black' if config_prompt[i] == "Detailed" else 'none'
    alpha = 1.0 if config_prompt[i] == "Detailed" else 0.6

    ax.scatter(config_rlpr[i], config_delta[i], c=color, marker=marker,
               s=80, edgecolors=edge, linewidths=0.8, alpha=alpha, zorder=zorder)

# Regression line
x_line = np.linspace(25, 90, 100)
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1, label=f'OLS fit ($R^2$={r_value**2:.3f})')

# Legend entries
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E63946',
           markeredgecolor='black', markersize=9, label='Reasoning model'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#457B9D',
           markeredgecolor='black', markersize=9, label='Standard model'),
    Line2D([0], [0], linestyle='--', color='black', alpha=0.5, label=f'OLS fit ($R^2$={r_value**2:.3f})'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8.5, framealpha=0.9)

# Annotate reasoning models
for i in range(len(config_rlpr)):
    if config_reasoning[i] and config_prompt[i] == "Detailed":
        ax.annotate(config_labels[i], (config_rlpr[i], config_delta[i]),
                    textcoords="offset points", xytext=(8, -3), fontsize=7.5,
                    color='#E63946', fontweight='bold')

ax.set_xlabel('RLPR (%)', fontsize=11)
ax.set_ylabel('Accuracy–Stability Gap Δ (pp)', fontsize=11)
ax.set_title('Reasoning vs. Standard Models:\nGap Controlled for Accuracy Level', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(25, 95)

plt.tight_layout()
outdir = os.path.join(os.path.dirname(__file__), '..', 'output', 'aggregate', 'figures')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'paper_figure5_reasoning_vs_standard.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {outpath}")
plt.close()
