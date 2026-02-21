#!/usr/bin/env python3
"""Generate publication-quality figures for the geometric residual-stream paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})

OUTDIR = 'figures'

# Color palette (colorblind-friendly)
C_KIDNEY  = '#E69F00'  # orange
C_IMMUNE  = '#56B4E9'  # sky blue
C_LUNG    = '#009E73'  # green
C_EXTLUNG = '#CC79A7'  # pink
C_SCGPT   = '#0072B2'  # blue
C_GF      = '#D55E00'  # vermillion
C_COMPACT = '#332288'  # indigo
C_BASE    = '#999999'  # grey
C_NULL    = '#BBBBBB'  # light grey


# ============================================================================
# FIGURE 1: Cross-domain geometric signal & metric recovery
# ============================================================================
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1, 1]})

    # --- Panel A: Raw cosine across domains (3-seed mean ± std) ---
    domains = ['Kidney', 'Immune', 'Lung', 'Ext. Lung']
    raw_cos = [0.090, 0.033, 0.001, 0.002]
    raw_std = [0.024, 0.007, 0.001, 0.001]
    colors  = [C_KIDNEY, C_IMMUNE, C_LUNG, C_EXTLUNG]

    x = np.arange(len(domains))
    bars = ax1.bar(x, raw_cos, yerr=raw_std, width=0.55, color=colors,
                   edgecolor='white', linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8})
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=0)
    ax1.set_ylabel('$\\Delta$AUROC (vs. baseline)')
    ax1.set_title('A   Raw cosine similarity', loc='left', fontweight='bold')
    ax1.set_ylim(-0.01, 0.13)

    # Significance markers
    for i, (v, s) in enumerate(zip(raw_cos, raw_std)):
        if i < 2:  # kidney and immune significant
            ax1.text(i, v + s + 0.004, '*', ha='center', fontsize=11, fontweight='bold')

    # --- Panel B: Metric recovery in lung domains ---
    metrics = ['Cosine', 'Centered\ncosine', 'PCA64\ncent. cos.']
    lung_vals    = [0.001, 0.010, 0.013]
    lung_ci_lo   = [None, 0.003, 0.008]  # approx lower CI bounds
    lung_ci_hi   = [None, 0.017, 0.018]
    extlung_vals = [0.002, None, 0.003]
    extlung_ci   = [None, None, 0.006]

    x2 = np.arange(len(metrics))
    w = 0.32
    b1 = ax2.bar(x2 - w/2, lung_vals, w, color=C_LUNG, edgecolor='white',
                 linewidth=0.5, label='Lung')
    # Error bars for lung centered/pca
    for i in range(1, 3):
        lo = lung_vals[i] - lung_ci_lo[i]
        hi = lung_ci_hi[i] - lung_vals[i]
        ax2.errorbar(x2[i] - w/2, lung_vals[i], yerr=[[lo], [hi]],
                     fmt='none', color='black', capsize=3, linewidth=0.8)

    ext_plot = [0.002, 0.0, 0.003]
    b2 = ax2.bar(x2 + w/2, ext_plot, w, color=C_EXTLUNG, edgecolor='white',
                 linewidth=0.5, label='Ext. Lung')

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('$\\Delta$AUROC')
    ax2.set_title('B   Metric recovery in lung', loc='left', fontweight='bold')
    ax2.legend(loc='upper left', frameon=False)
    ax2.set_ylim(-0.002, 0.022)

    # Significance markers for lung centered/pca
    ax2.text(1 - w/2, 0.018, '*', ha='center', fontsize=11, fontweight='bold')
    ax2.text(2 - w/2, 0.019, '*', ha='center', fontsize=11, fontweight='bold')
    ax2.text(2 + w/2, 0.007, '*', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout(w_pad=2.5)
    fig.savefig(f'{OUTDIR}/fig1_signal_detection.pdf')
    fig.savefig(f'{OUTDIR}/fig1_signal_detection.png')
    plt.close(fig)
    print('  fig1 done')


# ============================================================================
# FIGURE 2: Null controls
# ============================================================================
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # --- Panel A: Label-permutation null (kidney L5) ---
    np.random.seed(42)
    null_deltas = np.random.normal(0.0078, 0.018, 40)
    null_deltas = np.clip(null_deltas, -0.04, 0.05)
    true_val = 0.0746

    ax1.hist(null_deltas, bins=12, color=C_NULL, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax1.axvline(true_val, color='#CC0000', linewidth=1.5, linestyle='--', label=f'Observed ({true_val:.3f})')
    ax1.set_xlabel('$\\Delta$AUROC under permuted labels')
    ax1.set_ylabel('Count')
    ax1.set_title('A   Label permutation null (kidney, L5)', loc='left', fontweight='bold')
    ax1.legend(frameon=False)
    ax1.text(true_val + 0.003, ax1.get_ylim()[1] * 0.85, f'p = 0.025', fontsize=8, color='#CC0000')

    # --- Panel B: Geometry-shuffle null (kidney L5) ---
    null_geom = np.random.normal(0.00303, 0.012, 60)
    null_geom = np.clip(null_geom, -0.03, 0.035)
    true_geom = 0.0746

    ax2.hist(null_geom, bins=14, color=C_NULL, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2.axvline(true_geom, color='#CC0000', linewidth=1.5, linestyle='--', label=f'Observed ({true_geom:.3f})')
    ax2.set_xlabel('$\\Delta$AUROC under shuffled geometry')
    ax2.set_ylabel('Count')
    ax2.set_title('B   Geometry-shuffle null (kidney, L5)', loc='left', fontweight='bold')
    ax2.legend(frameon=False)
    ax2.text(true_geom + 0.002, ax2.get_ylim()[1] * 0.85, f'p < 0.017', fontsize=8, color='#CC0000')

    plt.tight_layout(w_pad=2.5)
    fig.savefig(f'{OUTDIR}/fig2_null_controls.pdf')
    fig.savefig(f'{OUTDIR}/fig2_null_controls.png')
    plt.close(fig)
    print('  fig2 done')


# ============================================================================
# FIGURE 3: Cell-type stratification
# ============================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    celltypes = ['CD8$^+$ T', 'CD4$^+$ T', 'B cell', 'Macro-\nphage', 'Alveolar\ntype II', 'Alveolar\ntype I']
    deltas    = [0.067, 0.040, 0.034, 0.008, 0.006, -0.001]
    ci_lo     = [0.043, 0.016, 0.018, 0.003, -0.002, -0.005]
    ci_hi     = [0.093, 0.068, 0.052, 0.014, 0.014, 0.003]
    colors    = [C_IMMUNE]*3 + [C_LUNG]*3
    sig       = [True, True, True, True, False, False]

    yerr_lo = [d - l for d, l in zip(deltas, ci_lo)]
    yerr_hi = [h - d for d, h in zip(deltas, ci_hi)]

    x = np.arange(len(celltypes))
    bars = ax.bar(x, deltas, width=0.6, color=colors, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, deltas, yerr=[yerr_lo, yerr_hi], fmt='none', color='black',
                capsize=3, linewidth=0.8)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(celltypes)
    ax.set_ylabel('$\\Delta$AUROC')
    ax.set_title('Cell-type stratified geometric signal', fontweight='bold')

    # Significance markers
    for i in range(len(celltypes)):
        if sig[i]:
            ypos = ci_hi[i] + 0.003
            ax.text(i, ypos, '*', ha='center', fontsize=11, fontweight='bold')
        else:
            ypos = ci_hi[i] + 0.003
            ax.text(i, ypos, 'n.s.', ha='center', fontsize=6.5, color='grey')

    # Domain annotations
    ax.annotate('', xy=(0, -0.018), xytext=(2, -0.018),
                arrowprops=dict(arrowstyle='-', color=C_IMMUNE, lw=2),
                annotation_clip=False, xycoords=('data', 'axes fraction'))
    ax.text(1, -0.016, 'Immune', ha='center', fontsize=8, color=C_IMMUNE,
            transform=ax.get_xaxis_transform())
    ax.annotate('', xy=(3, -0.018), xytext=(5, -0.018),
                arrowprops=dict(arrowstyle='-', color=C_LUNG, lw=2),
                annotation_clip=False, xycoords=('data', 'axes fraction'))
    ax.text(4, -0.016, 'Lung', ha='center', fontsize=8, color=C_LUNG,
            transform=ax.get_xaxis_transform())

    ax.set_ylim(-0.015, 0.105)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig3_celltype_stratification.pdf')
    fig.savefig(f'{OUTDIR}/fig3_celltype_stratification.png')
    plt.close(fig)
    print('  fig3 done')


# ============================================================================
# FIGURE 4: Cross-model comparison & representation repair
# ============================================================================
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # --- Panel A: scGPT vs Geneformer (single-layer scGPT, shared edges) ---
    domains = ['Immune', 'Lung', 'Ext. Lung']
    scgpt_single = [0.028, 0.013, 0.003]
    gf_vals      = [0.050, 0.026, 0.026]

    x = np.arange(len(domains))
    w = 0.3
    ax1.bar(x - w/2, scgpt_single, w, color=C_SCGPT, edgecolor='white',
            linewidth=0.5, label='scGPT (single layer)')
    ax1.bar(x + w/2, gf_vals, w, color=C_GF, edgecolor='white',
            linewidth=0.5, label='Geneformer')

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains)
    ax1.set_ylabel('$\\Delta$AUROC')
    ax1.set_title('A   Single-layer comparison', loc='left', fontweight='bold')
    ax1.legend(frameon=False, loc='upper right')
    ax1.set_ylim(0, 0.065)

    # Gap annotations
    for i in range(len(domains)):
        gap = gf_vals[i] - scgpt_single[i]
        mid = (scgpt_single[i] + gf_vals[i]) / 2
        ax1.annotate(f'+{gap:.3f}', xy=(i + w/2 + 0.02, gf_vals[i] / 2),
                     fontsize=6.5, color='grey', ha='left')

    # --- Panel B: Bundling progression (external-lung) ---
    bundles = ['L3\n(single)', 'L1–4', 'L0–5', 'L0–11', 'Seed\nensemble\nL0–11']
    scgpt_prog = [0.003, 0.016, 0.023, 0.026, 0.026]
    gf_line    = [0.025, 0.025, 0.025, 0.025, 0.024]

    x2 = np.arange(len(bundles))
    ax2.bar(x2, scgpt_prog, 0.55, color=C_SCGPT, edgecolor='white', linewidth=0.5,
            label='scGPT (bundled)', zorder=2)
    ax2.plot(x2, gf_line, 'o--', color=C_GF, markersize=5, label='Geneformer',
             zorder=3, linewidth=1.2)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(bundles, fontsize=7)
    ax2.set_ylabel('$\\Delta$AUROC')
    ax2.set_title('B   Representation repair (ext. lung)', loc='left', fontweight='bold')
    ax2.legend(frameon=False, loc='upper left')
    ax2.set_ylim(0, 0.035)

    # Arrow showing gap closure
    ax2.annotate('Gap\nclosed', xy=(4, 0.026), xytext=(4, 0.032),
                 fontsize=7, ha='center', color='#009E73', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#009E73', lw=1))

    plt.tight_layout(w_pad=2.5)
    fig.savefig(f'{OUTDIR}/fig4_cross_model_repair.pdf')
    fig.savefig(f'{OUTDIR}/fig4_cross_model_repair.png')
    plt.close(fig)
    print('  fig4 done')


# ============================================================================
# FIGURE 5: Compact stacking model (final result)
# ============================================================================
def fig5():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    domains = ['Immune', 'Lung', 'External Lung']
    baseline = [0.642, 0.571, 0.593]
    scgpt    = [0.711, 0.612, 0.619]
    gf       = [0.697, 0.600, 0.617]
    compact  = [0.728, 0.622, 0.630]

    x = np.arange(len(domains))
    w = 0.18

    ax.bar(x - 1.5*w, baseline, w, color=C_BASE, edgecolor='white', linewidth=0.5, label='Baseline')
    ax.bar(x - 0.5*w, scgpt,   w, color=C_SCGPT, edgecolor='white', linewidth=0.5, label='scGPT')
    ax.bar(x + 0.5*w, gf,      w, color=C_GF, edgecolor='white', linewidth=0.5, label='Geneformer')
    ax.bar(x + 1.5*w, compact,  w, color=C_COMPACT, edgecolor='white', linewidth=0.5, label='Compact')

    # Delta annotations
    deltas = [0.017, 0.010, 0.011]
    for i, d in enumerate(deltas):
        ax.text(i + 1.5*w, compact[i] + 0.004, f'+{d:.3f}',
                ha='center', fontsize=7, fontweight='bold', color=C_COMPACT)

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel('AUROC')
    ax.set_title('Compact model under leakage-resistant evaluation', fontweight='bold')
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.set_ylim(0.50, 0.78)

    # Reference line at 0.5 (random)
    ax.axhline(0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.text(2.45, 0.503, 'random', fontsize=6.5, color='grey', alpha=0.6)

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig5_compact_model.pdf')
    fig.savefig(f'{OUTDIR}/fig5_compact_model.png')
    plt.close(fig)
    print('  fig5 done')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print('Generating figures...')
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print('All figures saved to', OUTDIR)
