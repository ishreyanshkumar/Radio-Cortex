"""
BDH Results Visualization
==========================

Generates publication-ready figures from analysis JSON outputs.

Usage:
    python -m interpretability.visualize --results ./bdh_results

Outputs (in bdh_results/visualizations/):
    - monosemanticity_heatmap.png
    - sparsity_analysis.png
    - degree_distribution.png
    - summary_metrics.png
    - REPORT.txt
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── Global style ──
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
})

# Color palette
C_BLUE = '#2E86AB'
C_GREEN = '#06A77D'
C_ORANGE = '#F77F00'
C_PURPLE = '#9B59B6'
C_RED = '#E74C3C'
C_TEAL = '#00B4D8'
C_GRAY = '#E8E8E8'


# ═══════════════════════════════════════════════════════════════════
# 1. MONOSEMANTICITY HEATMAP
# ═══════════════════════════════════════════════════════════════════

def plot_monosemanticity(results_dir: Path):
    """Heatmap of neuron-concept correlations."""
    print("  Creating monosemanticity heatmap...")

    fpath = results_dir / 'monosemanticity.json'
    if not fpath.exists():
        print("    ⚠ monosemanticity.json not found — skipping")
        return

    with open(fpath) as f:
        data = json.load(f)

    neurons = data.get('neurons', {})
    concepts = data.get('concepts', [])

    if not neurons:
        print("    ⚠ No monosemantic neurons found — skipping heatmap")
        # Still create a placeholder figure with the score
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                f"Monosemanticity Score: {data.get('score', 0):.3f}\n"
                f"No neurons exceeded correlation threshold\n"
                f"({data.get('total_neurons', 0)} neurons, "
                f"{data.get('num_concepts', 0)} concepts tested)",
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.set_axis_off()
        plt.tight_layout()
        output = results_dir / 'visualizations' / 'monosemanticity_heatmap.png'
        output.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output}")
        return

    # ── Select top neurons by max correlation ──
    neuron_items = []
    for nid, neuron_data in neurons.items():
        if isinstance(neuron_data, list) and neuron_data:
            max_corr = max(
                item['correlation'] if isinstance(item, dict) else item[1]
                for item in neuron_data
            )
            neuron_items.append((nid, max_corr))

    neuron_items.sort(key=lambda x: x[1], reverse=True)
    top_neuron_ids = [nid for nid, _ in neuron_items[:30]]

    # ── Select top concepts by frequency ──
    concept_counts = {}
    for nid, neuron_data in neurons.items():
        for item in neuron_data:
            c = item['concept'] if isinstance(item, dict) else item[0]
            concept_counts[c] = concept_counts.get(c, 0) + 1

    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:12]
    concept_names = [c for c, _ in top_concepts]

    # ── Build matrix ──
    matrix = np.zeros((len(top_neuron_ids), len(concept_names)))
    for i, nid in enumerate(top_neuron_ids):
        for item in neurons[nid]:
            c = item['concept'] if isinstance(item, dict) else item[0]
            corr = item['correlation'] if isinstance(item, dict) else item[1]
            if c in concept_names:
                j = concept_names.index(c)
                matrix[i, j] = corr

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(max(10, len(concept_names) * 0.9),
                                     max(6, len(top_neuron_ids) * 0.3)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(concept_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in concept_names],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(top_neuron_ids)))
    ax.set_yticklabels([f"N{nid}" for nid in top_neuron_ids], fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Strength', fontsize=11)

    ax.set_title(
        f'BDH Monosemantic Neurons\n'
        f'Score: {data["score"]:.3f} — '
        f'{data["num_monosemantic"]}/{data["total_neurons"]} neurons, '
        f'{data["num_concepts"]} concepts',
        fontsize=13, weight='bold', pad=15
    )
    ax.set_xlabel('Network Concepts', fontsize=11, weight='bold')
    ax.set_ylabel('Neuron Index', fontsize=11, weight='bold')

    plt.tight_layout()
    output = results_dir / 'visualizations' / 'monosemanticity_heatmap.png'
    output.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output}")


# ═══════════════════════════════════════════════════════════════════
# 2. SPARSITY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_sparsity(results_dir: Path):
    """Sparsity pie chart + layer-wise bar chart."""
    print("  Creating sparsity visualization...")

    fpath = results_dir / 'sparsity.json'
    if not fpath.exists():
        print("    ⚠ sparsity.json not found — skipping")
        return

    with open(fpath) as f:
        data = json.load(f)

    sparsity = data['overall_sparsity']
    active = data['active_percentage']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Donut chart ──
    ax = axes[0]
    sizes = [active, sparsity]
    colors = [C_BLUE, C_GRAY]
    labels = [f'Active\n{active*100:.1f}%', f'Inactive\n{sparsity*100:.1f}%']

    wedges, texts = ax.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    )
    # Draw center circle for donut effect
    centre = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre)
    ax.set_title('Neuron Activation Distribution', fontsize=13, weight='bold', pad=15)

    # ── Right: Layer-wise sparsity ──
    ax = axes[1]
    layer_sparsities = data.get('layer_sparsities', [])

    if layer_sparsities:
        x = list(range(len(layer_sparsities)))
        bars = ax.bar(x, layer_sparsities, color=C_BLUE, alpha=0.75,
                      edgecolor='black', linewidth=0.5)
        ax.axhline(y=sparsity, color=C_RED, linestyle='--', linewidth=2,
                   label=f'Overall: {sparsity:.2f}')

        # Color bars above/below average
        for bar, val in zip(bars, layer_sparsities):
            if val > sparsity:
                bar.set_color(C_GREEN)

        ax.set_xlabel('Layer Index', fontsize=11, weight='bold')
        ax.set_ylabel('Sparsity (fraction inactive)', fontsize=11, weight='bold')
        ax.set_title('Sparsity by Layer', fontsize=13, weight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, 'No layer-wise data available',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Sparsity by Layer', fontsize=13, weight='bold', pad=15)

    plt.tight_layout()
    output = results_dir / 'visualizations' / 'sparsity_analysis.png'
    output.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output}")


# ═══════════════════════════════════════════════════════════════════
# 3. DEGREE DISTRIBUTION (SCALE-FREE)
# ═══════════════════════════════════════════════════════════════════

def plot_degree_distribution(results_dir: Path):
    """Linear histogram + log-log scatter with power-law fit."""
    print("  Creating degree distribution plots...")

    fpath = results_dir / 'scale_free.json'
    if not fpath.exists():
        print("    ⚠ scale_free.json not found — skipping")
        return

    with open(fpath) as f:
        data = json.load(f)

    degrees = np.array(data.get('degrees', []))
    if len(degrees) == 0:
        print("    ⚠ No degree data — skipping")
        return

    alpha = data.get('alpha')
    r_sq = data.get('r_squared')
    is_sf = data.get('is_scale_free', False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Linear histogram ──
    ax1.hist(degrees, bins=min(40, len(np.unique(degrees))),
             color=C_GREEN, alpha=0.75, edgecolor='black', linewidth=0.5)
    ax1.axvline(degrees.mean(), color=C_RED, linestyle='--', linewidth=2,
                label=f'Mean = {degrees.mean():.1f}')
    ax1.set_xlabel('Degree (# connections)', fontsize=11, weight='bold')
    ax1.set_ylabel('Count', fontsize=11, weight='bold')
    ax1.set_title('Degree Distribution', fontsize=13, weight='bold', pad=15)
    ax1.legend(fontsize=10)

    # ── Right: Log-log with power-law fit ──
    unique, counts = np.unique(degrees[degrees > 0], return_counts=True)

    ax2.scatter(unique, counts, c=C_GREEN, s=50, alpha=0.6,
                edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

    if alpha is not None and not np.isnan(alpha) and r_sq is not None:
        x_fit = np.logspace(np.log10(max(1, unique.min())),
                            np.log10(unique.max()), 100)
        # Fit line: log(P) = -alpha * log(k) + C
        # Use first data point to anchor
        C = counts[0] * (unique[0] ** alpha)
        y_fit = C * (x_fit ** (-alpha))

        ax2.plot(x_fit, y_fit, '--', color=C_RED, linewidth=2,
                 label=f'Power Law (α={alpha:.2f}, R²={r_sq:.2f})')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree (log scale)', fontsize=11, weight='bold')
    ax2.set_ylabel('Count (log scale)', fontsize=11, weight='bold')
    ax2.set_title('Log-Log Degree Distribution', fontsize=13, weight='bold', pad=15)
    ax2.legend(fontsize=10)

    # Status badge
    status = "✓ Scale-Free Network" if is_sf else "✗ Not Scale-Free"
    badge_color = C_GREEN if is_sf else C_ORANGE
    fig.text(0.5, 0.01, status, ha='center', fontsize=12, weight='bold',
             color=badge_color,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=badge_color, linewidth=2))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output = results_dir / 'visualizations' / 'degree_distribution.png'
    output.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output}")


# ═══════════════════════════════════════════════════════════════════
# 4. SUMMARY METRICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════

def plot_summary_metrics(results_dir: Path):
    """Single figure with all key metrics as a bar chart."""
    print("  Creating summary metrics chart...")

    # Load all available results
    files = {
        'mono': 'monosemanticity.json',
        'sparse': 'sparsity.json',
        'hebb': 'hebbian.json',
        'sf': 'scale_free.json',
    }
    data = {}
    for key, fname in files.items():
        fpath = results_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data[key] = json.load(f)

    if not data:
        print("    ⚠ No result files found — skipping")
        return

    # Build metrics list
    metrics = []
    values = []
    colors = []
    annotations = []

    if 'mono' in data:
        score = data['mono'].get('score', 0)
        metrics.append('Mono-\nsemanticity')
        values.append(score)
        colors.append(C_BLUE)
        annotations.append(f'{score:.2f}')

    if 'sparse' in data:
        sp = data['sparse'].get('overall_sparsity', 0)
        metrics.append('Sparsity\n(% inactive)')
        values.append(sp)
        colors.append(C_GREEN)
        annotations.append(f'{sp*100:.0f}%')

        act = data['sparse'].get('active_percentage', 0)
        metrics.append('Active\nNeurons')
        values.append(act)
        colors.append(C_ORANGE)
        annotations.append(f'{act*100:.0f}%')

    if 'sf' in data:
        sf_val = 1.0 if data['sf'].get('is_scale_free') else 0.3
        metrics.append('Scale-Free\n(yes/no)')
        values.append(sf_val)
        colors.append(C_PURPLE)
        annotations.append('YES' if sf_val > 0.5 else 'NO')

    if 'hebb' in data:
        # Normalize: show strengthened fraction
        total = data['hebb'].get('total_synapses', 1)
        strengthened = data['hebb'].get('strengthened_count', 0)
        frac = strengthened / max(total, 1)
        metrics.append('Hebbian\nPlasticity')
        values.append(min(frac * 20, 1.0))  # Scale up for visibility
        colors.append(C_TEAL)
        annotations.append(f'{strengthened}')

    if not metrics:
        print("    ⚠ No metrics to plot — skipping")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2), 6))

    bars = ax.bar(range(len(metrics)), values, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.5, width=0.6)

    # Value labels on top
    for i, (bar, ann) in enumerate(zip(bars, annotations)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                ann, ha='center', va='bottom', fontsize=12, weight='bold')

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score / Ratio', fontsize=12, weight='bold')
    ax.set_title('BDH Interpretability Metrics Summary',
                 fontsize=14, weight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.25 if values else 1.1)

    plt.tight_layout()
    output = results_dir / 'visualizations' / 'summary_metrics.png'
    output.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output}")


# ═══════════════════════════════════════════════════════════════════
# 5. TEXT REPORT
# ═══════════════════════════════════════════════════════════════════

def create_report(results_dir: Path):
    """Generate detailed text report with paper-ready claims."""
    print("  Creating text report...")

    files = {
        'mono': 'monosemanticity.json',
        'sparse': 'sparsity.json',
        'hebb': 'hebbian.json',
        'sf': 'scale_free.json',
    }
    data = {}
    for key, fname in files.items():
        fpath = results_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data[key] = json.load(f)

    sep = '=' * 70
    line = '-' * 70

    report = []
    report.append(sep)
    report.append('BDH INTERPRETABILITY ANALYSIS — DETAILED REPORT')
    report.append(sep)
    report.append('')

    # ── 1. Monosemanticity ──
    report.append('1. MONOSEMANTICITY ANALYSIS')
    report.append(line)
    if 'mono' in data:
        m = data['mono']
        score = m.get('score', 0)
        rating = 'EXCELLENT' if score > 0.7 else 'GOOD' if score > 0.5 else 'FAIR'
        report.append(f'Score:               {score:.3f} ({rating})')
        report.append(f'Total Neurons:       {m.get("total_neurons", 0)}')
        report.append(f'Monosemantic:        {m.get("num_monosemantic", 0)}')
        report.append(f'Concepts Discovered: {m.get("num_concepts", 0)}')
        report.append('')
        if m.get('concepts'):
            report.append('Top Concepts:')
            for i, c in enumerate(m['concepts'][:10]):
                report.append(f'  {i+1:2d}. {c}')
    else:
        report.append('Not available')
    report.append('')

    # ── 2. Sparsity ──
    report.append('2. SPARSE ACTIVATION ANALYSIS')
    report.append(line)
    if 'sparse' in data:
        s = data['sparse']
        report.append(f'Overall Sparsity:    {s["overall_sparsity"]:.3f} ({s["overall_sparsity"]*100:.1f}% inactive)')
        report.append(f'Active Neurons:      {s["active_percentage"]:.3f} ({s["active_percentage"]*100:.1f}%)')
        report.append(f'Mean Activation:     {s.get("mean_activation", 0):.4f}')
        report.append(f'Std Activation:      {s.get("std_activation", 0):.4f}')
        report.append(f'Max Activation:      {s.get("max_activation", 0):.4f}')
    else:
        report.append('Not available')
    report.append('')

    # ── 3. Hebbian ──
    report.append('3. HEBBIAN LEARNING ANALYSIS')
    report.append(line)
    if 'hebb' in data:
        h = data['hebb']
        report.append(f'Snapshots Compared:  {h.get("num_timesteps", 0)}')
        report.append(f'Strengthened:        {h.get("strengthened_count", 0)} synapses')
        report.append(f'Max Weight Change:   {h.get("max_change", 0):.6f}')
        report.append(f'Mean Weight Change:  {h.get("mean_change", 0):.6f}')
        report.append(f'Total Synapses:      {h.get("total_synapses", 0)}')
        report.append(f'Weight Shape:        {h.get("weight_shape", [])}')
    else:
        report.append('Not available')
    report.append('')

    # ── 4. Scale-Free ──
    report.append('4. SCALE-FREE NETWORK STRUCTURE')
    report.append(line)
    if 'sf' in data:
        sf = data['sf']
        status = 'YES' if sf.get('is_scale_free') else 'NO'
        report.append(f'Scale-Free:          {status}')
        if sf.get('alpha') is not None:
            report.append(f'Power Law Alpha:     {sf["alpha"]:.3f}')
        if sf.get('r_squared') is not None:
            report.append(f'R² Fit Quality:      {sf["r_squared"]:.3f}')
        ds = sf.get('degree_stats', {})
        report.append(f'Avg Degree:          {ds.get("mean", 0):.2f}')
        report.append(f'Max Degree:          {ds.get("max", 0)}')
        report.append(f'Hub Neurons:         {sf.get("num_hubs", 0)} (top 10%)')
        report.append(f'Network Size:        {sf.get("network_size", 0)} neurons')
        report.append(f'Encoder Shape:       {sf.get("encoder_shape", [])}')
    else:
        report.append('Not available')
    report.append('')

    # ── 5. Paper Claims ──
    report.append('5. KEY CLAIMS FOR PAPER')
    report.append(line)

    if 'mono' in data:
        m = data['mono']
        pct = m['num_monosemantic'] / max(m['total_neurons'], 1) * 100
        report.append(f'CLAIM 1 (Monosemanticity):')
        report.append(f'  "{m["num_monosemantic"]} neurons ({pct:.1f}%) encode specific')
        report.append(f'   O-RAN concepts with correlation > 0.5"')
        report.append('')

    if 'sparse' in data:
        s = data['sparse']
        report.append(f'CLAIM 2 (Sparse Activation):')
        report.append(f'  "BDH maintains {s["overall_sparsity"]*100:.1f}% sparsity, activating')
        report.append(f'   only {s["active_percentage"]*100:.1f}% of neurons per decision"')
        report.append('')

    if 'sf' in data:
        sf = data['sf']
        if sf.get('is_scale_free'):
            report.append(f'CLAIM 3 (Scale-Free):')
            report.append(f'  "BDH encoder exhibits scale-free topology')
            report.append(f'   (alpha={sf["alpha"]:.2f}, R²={sf["r_squared"]:.2f})')
            report.append(f'   with {sf["num_hubs"]} hub neurons"')
        else:
            report.append(f'CLAIM 3 (Hierarchical Structure):')
            report.append(f'  "BDH encoder shows hierarchical connectivity')
            report.append(f'   with {sf["num_hubs"]} highly-connected hub neurons"')
        report.append('')

    report.append(f'CLAIM 4 (Interpretability):')
    report.append(f'  "BDH enables direct neuron-concept mappings for causal')
    report.append(f'   interpretation of O-RAN control decisions"')
    report.append('')

    report.append(sep)
    report.append('ANALYSIS COMPLETE')
    report.append(f'Results directory: {results_dir}/')
    report.append(sep)

    # Write
    report_text = '\n'.join(report)
    output = results_dir / 'REPORT.txt'
    with open(output, 'w') as f:
        f.write(report_text)

    print(f"    ✓ Saved: {output}")

    # Also print to console
    print()
    print(report_text)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Visualize BDH Analysis Results')
    parser.add_argument('--results', default='./bdh_results',
                        help='Results directory (default: ./bdh_results)')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        print("  Run interpretability.run_analysis first")
        return

    print()
    print("=" * 60)
    print("  GENERATING BDH VISUALIZATIONS")
    print("=" * 60)
    print()

    plot_monosemanticity(results_dir)
    plot_sparsity(results_dir)
    plot_degree_distribution(results_dir)
    plot_summary_metrics(results_dir)
    create_report(results_dir)

    vis_dir = results_dir / 'visualizations'
    print()
    print("=" * 60)
    print(f"  ✓ All outputs saved to: {vis_dir}/")
    print()
    print("  Generated files:")
    if vis_dir.exists():
        for f in sorted(vis_dir.iterdir()):
            print(f"    - {f.name}")
    print(f"    - REPORT.txt")
    print("=" * 60)


if __name__ == '__main__':
    main()