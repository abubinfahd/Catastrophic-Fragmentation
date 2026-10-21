"""The notebook's two plots, using calculated tables instead of hard-coded values."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_figures(summary, ranking, folder, sample_size, config):
    folder.mkdir(exist_ok=True)
    domains = list(config['domains'])
    models = ranking.Model.tolist()
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3']
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=300)
    x, width = np.arange(len(domains)), .15
    for i, model in enumerate(models):
        values = summary[summary.Model == model].set_index('Domain').loc[domains, 'Disparity_Multiplier']
        bars = axis.bar(x + (i-(len(models)-1)/2)*width, values, width, label=model,
                        color=colors[i % len(colors)], edgecolor='black', linewidth=.5)
        axis.bar_label(bars, padding=3, fmt='%.2f', fontsize=8, weight='bold')
    axis.set_ylabel('Mean token-count ratio (BN / EN)')
    axis.set_title(f'Tokenization disparity ({sample_size:,} pairs per dataset)', pad=65)
    axis.set_xticks(x, ['CCMatrix' if 'Web & Wiki' in d else d.split(' (')[0] for d in domains])
    axis.set_ylim(0, max(1.2, float(summary.Disparity_Multiplier.max())*1.2))
    axis.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=8)
    figure.tight_layout()
    for ext in ['pdf', 'png']:
        figure.savefig(folder / ('fig1_cross_domain_disparity.' + ext), dpi=300, bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = axis.barh(ranking.Model, ranking.Grand_Multiplier)
    axis.bar_label(bars, padding=5, fmt='%.2f', fontsize=9)
    axis.invert_yaxis()
    axis.axvline(1, color='red', linestyle='--', linewidth=1)
    axis.set_xlim(0, max(1.2, float(ranking.Grand_Multiplier.max())*1.15))
    axis.set_xlabel('Overall arithmetic mean disparity (notebook calculation)')
    figure.tight_layout()
    for ext in ['pdf', 'png']:
        figure.savefig(folder / ('fig2_grand_inefficiency_ranking.' + ext), dpi=300, bbox_inches='tight')
    plt.close(figure)
