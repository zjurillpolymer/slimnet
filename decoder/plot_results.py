"""SLIMNet 结果可视化：训练曲线 + 预测 vs 真实散点图"""
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Nature 风格配置 ──
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})

PALETTE = {
    'blue_main': '#0F4D92',
    'blue_secondary': '#3775BA',
    'red_strong': '#B64342',
    'green_3': '#8BCF8B',
    'teal': '#42949E',
    'neutral_mid': '#767676',
}

TARGET_NAMES = ['Thermal diffusivity', 'Static dielectric const', 'Linear expansion']

_fallback = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('SLIMNET_ROOT', _fallback)
OUT_DIR = os.path.join(ROOT, 'figures')


def plot_training_curve(train_losses, val_losses, save=True):
    """Panel A: 训练 / 验证 loss 曲线"""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=PALETTE['blue_main'], lw=1.2, label='Train')
    ax.plot(epochs, val_losses, color=PALETTE['red_strong'], lw=1.2, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized MSE')
    ax.legend(fontsize=6.5)
    fig.tight_layout(pad=1.5)
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(f'{OUT_DIR}/training_curve.svg', bbox_inches='tight')
        fig.savefig(f'{OUT_DIR}/training_curve.png', dpi=300, bbox_inches='tight')
        print(f'  → saved training_curve.svg/png')
    plt.close(fig)
    return fig


def plot_scatter(preds, targets, save=True):
    """Panel B: 预测 vs 真实散点图（3 个子图）"""
    n_tasks = preds.shape[1]
    fig, axes = plt.subplots(1, n_tasks, figsize=(3 * n_tasks + 0.5, 3))

    for i, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        p, t = preds[:, i], targets[:, i]
        # R²
        mask = ~np.isnan(p) & ~np.isnan(t)
        r2 = stats.pearsonr(p[mask], t[mask])[0] ** 2

        ax.scatter(t, p, s=6, color=PALETTE['blue_main'], alpha=0.6, edgecolors='none')
        # 对角线 y=x
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                color=PALETTE['neutral_mid'], ls='--', lw=0.8)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel('Ground truth')
        ax.set_ylabel('Prediction')
        ax.set_title(name, fontsize=7.5)
        ax.text(0.95, 0.08, f'$R^2={r2:.3f}$', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    fig.tight_layout(pad=1.5)
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(f'{OUT_DIR}/pred_vs_true.svg', bbox_inches='tight')
        fig.savefig(f'{OUT_DIR}/pred_vs_true.png', dpi=300, bbox_inches='tight')
        print(f'  → saved pred_vs_true.svg/png')
    plt.close(fig)
    return fig


def plot_all(train_losses, val_losses, preds, targets):
    """生成两张图"""
    print('Generating figures...')
    plot_training_curve(train_losses, val_losses)
    plot_scatter(preds, targets)
    print(f'All figures saved to {OUT_DIR}/')
