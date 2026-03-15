"""
Phase 2B: Cross-fragment generalization test for 3D depth features.

Loads pre-computed depth feature maps from results/depth_features/ (must run
run_depth_features.py first), then evaluates cross-fragment generalization
using corrected methodology:
  - Block-level sampling at 16x16, 32x32, 64x64
  - Balanced accuracy as primary metric (threshold-dependent)
  - AUC as secondary metric
  - Compares depth features vs 2D baselines

Usage:
    conda activate vesuvius
    python -u scripts/run_depth_cross_validation.py
"""
import sys, os, gc, glob, time
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vesuvius_preprocess.texture_enhance import (
    compute_gradient_magnitude, compute_structure_tensor_features)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', 'data', 'fragments')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

FRAGMENTS = [
    ('frag1', 59, 'Paris 2'),
    ('frag2', 64, 'Paris 2'),
    ('frag3', 49, 'Paris 1'),
]

BLOCK_SIZES = [16, 32, 64]
INK_THRESHOLD = 0.5
MIN_VALID_FRAC = 0.1

DEPTH_FEATURE_NAMES = [
    'depth_gradient_mag', 'max_depth_gradient', 'depth_variance',
    'depth_range', 'intensity_centroid', 'peak_depth_layer',
    'depth_skewness', 'depth_kurtosis', 'top_bottom_ratio', 'adj_layer_corr',
]


def load_feature_maps(frag_name, peak_layer):
    """Load cached depth feature .npy files + compute 2D baselines from peak layer."""
    feat_dir = os.path.join(RESULTS_DIR, 'depth_features', frag_name)
    features = {}

    # Load cached depth features
    for fn in DEPTH_FEATURE_NAMES:
        path = os.path.join(feat_dir, f'{fn}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cached feature: {path}. Run run_depth_features.py first.")
        features[fn] = np.load(path)

    # Load ink mask
    frag_dir = os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface')
    ink = np.array(Image.open(os.path.join(frag_dir, 'inklabels.png'))) > 0

    # Load peak layer for 2D baselines
    sv_dir = os.path.join(frag_dir, 'surface_volume')
    tif_files = sorted(glob.glob(os.path.join(sv_dir, '*.tif')))
    peak = tifffile.imread(tif_files[peak_layer]).astype(np.float32)
    features['raw_intensity'] = peak
    features['gradient_magnitude'] = compute_gradient_magnitude(peak)
    st = compute_structure_tensor_features(peak, sigma=1.0)
    features['st_coherence'] = st['coherence']
    del peak, st

    return features, ink


def blockify_vectorized(feature_maps, ink_mask, block_size):
    """Tile feature maps into non-overlapping blocks."""
    H, W = ink_mask.shape
    n_rows = H // block_size
    n_cols = W // block_size
    H_trim = n_rows * block_size
    W_trim = n_cols * block_size

    ink_trimmed = ink_mask[:H_trim, :W_trim].astype(np.float32)
    ink_blocks = ink_trimmed.reshape(n_rows, block_size, n_cols, block_size)
    ink_frac = ink_blocks.mean(axis=(1, 3))

    first_feat = next(iter(feature_maps.values()))
    feat_trimmed = first_feat[:H_trim, :W_trim]
    feat_blocks = feat_trimmed.reshape(n_rows, block_size, n_cols, block_size)
    valid_frac = (feat_blocks != 0).astype(np.float32).mean(axis=(1, 3))

    valid_mask = valid_frac >= MIN_VALID_FRAC
    labels = (ink_frac > INK_THRESHOLD).astype(np.float64)

    flat_valid = valid_mask.ravel()
    block_labels = labels.ravel()[flat_valid]

    block_features = {}
    for name, fmap in feature_maps.items():
        ft = fmap[:H_trim, :W_trim].reshape(n_rows, block_size, n_cols, block_size)
        block_means = ft.mean(axis=(1, 3))
        block_features[name] = block_means.ravel()[flat_valid].astype(np.float32)

    n_ink = int(block_labels.sum())
    n_noink = int(len(block_labels) - n_ink)
    return block_features, block_labels, n_ink, n_noink


def evaluate_model(clf, scaler, X_test, y_test):
    """Compute AUC, balanced accuracy, log-loss."""
    X_scaled = scaler.transform(X_test)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    y_pred = clf.predict(X_scaled)
    auc = roc_auc_score(y_test, y_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    return auc, bal_acc, ll


def main():
    print("="*70, flush=True)
    print("PHASE 2B: 3D DEPTH FEATURES — CROSS-FRAGMENT GENERALIZATION", flush=True)
    print("Block-level sampling at multiple sizes, balanced accuracy primary", flush=True)
    print("="*70, flush=True)

    scroll_map = {f[0]: f[2] for f in FRAGMENTS}
    frag_names = [f[0] for f in FRAGMENTS]

    # Load all feature maps (from cached .npy files + compute 2D baselines)
    frag_feature_maps = {}
    frag_ink_masks = {}

    for frag_name, peak_layer, scroll in FRAGMENTS:
        print(f"\n--- Loading {frag_name} ({scroll}) ---", flush=True)
        t0 = time.time()
        features, ink = load_feature_maps(frag_name, peak_layer)
        print(f"  Loaded {len(features)} features in {time.time()-t0:.1f}s", flush=True)
        print(f"  Ink: {ink.sum():,}/{ink.size:,} ({100*ink.sum()/ink.size:.1f}%)", flush=True)
        frag_feature_maps[frag_name] = features
        frag_ink_masks[frag_name] = ink

    feature_names = list(frag_feature_maps[frag_names[0]].keys())
    print(f"\n{len(feature_names)} features: {feature_names}", flush=True)

    # Define feature sets
    depth_only = [f for f in feature_names
                  if f not in ('raw_intensity', 'gradient_magnitude', 'st_coherence')]
    twod_only = ['raw_intensity', 'gradient_magnitude', 'st_coherence']

    feature_sets = {name: [name] for name in feature_names}
    feature_sets['ALL_DEPTH'] = depth_only
    feature_sets['ALL_2D'] = twod_only
    feature_sets['ALL_COMBINED'] = feature_names

    # Cross-validation at each block size
    all_results = []

    for block_size in BLOCK_SIZES:
        print(f"\n{'='*70}", flush=True)
        print(f"BLOCK SIZE: {block_size}x{block_size}", flush=True)
        print(f"{'='*70}", flush=True)

        frag_blocks = {}
        for frag_name in frag_names:
            t0 = time.time()
            bf, bl, ni, nn = blockify_vectorized(
                frag_feature_maps[frag_name],
                frag_ink_masks[frag_name],
                block_size)
            frag_blocks[frag_name] = {'features': bf, 'y': bl}
            print(f"  {frag_name}: {ni+nn} blocks ({ni} ink, {nn} non-ink) "
                  f"[{time.time()-t0:.1f}s]", flush=True)

        for set_name, feat_list in feature_sets.items():
            for i, train_frag in enumerate(frag_names):
                fd = frag_blocks[train_frag]
                X_train = np.column_stack([fd['features'][f] for f in feat_list])
                y_train = fd['y']

                valid_train = np.all(np.isfinite(X_train), axis=1)
                X_train = X_train[valid_train]
                y_train = y_train[valid_train]

                if len(np.unique(y_train)) < 2:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_train_s, y_train)

                for j, test_frag in enumerate(frag_names):
                    fd_test = frag_blocks[test_frag]
                    X_test = np.column_stack([fd_test['features'][f] for f in feat_list])
                    y_test = fd_test['y']

                    valid_test = np.all(np.isfinite(X_test), axis=1)
                    X_test = X_test[valid_test]
                    y_test = y_test[valid_test]

                    if len(np.unique(y_test)) < 2:
                        continue

                    auc, bal_acc, ll = evaluate_model(clf, scaler, X_test, y_test)

                    all_results.append({
                        'block_size': block_size,
                        'feature': set_name,
                        'n_features': len(feat_list),
                        'train_fragment': train_frag,
                        'test_fragment': test_frag,
                        'auc': round(auc, 4),
                        'balanced_accuracy': round(bal_acc, 4),
                        'log_loss': round(ll, 4),
                        'same_fragment': train_frag == test_frag,
                        'same_scroll': scroll_map[train_frag] == scroll_map[test_frag],
                    })

        # Print key results for this block size
        df_bs = pd.DataFrame([r for r in all_results if r['block_size'] == block_size])
        if df_bs.empty:
            continue

        for sn in ['raw_intensity', 'depth_variance', 'depth_range',
                    'top_bottom_ratio', 'ALL_DEPTH', 'ALL_COMBINED']:
            sub = df_bs[df_bs['feature'] == sn]
            if sub.empty:
                continue
            print(f"\n  {sn} @ {block_size}x{block_size}:", flush=True)
            print(f"    {'Train':<8} {'Test':<8} {'BalAcc':>7} {'AUC':>7}", flush=True)
            for _, row in sub.iterrows():
                marker = '*' if row['same_fragment'] else ' '
                print(f"    {row['train_fragment']:<8} {row['test_fragment']:<8} "
                      f"{row['balanced_accuracy']:>7.4f} {row['auc']:>7.4f}{marker}", flush=True)

    # Summary table
    df_all = pd.DataFrame(all_results)
    detail_path = os.path.join(RESULTS_DIR, 'tables', 'depth_cross_fragment_detail.csv')
    df_all.to_csv(detail_path, index=False)
    print(f"\nSaved: {detail_path}", flush=True)

    summary_rows = []
    for block_size in BLOCK_SIZES:
        df_bs = df_all[df_all['block_size'] == block_size]
        for feat_name in feature_sets.keys():
            sub = df_bs[df_bs['feature'] == feat_name]
            if sub.empty:
                continue

            within = sub[sub['same_fragment']]['balanced_accuracy'].mean()
            cross = sub[~sub['same_fragment']]['balanced_accuracy'].mean()
            within_auc = sub[sub['same_fragment']]['auc'].mean()
            cross_auc = sub[~sub['same_fragment']]['auc'].mean()
            cs_ba = sub[(~sub['same_fragment']) & (~sub['same_scroll'])]['balanced_accuracy']
            cs_auc = sub[(~sub['same_fragment']) & (~sub['same_scroll'])]['auc']

            summary_rows.append({
                'block_size': block_size,
                'feature': feat_name,
                'n_features': sub['n_features'].iloc[0],
                'primary_metric': 'balanced_accuracy',
                'within_bal_acc': round(within, 4),
                'cross_bal_acc': round(cross, 4),
                'bal_acc_gap': round(within - cross, 4),
                'cross_scroll_bal_acc': round(cs_ba.mean(), 4) if len(cs_ba) > 0 else np.nan,
                'within_auc': round(within_auc, 4),
                'cross_auc': round(cross_auc, 4),
                'auc_gap': round(within_auc - cross_auc, 4),
                'cross_scroll_auc': round(cs_auc.mean(), 4) if len(cs_auc) > 0 else np.nan,
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, 'tables', 'depth_feature_auc_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}", flush=True)

    for block_size in BLOCK_SIZES:
        print(f"\n{'='*80}", flush=True)
        print(f"SUMMARY — Block size {block_size}x{block_size}", flush=True)
        print(f"{'='*80}", flush=True)
        sub = df_summary[df_summary['block_size'] == block_size].sort_values(
            'cross_bal_acc', ascending=False)
        print(sub[['feature', 'n_features', 'within_bal_acc', 'cross_bal_acc',
                    'bal_acc_gap', 'within_auc', 'cross_auc', 'auc_gap']].to_string(index=False),
              flush=True)

    # Visualizations
    print("\nGenerating visualizations...", flush=True)

    # Balanced accuracy heatmaps at 32x32
    target_bs = 32
    key_features = ['raw_intensity', 'depth_variance', 'depth_range',
                    'top_bottom_ratio', 'intensity_centroid', 'depth_gradient_mag',
                    'ALL_DEPTH', 'ALL_2D', 'ALL_COMBINED']
    key_features = [f for f in key_features if f in feature_sets]

    n_plots = len(key_features)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    axes = axes.flatten()

    df_target = df_all[df_all['block_size'] == target_bs]
    for idx, feat_name in enumerate(key_features):
        ax = axes[idx]
        sub = df_target[df_target['feature'] == feat_name]
        if sub.empty:
            ax.set_visible(False)
            continue
        matrix = np.zeros((3, 3))
        for _, row in sub.iterrows():
            i = frag_names.index(row['train_fragment'])
            j = frag_names.index(row['test_fragment'])
            matrix[i, j] = row['balanced_accuracy']

        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=frag_names, yticklabels=frag_names,
                    vmin=0.4, vmax=0.8, ax=ax, cbar=False,
                    annot_kws={'fontsize': 10})
        ax.set_title(feat_name, fontsize=9)
        ax.set_ylabel('Train' if idx % ncols == 0 else '')
        ax.set_xlabel('Test')

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Balanced Accuracy — {target_bs}x{target_bs} blocks', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'depth_cross_fragment_heatmaps.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    # Depth vs 2D comparison bar chart
    fig, ax = plt.subplots(figsize=(13, 7))
    sub = df_summary[df_summary['block_size'] == target_bs].copy()
    sub = sub.sort_values('cross_bal_acc', ascending=True)
    y_pos = np.arange(len(sub))
    bh = 0.35

    ax.barh(y_pos + bh/2, sub['within_bal_acc'], bh,
            label='Within-fragment', color='#2ca02c', alpha=0.8)
    ax.barh(y_pos - bh/2, sub['cross_bal_acc'], bh,
            label='Cross-fragment', color='#d62728', alpha=0.8)

    labels = []
    for _, row in sub.iterrows():
        name = row['feature']
        if name in ('raw_intensity', 'gradient_magnitude', 'st_coherence', 'ALL_2D'):
            labels.append(f'{name} [2D]')
        elif name == 'ALL_COMBINED':
            labels.append(f'{name} [all]')
        else:
            labels.append(f'{name} [3D]')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Balanced Accuracy')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.legend(loc='lower right')
    ax.set_title(f'Depth vs 2D Features — Balanced Accuracy ({target_bs}x{target_bs} blocks)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'depth_vs_2d_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    # Block size comparison
    key_for_bscomp = ['raw_intensity', 'depth_variance', 'depth_range',
                      'top_bottom_ratio', 'ALL_DEPTH', 'ALL_COMBINED']
    key_for_bscomp = [f for f in key_for_bscomp if f in feature_sets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for feat in key_for_bscomp:
        vals = []
        for bs in BLOCK_SIZES:
            row = df_summary[(df_summary['block_size'] == bs) & (df_summary['feature'] == feat)]
            vals.append(row['cross_bal_acc'].values[0] if not row.empty else np.nan)
        ax.plot(BLOCK_SIZES, vals, 'o-', label=feat, markersize=5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Block size')
    ax.set_ylabel('Cross-fragment balanced accuracy')
    ax.set_title('Cross-fragment bal. acc. vs block size')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for feat in key_for_bscomp:
        vals = []
        for bs in BLOCK_SIZES:
            row = df_summary[(df_summary['block_size'] == bs) & (df_summary['feature'] == feat)]
            vals.append(row['bal_acc_gap'].values[0] if not row.empty else np.nan)
        ax.plot(BLOCK_SIZES, vals, 'o-', label=feat, markersize=5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Block size')
    ax.set_ylabel('Generalization gap')
    ax.set_title('Generalization gap vs block size')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'block_size_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    del frag_feature_maps, frag_ink_masks
    gc.collect()

    print(f"\n{'='*70}", flush=True)
    print("DONE — Phase 2B depth feature cross-validation complete.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
