"""
Phase 2: Cross-fragment generalization matrix (corrected methodology).

Fixes vs. prior version:
  1. Spatially-blocked sampling: tiles each fragment into non-overlapping NxN
     blocks, computes block-level mean features and majority-vote ink labels.
     This eliminates spatial autocorrelation inflation.
  2. Threshold-dependent metrics: uses balanced accuracy (primary) and log-loss
     in addition to AUC. For single-feature classifiers, AUC is rank-invariant
     and therefore train-invariant — balanced accuracy captures whether the
     learned decision boundary actually transfers.

For each feature (and an ALL_COMBINED multi-feature model), trains logistic
regression on one fragment's blocks, tests on all fragments' blocks.

Usage:
    conda activate vesuvius
    python -u scripts/run_cross_validation.py
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
from vesuvius_preprocess.texture_enhance import compute_all_texture_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', 'data', 'fragments')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

# Fragment configs: name, peak layer from depth profile analysis, scroll
FRAGMENTS = [
    ('frag1', 59, 'Paris 2'),
    ('frag2', 64, 'Paris 2'),
    ('frag3', 49, 'Paris 1'),
]

BLOCK_SIZE = 64  # pixels per block side
INK_THRESHOLD = 0.5  # fraction of ink pixels to label a block as ink
MIN_VALID_FRAC = 0.1  # discard blocks with <10% non-zero pixels (background)


def load_layer_and_ink(frag_name, layer_idx):
    """Load a single depth layer and ink mask for a fragment."""
    frag_dir = os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface')
    ink_path = os.path.join(frag_dir, 'inklabels.png')
    ink = np.array(Image.open(ink_path)) > 0

    sv_dir = os.path.join(frag_dir, 'surface_volume')
    tif_files = sorted(glob.glob(os.path.join(sv_dir, '*.tif')))
    layer = tifffile.imread(tif_files[layer_idx]).astype(np.float32)

    return layer, ink


def blockify(feature_maps, ink_mask, block_size=BLOCK_SIZE):
    """Tile feature maps and ink mask into non-overlapping blocks.

    Parameters
    ----------
    feature_maps : dict of str -> np.ndarray (H, W)
    ink_mask : np.ndarray bool (H, W)
    block_size : int

    Returns
    -------
    block_features : dict of str -> np.ndarray (n_blocks,)
        Mean feature value per block.
    block_labels : np.ndarray (n_blocks,)
        1 if majority ink, 0 otherwise.
    n_ink_blocks, n_noink_blocks : int
    """
    H, W = ink_mask.shape
    n_rows = H // block_size
    n_cols = W // block_size

    # Compute block-level ink fraction and validity (non-background)
    # Use the first feature map to detect background (zero pixels)
    first_feat = next(iter(feature_maps.values()))

    block_ink_frac = np.zeros((n_rows, n_cols))
    block_valid_frac = np.zeros((n_rows, n_cols))
    block_feat_means = {name: np.zeros((n_rows, n_cols)) for name in feature_maps}

    for r in range(n_rows):
        r0, r1 = r * block_size, (r + 1) * block_size
        for c in range(n_cols):
            c0, c1 = c * block_size, (c + 1) * block_size
            ink_patch = ink_mask[r0:r1, c0:c1]
            feat_patch = first_feat[r0:r1, c0:c1]

            n_pixels = block_size * block_size
            n_nonzero = np.count_nonzero(feat_patch)
            block_valid_frac[r, c] = n_nonzero / n_pixels
            block_ink_frac[r, c] = ink_patch.sum() / n_pixels

            for name, fmap in feature_maps.items():
                patch = fmap[r0:r1, c0:c1]
                block_feat_means[name][r, c] = np.mean(patch)

    # Filter: keep blocks with enough valid pixels
    valid = block_valid_frac >= MIN_VALID_FRAC

    # Assign labels: ink if >50% ink pixels in block
    labels_2d = (block_ink_frac > INK_THRESHOLD).astype(np.float64)

    # Flatten and filter
    mask = valid.ravel()
    block_labels = labels_2d.ravel()[mask]
    block_features = {}
    for name in feature_maps:
        block_features[name] = block_feat_means[name].ravel()[mask].astype(np.float32)

    n_ink = int(block_labels.sum())
    n_noink = int((block_labels == 0).sum())

    return block_features, block_labels, n_ink, n_noink


def evaluate_model(clf, scaler, X_test, y_test):
    """Compute AUC, balanced accuracy, and log-loss for a fitted model."""
    X_scaled = scaler.transform(X_test)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    y_pred = clf.predict(X_scaled)

    auc = roc_auc_score(y_test, y_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)

    return auc, bal_acc, ll


def print_matrix(matrix, frag_names, metric_name):
    """Pretty-print a 3x3 matrix."""
    print(f"    [{metric_name}]", flush=True)
    print(f"    Train\\Test  ", end="", flush=True)
    for fn in frag_names:
        print(f"  {fn:>6}", end="")
    print(flush=True)
    for i, train_frag in enumerate(frag_names):
        print(f"    {train_frag:>10}  ", end="")
        for j in range(len(frag_names)):
            print(f"  {matrix[i,j]:.4f}", end="")
        print(flush=True)


def compute_summary(matrices_auc, matrices_balacc, feature_sets, frag_names):
    """Compute summary metrics for each feature set."""
    scroll_map = {f[0]: f[2] for f in FRAGMENTS}
    rows = []

    for feat_name in feature_sets.keys():
        m_auc = matrices_auc[feat_name]
        m_ba = matrices_balacc[feat_name]
        n = len(frag_names)

        def _stats(m):
            within = np.mean(np.diag(m))
            off = [m[i,j] for i in range(n) for j in range(n) if i != j]
            cross = np.mean(off)
            same_scroll = np.mean([m[0,1], m[1,0]])
            cross_scroll = np.mean([m[0,2], m[1,2], m[2,0], m[2,1]])
            gap = within - cross
            return within, cross, same_scroll, cross_scroll, gap

        w_auc, c_auc, ss_auc, cs_auc, g_auc = _stats(m_auc)
        w_ba, c_ba, ss_ba, cs_ba, g_ba = _stats(m_ba)

        rows.append({
            'feature': feat_name,
            'primary_metric': 'balanced_accuracy',
            'within_bal_acc': round(w_ba, 4),
            'cross_bal_acc': round(c_ba, 4),
            'same_scroll_bal_acc': round(ss_ba, 4),
            'cross_scroll_bal_acc': round(cs_ba, 4),
            'bal_acc_gap': round(g_ba, 4),
            'within_auc': round(w_auc, 4),
            'cross_auc': round(c_auc, 4),
            'same_scroll_auc': round(ss_auc, 4),
            'cross_scroll_auc': round(cs_auc, 4),
            'auc_gap': round(g_auc, 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('cross_bal_acc', ascending=False)
    return df


def main():
    print("="*70, flush=True)
    print("PHASE 2: CROSS-FRAGMENT GENERALIZATION (CORRECTED)", flush=True)
    print("Block-based sampling (64x64), balanced accuracy as primary metric", flush=True)
    print("="*70, flush=True)

    # Step 1: Load, compute features, blockify
    frag_data = {}  # frag_name -> {'features': {name: array}, 'y': array}

    for frag_name, peak_layer, scroll in FRAGMENTS:
        print(f"\n--- {frag_name} (peak layer {peak_layer}, {scroll}) ---", flush=True)

        t0 = time.time()
        layer, ink = load_layer_and_ink(frag_name, peak_layer)
        print(f"  Loaded in {time.time()-t0:.1f}s, shape={layer.shape}, "
              f"ink pixels={ink.sum():,}/{ink.size:,} ({100*ink.sum()/ink.size:.1f}%)", flush=True)

        print(f"  Computing texture features...", flush=True)
        t0 = time.time()
        features = compute_all_texture_features(layer, sigma=1.0, variance_window=11)
        features['raw_intensity'] = layer
        print(f"    Done in {time.time()-t0:.1f}s, {len(features)} features", flush=True)

        print(f"  Blockifying ({BLOCK_SIZE}x{BLOCK_SIZE} blocks)...", flush=True)
        t0 = time.time()
        block_features, block_labels, n_ink, n_noink = blockify(features, ink)
        n_total = n_ink + n_noink
        print(f"    {n_total} blocks ({n_ink} ink, {n_noink} non-ink) in {time.time()-t0:.1f}s", flush=True)

        frag_data[frag_name] = {'features': block_features, 'y': block_labels}

        del features, ink, layer
        gc.collect()

    feature_names = list(frag_data[FRAGMENTS[0][0]]['features'].keys())
    frag_names = [f[0] for f in FRAGMENTS]

    # Feature sets: each single feature + all combined
    feature_sets = {name: [name] for name in feature_names}
    feature_sets['ALL_COMBINED'] = feature_names

    # Step 2: Cross-fragment classification
    print(f"\n{'='*70}", flush=True)
    print("Training cross-fragment classifiers...", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = []
    matrices_auc = {}
    matrices_balacc = {}
    matrices_logloss = {}
    scroll_map = {f[0]: f[2] for f in FRAGMENTS}

    for set_name, feat_list in feature_sets.items():
        print(f"\n  Feature set: {set_name} ({len(feat_list)} feat)", flush=True)
        m_auc = np.zeros((len(frag_names), len(frag_names)))
        m_ba = np.zeros((len(frag_names), len(frag_names)))
        m_ll = np.zeros((len(frag_names), len(frag_names)))

        for i, train_frag in enumerate(frag_names):
            fd = frag_data[train_frag]
            X_train = np.column_stack([fd['features'][f] for f in feat_list])
            y_train = fd['y']

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train_scaled, y_train)

            for j, test_frag in enumerate(frag_names):
                fd_test = frag_data[test_frag]
                X_test = np.column_stack([fd_test['features'][f] for f in feat_list])
                y_test = fd_test['y']

                auc, bal_acc, ll = evaluate_model(clf, scaler, X_test, y_test)
                m_auc[i, j] = auc
                m_ba[i, j] = bal_acc
                m_ll[i, j] = ll

                all_results.append({
                    'feature': set_name,
                    'train_fragment': train_frag,
                    'test_fragment': test_frag,
                    'balanced_accuracy': round(bal_acc, 4),
                    'auc': round(auc, 4),
                    'log_loss': round(ll, 4),
                    'same_fragment': train_frag == test_frag,
                    'same_scroll': scroll_map[train_frag] == scroll_map[test_frag],
                })

        matrices_auc[set_name] = m_auc
        matrices_balacc[set_name] = m_ba
        matrices_logloss[set_name] = m_ll

        print_matrix(m_ba, frag_names, 'Balanced Accuracy')
        print_matrix(m_auc, frag_names, 'AUC')

    # Step 3: Summary
    print(f"\n{'='*70}", flush=True)
    print("GENERALIZATION SUMMARY (sorted by cross-fragment balanced accuracy)", flush=True)
    print("Primary metric: balanced_accuracy (threshold-dependent)", flush=True)
    print(f"{'='*70}", flush=True)

    df_summary = compute_summary(matrices_auc, matrices_balacc, feature_sets, frag_names)

    print(df_summary[['feature', 'within_bal_acc', 'cross_bal_acc',
                       'cross_scroll_bal_acc', 'bal_acc_gap',
                       'within_auc', 'cross_auc', 'auc_gap']].to_string(index=False))

    # Save tables
    df_detail = pd.DataFrame(all_results)
    detail_path = os.path.join(RESULTS_DIR, 'tables', 'cross_fragment_auc_detail.csv')
    df_detail.to_csv(detail_path, index=False)
    print(f"\nSaved: {detail_path}", flush=True)

    summary_path = os.path.join(RESULTS_DIR, 'tables', 'cross_fragment_auc_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}", flush=True)

    # Step 4: Visualizations

    # 4a: Balanced accuracy heatmaps
    all_set_names = list(feature_sets.keys())
    n_sets = len(all_set_names)
    ncols = 3
    nrows = (n_sets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    axes = axes.flatten()

    for idx, feat_name in enumerate(all_set_names):
        ax = axes[idx]
        m = matrices_balacc[feat_name]
        sns.heatmap(m, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=frag_names, yticklabels=frag_names,
                    vmin=0.4, vmax=0.85, ax=ax, cbar=False,
                    annot_kws={'fontsize': 10})
        ax.set_title(feat_name, fontsize=9)
        ax.set_ylabel('Train' if idx % ncols == 0 else '')
        ax.set_xlabel('Test')

    for j in range(n_sets, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Cross-Fragment Balanced Accuracy (Train=row, Test=col)\n'
                 f'Block-level ({BLOCK_SIZE}x{BLOCK_SIZE}), threshold-dependent', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'cross_fragment_balacc_heatmaps.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    # 4b: AUC heatmaps (secondary)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    axes = axes.flatten()

    for idx, feat_name in enumerate(all_set_names):
        ax = axes[idx]
        m = matrices_auc[feat_name]
        sns.heatmap(m, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=frag_names, yticklabels=frag_names,
                    vmin=0.4, vmax=0.85, ax=ax, cbar=False,
                    annot_kws={'fontsize': 10})
        ax.set_title(feat_name, fontsize=9)
        ax.set_ylabel('Train' if idx % ncols == 0 else '')
        ax.set_xlabel('Test')

    for j in range(n_sets, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Cross-Fragment AUC (Train=row, Test=col)\n'
                 f'Block-level ({BLOCK_SIZE}x{BLOCK_SIZE}), note: single-feature AUC is train-invariant',
                 fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'cross_fragment_auc_heatmaps.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    # 4c: Within vs cross balanced accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_summary.sort_values('cross_bal_acc', ascending=True)
    y_pos = np.arange(len(df_plot))
    bar_height = 0.35

    ax.barh(y_pos + bar_height/2, df_plot['within_bal_acc'], bar_height,
            label='Within-fragment', color='#2ca02c', alpha=0.8)
    ax.barh(y_pos - bar_height/2, df_plot['cross_bal_acc'], bar_height,
            label='Cross-fragment', color='#d62728', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'])
    ax.set_xlabel('Balanced Accuracy')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.legend(loc='lower right')
    ax.set_title(f'Feature Generalization: Balanced Accuracy (block-level, {BLOCK_SIZE}x{BLOCK_SIZE})')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'cross_fragment_generalization_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    # 4d: Generalization gap (balanced accuracy)
    fig, ax = plt.subplots(figsize=(10, 5))
    df_gap = df_summary.sort_values('bal_acc_gap', ascending=True)
    colors = ['#2ca02c' if abs(g) < 0.03 else '#ff7f0e' if abs(g) < 0.08 else '#d62728'
              for g in df_gap['bal_acc_gap']]
    ax.barh(df_gap['feature'], df_gap['bal_acc_gap'], color=colors)
    ax.set_xlabel('Bal. Accuracy Gap (within - cross, smaller = better transfer)')
    ax.set_title(f'Generalization Gap — Balanced Accuracy ({BLOCK_SIZE}x{BLOCK_SIZE} blocks)')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'generalization_gap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE — Phase 2 cross-fragment generalization (corrected).", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
