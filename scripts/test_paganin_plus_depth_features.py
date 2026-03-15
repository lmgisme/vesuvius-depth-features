"""Test whether Paganin preprocessing improves depth feature cross-fragment generalization.

Architecture designed for 32 GB RAM systems:
  Phase 1: Compute and cache filtered depth features one fragment at a time.
           Peak memory: ~4 GB (accumulators + 1 layer + FFT workspace).
           Each fragment is fully freed before the next starts.
  Phase 2: Load only the tiny block-level summaries (~KB each) for cross-validation.
           Peak memory: negligible.

Outputs:
    ../results/tables/paganin_depth_feature_auc.csv
    ../results/figures/paganin_depth_feature_comparison.png
"""
import sys
import os
import gc
import time

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_A_DIR = os.path.join(SCRIPT_DIR, '..')
PROJECT_DIR = os.path.join(TRACK_A_DIR, '..')

sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))
sys.path.insert(0, TRACK_A_DIR)

from src.paganin import paganin_filter_from_params

DATA_ROOT = os.path.join(PROJECT_DIR, 'data', 'fragments')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DEPTH_FEAT_DIR = os.path.join(RESULTS_DIR, 'depth_features')

os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

# DLS scan parameters
DLS_PIXEL_SIZE_UM = 7.91
DLS_PROP_DISTANCE_M = 0.2
DLS_ENERGY_KEV = 54.0

DELTA_BETA_VALUES = [200, 5000]

FEATURE_NAMES = ['depth_gradient_mag', 'max_depth_gradient', 'depth_variance', 'depth_range']

FRAGMENTS = [
    ('frag1', 59, 'Paris 2'),
    ('frag2', 64, 'Paris 2'),
    ('frag3', 49, 'Paris 1'),
]

BLOCK_SIZE = 64
INK_THRESHOLD = 0.5
MIN_VALID_FRAC = 0.1

# --- RAM safety ---
MIN_FREE_GB = 6.0  # abort if less than this available before processing a fragment


def check_ram(context=""):
    """Check available RAM. Abort if below threshold."""
    try:
        import psutil
        m = psutil.virtual_memory()
        free_gb = m.available / 1e9
        print(f"  RAM: {free_gb:.1f} GB free / {m.total/1e9:.0f} GB total ({m.percent}% used)"
              f"  [{context}]", flush=True)
        if free_gb < MIN_FREE_GB:
            raise MemoryError(
                f"Only {free_gb:.1f} GB free (need {MIN_FREE_GB} GB). "
                f"Close other applications and retry.")
        return free_gb
    except ImportError:
        print(f"  RAM: psutil not available, skipping check [{context}]", flush=True)
        return None


def compute_features_streaming(frag_name, delta_beta):
    """Stream layers, filter, compute depth features with online algorithms.

    Peak memory: 6 accumulators (float64/32) + 2 layers (float32) + FFT workspace.
    For frag2 (14830x9506): ~4 GB total.
    """
    frag_dir = os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface', 'surface_volume')
    tif_files = sorted([f for f in os.listdir(frag_dir) if f.endswith('.tif')])
    n_layers = len(tif_files)

    first = tifffile.imread(os.path.join(frag_dir, tif_files[0]))
    H, W = first.shape
    est_gb = 6 * H * W * 8 / 1e9  # float64 accumulators
    print(f"  Volume: ({H}, {W}, {n_layers}), est peak: {est_gb:.1f} GB", flush=True)

    # Accumulators
    sum_abs_grad = np.zeros((H, W), dtype=np.float64)
    max_abs_grad = np.zeros((H, W), dtype=np.float32)
    welford_mean = np.zeros((H, W), dtype=np.float64)
    welford_m2 = np.zeros((H, W), dtype=np.float64)
    layer_max = np.full((H, W), -np.inf, dtype=np.float32)
    layer_min = np.full((H, W), np.inf, dtype=np.float32)

    prev_layer = None

    for i, fname in enumerate(tif_files):
        layer = tifffile.imread(os.path.join(frag_dir, fname)).astype(np.float32)
        if delta_beta is not None:
            layer = paganin_filter_from_params(
                layer, DLS_PIXEL_SIZE_UM, DLS_PROP_DISTANCE_M, DLS_ENERGY_KEV,
                delta_beta, handle_mask=False
            )

        np.maximum(layer_max, layer, out=layer_max)
        np.minimum(layer_min, layer, out=layer_min)

        n = i + 1
        delta = layer.astype(np.float64) - welford_mean
        welford_mean += delta / n
        delta2 = layer.astype(np.float64) - welford_mean
        welford_m2 += delta * delta2

        if prev_layer is not None:
            abs_grad = np.abs(layer - prev_layer)
            sum_abs_grad += abs_grad.astype(np.float64)
            np.maximum(max_abs_grad, abs_grad, out=max_abs_grad)
            del abs_grad

        prev_layer = layer

        if (i + 1) % 13 == 0 or i == n_layers - 1:
            print(f"    Layer {i+1}/{n_layers}", flush=True)

    n_grad = n_layers - 1
    features = {
        'depth_gradient_mag': (sum_abs_grad / n_grad).astype(np.float32),
        'max_depth_gradient': max_abs_grad,
        'depth_variance': (welford_m2 / n_layers).astype(np.float32),
        'depth_range': (layer_max - layer_min).astype(np.float32),
    }

    del sum_abs_grad, max_abs_grad, welford_mean, welford_m2, layer_max, layer_min, prev_layer
    gc.collect()

    return features


def blockify_and_save(feature_maps, ink_mask, block_size, save_path):
    """Blockify features and save as tiny .npz (just the block-level vectors).

    This reduces a ~2 GB feature map to a ~100 KB block summary.
    """
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

    # Save as npz
    np.savez_compressed(save_path, labels=block_labels, **block_features)
    n_ink = int(block_labels.sum())
    print(f"  Blockified: {len(block_labels)} blocks ({n_ink} ink) -> {save_path}", flush=True)


def load_blocks(npz_path):
    """Load blockified data from .npz file."""
    data = np.load(npz_path)
    labels = data['labels']
    features = {name: data[name] for name in FEATURE_NAMES if name in data}
    return features, labels


def main():
    print("=" * 70, flush=True)
    print("PAGANIN + DEPTH FEATURES: CROSS-FRAGMENT GENERALIZATION TEST", flush=True)
    print(f"DLS params: pixel={DLS_PIXEL_SIZE_UM}um, z={DLS_PROP_DISTANCE_M}m, "
          f"E={DLS_ENERGY_KEV}keV", flush=True)
    print(f"Block size: {BLOCK_SIZE}x{BLOCK_SIZE}", flush=True)
    print(f"d/b values: {DELTA_BETA_VALUES}", flush=True)
    print("=" * 70, flush=True)

    check_ram("startup")

    frag_names = [f[0] for f in FRAGMENTS]
    blocks_dir = os.path.join(DEPTH_FEAT_DIR, 'blocks')
    os.makedirs(blocks_dir, exist_ok=True)

    # ================================================================
    # PHASE 1: Compute features and blockify, one fragment at a time
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print("PHASE 1: Compute and blockify features (one fragment at a time)", flush=True)
    print(f"{'='*70}", flush=True)

    conditions = ['raw'] + [f'paganin_db{db}' for db in DELTA_BETA_VALUES]

    for frag_name, _, scroll in FRAGMENTS:
        print(f"\n--- {frag_name} ({scroll}) ---", flush=True)
        check_ram(f"before {frag_name}")

        ink_mask = Image.open(
            os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface', 'inklabels.png'))
        ink_mask = np.array(ink_mask)
        if ink_mask.ndim == 3:
            ink_mask = ink_mask[:, :, 0]
        ink_mask = ink_mask > 0

        # --- Raw features (from cache) ---
        blocks_path = os.path.join(blocks_dir, f'{frag_name}_raw_b{BLOCK_SIZE}.npz')
        if not os.path.exists(blocks_path):
            print(f"  Loading raw cached features...", flush=True)
            feat_dir = os.path.join(DEPTH_FEAT_DIR, frag_name)
            raw_feats = {}
            for fn in FEATURE_NAMES:
                raw_feats[fn] = np.load(os.path.join(feat_dir, f'{fn}.npy'))
            blockify_and_save(raw_feats, ink_mask, BLOCK_SIZE, blocks_path)
            del raw_feats
            gc.collect()
        else:
            print(f"  Raw blocks cached: {blocks_path}", flush=True)

        # --- Filtered features for each d/b ---
        for db in DELTA_BETA_VALUES:
            blocks_path = os.path.join(blocks_dir, f'{frag_name}_paganin_db{db}_b{BLOCK_SIZE}.npz')
            if os.path.exists(blocks_path):
                print(f"  d/b={db} blocks cached: {blocks_path}", flush=True)
                continue

            # Check if full-res features are cached
            cache_dir = os.path.join(DEPTH_FEAT_DIR, f'{frag_name}_paganin_db{db}')
            if os.path.exists(cache_dir) and all(
                os.path.exists(os.path.join(cache_dir, f'{fn}.npy'))
                for fn in FEATURE_NAMES
            ):
                print(f"  Loading cached d/b={db} features...", flush=True)
                feats = {}
                for fn in FEATURE_NAMES:
                    feats[fn] = np.load(os.path.join(cache_dir, f'{fn}.npy'))
                blockify_and_save(feats, ink_mask, BLOCK_SIZE, blocks_path)
                del feats
                gc.collect()
                continue

            # Compute from scratch (streaming)
            check_ram(f"before {frag_name} d/b={db}")
            print(f"  Computing d/b={db} (streaming)...", flush=True)
            t0 = time.time()
            feats = compute_features_streaming(frag_name, db)
            print(f"  Done in {time.time()-t0:.0f}s", flush=True)

            # Cache full-res features
            os.makedirs(cache_dir, exist_ok=True)
            for fn, arr in feats.items():
                np.save(os.path.join(cache_dir, f'{fn}.npy'), arr)
            print(f"  Cached full-res to {cache_dir}", flush=True)

            # Blockify and save
            blockify_and_save(feats, ink_mask, BLOCK_SIZE, blocks_path)

            del feats
            gc.collect()

        del ink_mask
        gc.collect()
        check_ram(f"after {frag_name}")

    # ================================================================
    # PHASE 2: Cross-validation using only block-level data (~KB)
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2: Cross-fragment evaluation (block-level data only)", flush=True)
    print(f"{'='*70}", flush=True)

    check_ram("phase 2 start")

    # Load all block summaries (tiny — a few KB each)
    all_blocks = {}  # {condition: {frag: {features, y}}}
    for cond in conditions:
        all_blocks[cond] = {}
        for frag_name in frag_names:
            npz_path = os.path.join(blocks_dir, f'{frag_name}_{cond}_b{BLOCK_SIZE}.npz')
            features, labels = load_blocks(npz_path)
            all_blocks[cond][frag_name] = {'features': features, 'y': labels}

    # Cross-validation
    scroll_map = {f[0]: f[2] for f in FRAGMENTS}
    all_results = []

    for cond in conditions:
        for feat_name in FEATURE_NAMES:
            for train_frag in frag_names:
                X_train = all_blocks[cond][train_frag]['features'][feat_name].reshape(-1, 1)
                y_train = all_blocks[cond][train_frag]['y']

                valid = np.isfinite(X_train.ravel())
                X_train = X_train[valid]
                y_train = y_train[valid]

                if len(np.unique(y_train)) < 2:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_train_s, y_train)

                for test_frag in frag_names:
                    X_test = all_blocks[cond][test_frag]['features'][feat_name].reshape(-1, 1)
                    y_test = all_blocks[cond][test_frag]['y']

                    valid_t = np.isfinite(X_test.ravel())
                    X_test = X_test[valid_t]
                    y_test = y_test[valid_t]

                    if len(np.unique(y_test)) < 2:
                        continue

                    X_test_s = scaler.transform(X_test)
                    y_prob = clf.predict_proba(X_test_s)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                    bal_acc = balanced_accuracy_score(y_test, clf.predict(X_test_s))

                    all_results.append({
                        'condition': cond,
                        'feature': feat_name,
                        'train_fragment': train_frag,
                        'test_fragment': test_frag,
                        'auc': round(auc, 4),
                        'balanced_accuracy': round(bal_acc, 4),
                        'same_fragment': train_frag == test_frag,
                        'same_scroll': scroll_map[train_frag] == scroll_map[test_frag],
                    })

    df = pd.DataFrame(all_results)

    # Summary
    summary_rows = []
    for cond in conditions:
        for feat_name in FEATURE_NAMES:
            sub = df[(df['condition'] == cond) & (df['feature'] == feat_name)]
            if sub.empty:
                continue
            summary_rows.append({
                'condition': cond,
                'feature': feat_name,
                'within_auc': round(sub[sub['same_fragment']]['auc'].mean(), 4),
                'cross_auc': round(sub[~sub['same_fragment']]['auc'].mean(), 4),
                'auc_gap': round(sub[sub['same_fragment']]['auc'].mean() -
                                 sub[~sub['same_fragment']]['auc'].mean(), 4),
                'within_bal_acc': round(sub[sub['same_fragment']]['balanced_accuracy'].mean(), 4),
                'cross_bal_acc': round(sub[~sub['same_fragment']]['balanced_accuracy'].mean(), 4),
            })

    df_summary = pd.DataFrame(summary_rows)

    # Save
    detail_path = os.path.join(RESULTS_DIR, 'tables', 'paganin_depth_feature_auc.csv')
    df.to_csv(detail_path, index=False)
    print(f"\nSaved: {detail_path}", flush=True)

    summary_path = os.path.join(RESULTS_DIR, 'tables', 'paganin_depth_feature_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}", flush=True)

    # ---- Print results ----
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS: Paganin + Depth Features @ {BLOCK_SIZE}x{BLOCK_SIZE} blocks", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Feature':<25} {'Raw xAUC':>10} {'db=200 xAUC':>12} "
          f"{'db=5000 xAUC':>13} {'d200':>6} {'d5000':>7}", flush=True)
    print(f"{'='*25} {'='*10} {'='*12} {'='*13} {'='*6} {'='*7}", flush=True)

    for feat_name in FEATURE_NAMES:
        vals = {}
        for cond in conditions:
            row = df_summary[(df_summary['condition'] == cond) &
                             (df_summary['feature'] == feat_name)]
            vals[cond] = row['cross_auc'].values[0] if not row.empty else float('nan')

        raw = vals['raw']
        d200 = vals['paganin_db200']
        d5000 = vals['paganin_db5000']
        print(f"{feat_name:<25} {raw:>10.4f} {d200:>12.4f} "
              f"{d5000:>13.4f} {d200-raw:>+6.3f} {d5000-raw:>+7.3f}", flush=True)

    # ---- Figure ----
    feat_list = list(FEATURE_NAMES)
    x = np.arange(len(feat_list))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    raw_aucs = [df_summary[(df_summary['condition'] == 'raw') &
                            (df_summary['feature'] == f)]['cross_auc'].values[0]
                for f in feat_list]
    db200_aucs = [df_summary[(df_summary['condition'] == 'paganin_db200') &
                              (df_summary['feature'] == f)]['cross_auc'].values[0]
                  for f in feat_list]
    db5000_aucs = [df_summary[(df_summary['condition'] == 'paganin_db5000') &
                               (df_summary['feature'] == f)]['cross_auc'].values[0]
                   for f in feat_list]

    ax = axes[0]
    ax.bar(x - width, raw_aucs, width, label='Raw', color='#2196F3', alpha=0.8)
    ax.bar(x, db200_aucs, width, label='Paganin d/b=200', color='#FF9800', alpha=0.8)
    ax.bar(x + width, db5000_aucs, width, label='Paganin d/b=5000', color='#F44336', alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel('Cross-fragment AUC')
    ax.set_title('Cross-Fragment AUC')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in feat_list], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.45, max(max(raw_aucs), max(db200_aucs), max(db5000_aucs)) + 0.05)

    ax = axes[1]
    d200_ch = [db200_aucs[i] - raw_aucs[i] for i in range(len(feat_list))]
    d5000_ch = [db5000_aucs[i] - raw_aucs[i] for i in range(len(feat_list))]
    ax.bar(x - width/2, d200_ch, width, label='d/b=200', color='#FF9800', alpha=0.8)
    ax.bar(x + width/2, d5000_ch, width, label='d/b=5000', color='#F44336', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('AUC Change from Raw')
    ax.set_title('Effect of Paganin Filtering')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in feat_list], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Paganin + Depth Features ({BLOCK_SIZE}x{BLOCK_SIZE} blocks)\n'
                 f'DLS: {DLS_PIXEL_SIZE_UM}um, z={DLS_PROP_DISTANCE_M}m, E={DLS_ENERGY_KEV}keV',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(RESULTS_DIR, 'figures', 'paganin_depth_feature_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
