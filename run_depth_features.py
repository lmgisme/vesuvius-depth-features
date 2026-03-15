"""
Compute 3D depth features for each fragment and evaluate pixel-level AUC.

Memory-efficient: loads volume as uint16 (~18GB for frag2) and converts
chunks to float32 for feature computation. For very large fragments,
uses chunked processing.

Usage:
    conda activate vesuvius
    python -u scripts/run_depth_features.py
"""
import sys, os, gc, glob, time
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vesuvius_preprocess.depth_features import compute_all_depth_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', 'data', 'fragments')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

FRAGMENTS = [
    ('frag1', 59, 'Paris 2'),
    ('frag2', 64, 'Paris 2'),
    ('frag3', 49, 'Paris 1'),
]


def load_volume_uint16(frag_name):
    """Load volume as uint16 to save memory. Returns (H,W,D) uint16 and ink mask."""
    frag_dir = os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface')
    ink = np.array(Image.open(os.path.join(frag_dir, 'inklabels.png'))) > 0

    sv_dir = os.path.join(frag_dir, 'surface_volume')
    tif_files = sorted(glob.glob(os.path.join(sv_dir, '*.tif')))
    D = len(tif_files)
    first = tifffile.imread(tif_files[0])
    H, W = first.shape[:2]

    print(f"  Loading {D} layers as uint16 ({H}x{W})...", flush=True)
    vol = np.empty((H, W, D), dtype=np.uint16)
    for i, f in enumerate(tif_files):
        vol[:, :, i] = tifffile.imread(f)
        if (i + 1) % 10 == 0:
            print(f"    Layer {i+1}/{D}", flush=True)

    print(f"  Volume: {vol.shape}, {vol.nbytes / 1e9:.1f} GB (uint16)", flush=True)
    return vol, ink


def compute_auc(feature_map, ink_mask):
    """AUC for a feature separating ink from non-ink."""
    y = ink_mask.ravel().astype(np.int32)
    s = feature_map.ravel().astype(np.float64)
    valid = np.isfinite(s)
    if valid.sum() < 100:
        return np.nan, '?'
    y, s = y[valid], s[valid]
    auc = roc_auc_score(y, s)
    if auc >= 0.5:
        return auc, '+'
    else:
        return 1.0 - auc, '-'


def plot_feature_diagnostics(features, ink, frag_name, save_dir):
    """Plot each depth feature: feature map + ink/noink histograms."""
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for idx, (feat_name, feat_map) in enumerate(features.items()):
        ax = axes[idx, 0]
        valid = np.isfinite(feat_map)
        vmin, vmax = np.percentile(feat_map[valid], [2, 98]) if valid.any() else (0, 1)
        ax.imshow(feat_map, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(feat_name, fontsize=9)
        ax.axis('off')

        ax = axes[idx, 1]
        # Subsample for histogram to avoid memory issues
        n_pix = ink.size
        step = max(1, n_pix // 500000)
        ink_flat = ink.ravel()[::step]
        feat_flat = feat_map.ravel()[::step]
        valid_flat = np.isfinite(feat_flat)
        ink_vals = feat_flat[ink_flat & valid_flat]
        noink_vals = feat_flat[~ink_flat & valid_flat]
        if len(ink_vals) > 0:
            ax.hist(ink_vals, bins=100, alpha=0.5, density=True, label='Ink', color='red')
        if len(noink_vals) > 0:
            ax.hist(noink_vals, bins=100, alpha=0.5, density=True, label='Non-ink', color='blue')
        ax.set_title(f'{feat_name} distribution', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Depth Features — {frag_name}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f'depth_features_diagnostic_{frag_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

    all_auc_rows = []

    for frag_name, peak_layer, scroll in FRAGMENTS:
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {frag_name} ({scroll})", flush=True)
        print(f"{'='*60}", flush=True)

        # Check if features already computed
        feat_dir = os.path.join(RESULTS_DIR, 'depth_features', frag_name)
        expected_features = ['depth_gradient_mag', 'max_depth_gradient', 'depth_variance',
                             'depth_range', 'intensity_centroid', 'peak_depth_layer',
                             'depth_skewness', 'depth_kurtosis', 'top_bottom_ratio',
                             'adj_layer_corr']

        all_cached = all(
            os.path.exists(os.path.join(feat_dir, f'{fn}.npy'))
            for fn in expected_features
        )

        if all_cached:
            print(f"  Loading cached features from {feat_dir}", flush=True)
            features = {}
            for fn in expected_features:
                features[fn] = np.load(os.path.join(feat_dir, f'{fn}.npy'))
            ink = np.array(Image.open(
                os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface', 'inklabels.png')
            )) > 0
        else:
            t0 = time.time()
            vol, ink = load_volume_uint16(frag_name)
            n_ink = int(ink.sum())
            print(f"  Ink: {n_ink:,}/{ink.size:,} ({100*n_ink/ink.size:.1f}%)", flush=True)
            print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

            # Pass uint16 directly — chunked functions convert to float32 per-chunk
            print("  Computing depth features (chunked, uint16 input)...", flush=True)
            t0 = time.time()
            features = compute_all_depth_features(vol, adj_corr_patch_size=16)
            print(f"  All depth features computed in {time.time()-t0:.1f}s", flush=True)

            del vol
            gc.collect()

            # Save feature maps
            os.makedirs(feat_dir, exist_ok=True)
            for feat_name_key, feat_map in features.items():
                np.save(os.path.join(feat_dir, f'{feat_name_key}.npy'), feat_map)
            print(f"  Saved {len(features)} feature maps to {feat_dir}", flush=True)

        # Load peak layer for raw intensity baseline
        sv_dir = os.path.join(DATA_ROOT, frag_name, '54keV_exposed_surface', 'surface_volume')
        tif_files = sorted(glob.glob(os.path.join(sv_dir, '*.tif')))
        raw_peak = tifffile.imread(tif_files[peak_layer]).astype(np.float32)

        # Pixel-level AUC
        print("  Computing pixel-level AUCs...", flush=True)
        auc_raw, pol_raw = compute_auc(raw_peak, ink)
        all_auc_rows.append({
            'fragment': frag_name, 'scroll': scroll,
            'feature': f'raw_intensity_layer{peak_layer}',
            'pixel_auc': round(auc_raw, 4), 'polarity': pol_raw,
            'feature_type': '2D_baseline',
        })
        print(f"    raw_intensity_layer{peak_layer}: AUC={auc_raw:.4f} ({pol_raw})", flush=True)

        for feat_name_key, feat_map in features.items():
            auc, pol = compute_auc(feat_map, ink)
            all_auc_rows.append({
                'fragment': frag_name, 'scroll': scroll,
                'feature': feat_name_key,
                'pixel_auc': round(auc, 4), 'polarity': pol,
                'feature_type': '3D_depth',
            })
            print(f"    {feat_name_key}: AUC={auc:.4f} ({pol})", flush=True)

        # Diagnostic plots
        plot_feature_diagnostics(features, ink, frag_name,
                                 os.path.join(RESULTS_DIR, 'figures'))

        del features, ink, raw_peak
        gc.collect()
        print(f"  Memory freed.", flush=True)

    # Summary
    df = pd.DataFrame(all_auc_rows)
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'depth_feature_pixel_auc.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DEPTH FEATURE PIXEL-LEVEL AUC SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    pivot = df.pivot_table(index='feature', columns='fragment', values='pixel_auc')
    pivot['mean_auc'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_auc', ascending=False)
    print(pivot.to_string(), flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
