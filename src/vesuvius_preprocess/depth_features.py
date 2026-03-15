"""
3D depth features for ink detection.

Computes per-pixel features derived from the full depth stack (65 layers)
at each spatial location. All operations are chunked along the row axis
to stay within ~4 GB working memory per feature, allowing processing of
large fragments (e.g., frag2 at 14830 x 9506 x 65).
"""

import numpy as np
from scipy.stats import skew, kurtosis

# Default chunk size: process this many rows at a time.
# For a 9506-wide fragment with 65 layers, 500 rows = ~1.2 GB per chunk.
CHUNK_ROWS = 500


def _chunked_reduce(volume, func, chunk_rows=CHUNK_ROWS):
    """Apply func(chunk_float32) -> (chunk_H, W) to volume in row chunks.

    Converts each chunk to float32 on the fly if the volume is uint16,
    so the full float32 volume never needs to be in memory.
    """
    H, W = volume.shape[:2]
    result = np.empty((H, W), dtype=np.float32)
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        chunk = volume[r0:r1]
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        result[r0:r1] = func(chunk)
    return result


def compute_depth_gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    """Mean |dI/dz| across all depth layers at each (x,y)."""
    def _func(chunk):
        # Simple finite difference: diff along axis 2
        diff = np.diff(chunk, axis=2)
        return np.mean(np.abs(diff), axis=2).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_max_depth_gradient(volume: np.ndarray) -> np.ndarray:
    """Maximum |dI/dz| across depth layers at each (x,y)."""
    def _func(chunk):
        diff = np.diff(chunk, axis=2)
        return np.max(np.abs(diff), axis=2).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_depth_variance(volume: np.ndarray) -> np.ndarray:
    """Variance of intensity across depth layers at each (x,y)."""
    def _func(chunk):
        return np.var(chunk, axis=2).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_depth_range(volume: np.ndarray) -> np.ndarray:
    """Max - min intensity across depth layers at each (x,y)."""
    def _func(chunk):
        return (np.max(chunk, axis=2) - np.min(chunk, axis=2)).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_intensity_centroid(volume: np.ndarray) -> np.ndarray:
    """Weighted average depth index, weighted by intensity."""
    D = volume.shape[2]
    layer_indices = np.arange(D, dtype=np.float32).reshape(1, 1, D)

    def _func(chunk):
        s = np.sum(chunk, axis=2)
        s = np.maximum(s, 1e-8)
        return (np.sum(chunk * layer_indices, axis=2) / s).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_peak_depth_layer(volume: np.ndarray) -> np.ndarray:
    """Argmax of intensity across depth layers at each (x,y)."""
    def _func(chunk):
        return np.argmax(chunk, axis=2).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_depth_skewness(volume: np.ndarray) -> np.ndarray:
    """Skewness of the depth profile at each (x,y)."""
    def _func(chunk):
        return skew(chunk, axis=2, nan_policy='omit').astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_depth_kurtosis(volume: np.ndarray) -> np.ndarray:
    """Kurtosis (excess) of the depth profile at each (x,y)."""
    def _func(chunk):
        return kurtosis(chunk, axis=2, nan_policy='omit').astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_top_bottom_ratio(volume: np.ndarray) -> np.ndarray:
    """Mean intensity of layers 0-32 / mean intensity of layers 33-64."""
    D = volume.shape[2]
    mid = D // 2

    def _func(chunk):
        top = np.mean(chunk[:, :, :mid], axis=2)
        bot = np.mean(chunk[:, :, mid:], axis=2)
        bot = np.maximum(bot, 1e-8)
        return (top / bot).astype(np.float32)
    return _chunked_reduce(volume, _func)


def compute_adjacent_layer_correlation(
    volume: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray:
    """Mean Pearson correlation between adjacent layers in local patches.

    Computed at patch level, one row of patches at a time. Result upsampled to (H, W).
    """
    H, W, D = volume.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    W_trim = n_cols * patch_size
    result_small = np.zeros((n_rows, n_cols), dtype=np.float32)

    for r in range(n_rows):
        r0 = r * patch_size
        r1 = r0 + patch_size
        row_strip = volume[r0:r1, :W_trim, :]
        if row_strip.dtype != np.float32:
            row_strip = row_strip.astype(np.float32)
        row_patches = row_strip.reshape(patch_size, n_cols, patch_size, D)
        row_patches = row_patches.transpose(1, 0, 2, 3).reshape(n_cols, patch_size * patch_size, D)

        corr_sum = np.zeros(n_cols, dtype=np.float64)
        for z in range(D - 1):
            a = row_patches[:, :, z]
            b = row_patches[:, :, z + 1]
            mean_a = a.mean(axis=1)
            mean_b = b.mean(axis=1)
            std_a = a.std(axis=1)
            std_b = b.std(axis=1)
            cov = (a * b).mean(axis=1) - mean_a * mean_b
            denom = std_a * std_b
            valid = denom > 1e-8
            corr_sum += np.where(valid, cov / denom, 0.0)

        result_small[r, :] = (corr_sum / (D - 1)).astype(np.float32)

    # Upsample to full resolution, pad to original size
    upsampled = np.repeat(np.repeat(result_small, patch_size, axis=0), patch_size, axis=1)
    result = np.zeros((H, W), dtype=np.float32)
    h_up, w_up = upsampled.shape
    result[:h_up, :w_up] = upsampled
    if h_up < H:
        result[h_up:, :w_up] = result[h_up - 1:h_up, :w_up]
    if w_up < W:
        result[:, w_up:] = result[:, w_up - 1:w_up]
    return result


def compute_all_depth_features(
    volume: np.ndarray,
    adj_corr_patch_size: int = 16,
) -> dict:
    """Compute all depth features for a 3D surface volume.

    All operations are chunked to limit memory. The full volume must be
    in memory but intermediate arrays are small.

    Parameters
    ----------
    volume : np.ndarray, shape (H, W, D), float32
    adj_corr_patch_size : int
        Patch size for adjacent-layer correlation.

    Returns
    -------
    dict mapping feature name -> np.ndarray (H, W)
    """
    features = {}

    print("    depth_gradient_magnitude...", flush=True)
    features['depth_gradient_mag'] = compute_depth_gradient_magnitude(volume)

    print("    max_depth_gradient...", flush=True)
    features['max_depth_gradient'] = compute_max_depth_gradient(volume)

    print("    depth_variance...", flush=True)
    features['depth_variance'] = compute_depth_variance(volume)

    print("    depth_range...", flush=True)
    features['depth_range'] = compute_depth_range(volume)

    print("    intensity_centroid...", flush=True)
    features['intensity_centroid'] = compute_intensity_centroid(volume)

    print("    peak_depth_layer...", flush=True)
    features['peak_depth_layer'] = compute_peak_depth_layer(volume)

    print("    depth_skewness...", flush=True)
    features['depth_skewness'] = compute_depth_skewness(volume)

    print("    depth_kurtosis...", flush=True)
    features['depth_kurtosis'] = compute_depth_kurtosis(volume)

    print("    top_bottom_ratio...", flush=True)
    features['top_bottom_ratio'] = compute_top_bottom_ratio(volume)

    print("    adjacent_layer_correlation...", flush=True)
    features['adj_layer_corr'] = compute_adjacent_layer_correlation(
        volume, patch_size=adj_corr_patch_size)

    return features
