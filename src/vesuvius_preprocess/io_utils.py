"""
I/O utilities for loading Vesuvius Challenge fragment and scroll data.

Handles loading surface volumes from TIF stacks, ink label PNGs,
and infrared reference images. All loaders return numpy arrays.

Expected local directory layout (after download_fragments.py):
    data/fragments/frag1/
        54keV_exposed_surface/
            surface_volume/00.tif, 01.tif, ...
            ir.png
            inklabels.png
        88keV_exposed_surface/
            surface_volume/00.tif, 01.tif, ...
            ir.png
            inklabels.png
"""

import glob
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tifffile
from PIL import Image


def load_surface_volume(
    surface_volume_dir: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a surface volume from a directory of TIF files.

    Each TIF file is one depth layer. Files are sorted lexicographically
    (typically named 00.tif, 01.tif, ... or similar zero-padded).

    Parameters
    ----------
    surface_volume_dir : str
        Path to directory containing .tif files.
    dtype : np.dtype
        Output dtype. Default float32 for downstream computation.

    Returns
    -------
    np.ndarray
        3D array of shape (H, W, D) where D is the number of depth layers.
    """
    tif_files = sorted(glob.glob(os.path.join(surface_volume_dir, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files found in {surface_volume_dir}. "
            f"Contents: {os.listdir(surface_volume_dir)[:10] if os.path.exists(surface_volume_dir) else 'DIR NOT FOUND'}"
        )

    first = tifffile.imread(tif_files[0])
    H, W = first.shape[:2]
    D = len(tif_files)

    volume = np.empty((H, W, D), dtype=dtype)
    volume[:, :, 0] = first.astype(dtype)

    for i, f in enumerate(tif_files[1:], start=1):
        volume[:, :, i] = tifffile.imread(f).astype(dtype)

    return volume


def load_ink_labels(label_path: str) -> np.ndarray:
    """Load binary ink label mask from PNG.

    Parameters
    ----------
    label_path : str
        Path to inklabels.png (binary mask, typically 0/255 or 0/1).

    Returns
    -------
    np.ndarray
        Boolean 2D array, True where ink is present.
    """
    img = np.array(Image.open(label_path))
    return img > 0


def load_ir_image(ir_path: str) -> np.ndarray:
    """Load infrared reference image.

    Parameters
    ----------
    ir_path : str
        Path to ir.png.

    Returns
    -------
    np.ndarray
        2D grayscale array (or 3-channel if RGB).
    """
    return np.array(Image.open(ir_path))


def list_fragment_energies(fragment_dir: str) -> List[str]:
    """List available energy levels for a fragment.

    Looks for directories matching *_exposed_surface/ pattern.

    Parameters
    ----------
    fragment_dir : str
        Path to a fragment directory (e.g., data/fragments/frag1/).

    Returns
    -------
    list of str
        Energy labels (e.g., ['54keV', '88keV']).
    """
    fragment_dir = Path(fragment_dir)
    if not fragment_dir.exists():
        return []

    energies = []
    for d in sorted(fragment_dir.iterdir()):
        if d.is_dir() and d.name.endswith("_exposed_surface"):
            energy = d.name.replace("_exposed_surface", "")
            energies.append(energy)
    return energies


def get_fragment_paths(fragment_dir: str, energy: Optional[str] = None) -> dict:
    """Get paths to all data files for a fragment at a given energy.

    Parameters
    ----------
    fragment_dir : str
        Path to a fragment directory (e.g., data/fragments/frag1/).
    energy : str, optional
        Energy label (e.g., '54keV'). If None, uses the first available.

    Returns
    -------
    dict with keys: 'surface_volume_dir', 'inklabels', 'ir', 'energy'
    """
    fragment_dir = Path(fragment_dir)

    if energy is None:
        energies = list_fragment_energies(str(fragment_dir))
        if not energies:
            raise FileNotFoundError(
                f"No *_exposed_surface/ directories found in {fragment_dir}"
            )
        energy = energies[0]

    energy_dir = fragment_dir / f"{energy}_exposed_surface"
    if not energy_dir.exists():
        raise FileNotFoundError(f"Energy directory not found: {energy_dir}")

    paths = {
        "energy": energy,
        "surface_volume_dir": str(energy_dir / "surface_volume"),
        "inklabels": str(energy_dir / "inklabels.png"),
        "ir": str(energy_dir / "ir.png"),
    }

    # Validate that critical files exist
    if not os.path.isdir(paths["surface_volume_dir"]):
        raise FileNotFoundError(f"Surface volume dir not found: {paths['surface_volume_dir']}")
    if not os.path.isfile(paths["inklabels"]):
        raise FileNotFoundError(f"Ink labels not found: {paths['inklabels']}")

    return paths


def load_fragment(
    fragment_dir: str,
    energy: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a complete fragment: surface volume, ink labels, and IR image.

    Parameters
    ----------
    fragment_dir : str
        Path to a fragment directory (e.g., data/fragments/frag1/).
    energy : str, optional
        Energy label (e.g., '54keV'). If None, uses the first available.

    Returns
    -------
    volume : np.ndarray
        Surface volume, shape (H, W, D).
    ink_labels : np.ndarray
        Boolean ink mask, shape (H, W).
    ir_image : np.ndarray or None
        IR reference image if found, else None.
    """
    paths = get_fragment_paths(fragment_dir, energy)

    print(f"Loading {fragment_dir} @ {paths['energy']}...")
    volume = load_surface_volume(paths["surface_volume_dir"])
    ink_labels = load_ink_labels(paths["inklabels"])

    ir_image = None
    if os.path.isfile(paths["ir"]):
        ir_image = load_ir_image(paths["ir"])

    return volume, ink_labels, ir_image


def print_volume_diagnostics(
    volume: np.ndarray,
    ink_labels: np.ndarray,
    label: str = "",
) -> None:
    """Print basic diagnostics for a loaded fragment."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Volume shape: {volume.shape}, dtype: {volume.dtype}")
    print(f"{prefix}Value range: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"{prefix}Mean: {volume.mean():.1f}, Std: {volume.std():.1f}")
    print(f"{prefix}Ink label shape: {ink_labels.shape}")
    n_ink = ink_labels.sum()
    n_total = ink_labels.size
    print(f"{prefix}Ink pixels: {n_ink:,} / {n_total:,} ({100*n_ink/n_total:.1f}%)")
