# Cross-Fragment Ink Signal Generalization in Herculaneum Scroll CT Data

A systematic study of which features generalize across scroll fragments 
for ink detection. The central question: why do current ink detection 
models fail to transfer between scrolls, and what features actually 
transfer?

This is not a new model. It is a feature analysis that quantifies where 
generalization fails and identifies one class of features — 3D depth 
features, optionally with Paganin preprocessing — that genuinely 
transfers across fragments from different scrolls.

## Background

Current TimeSformer-based ink detection models achieve strong results 
within-scroll but fail to transfer across scrolls. The working hypothesis 
is that these models overfit to scroll-specific artifacts (crack 
morphology, surface topology) rather than learning generalizable ink 
features. This study tests that hypothesis quantitatively.

## Data

Three DLS fragments with ground-truth ink labels from IR photography:

| Fragment | Scroll | Dimensions | Peak SNR Layer |
|----------|--------|------------|----------------|
| Frag1 (PHerc. Paris 2 Fr 47) | Paris 2 | 8181×6330×65 | Layer 59 |
| Frag2 (PHerc. Paris 2 Fr 143) | Paris 2 | 14830×9506×65 | Layer 64 |
| Frag3 (PHerc. Paris 1 Fr 34) | Paris 1 | 7606×5249×65 | Layer 49 |

54 keV surface volumes only. Frag1 and Frag2 are from the same scroll; 
Frag3 is from a different scroll, making Frag1/2 vs Frag3 a true 
cross-scroll test.

Data is from the EduceLab-Scrolls dataset (open access via 
https://dl.ash2txt.org/fragments/). Fragment surface volumes are not 
included in this repo.

## Methodology

### The problem with pixel-level evaluation

Pixel-level AUC on ink detection is inflated by spatial autocorrelation. 
Ink regions cluster spatially — a classifier trained on one part of a 
fragment achieves high AUC on another part simply because neighboring 
pixels share labels. This is not generalization.

**Corrected approach:** Spatially blocked sampling at 64×64 block level. 
Blocks are assigned entirely to train or test sets, eliminating 
pixel-level leakage. Balanced accuracy is the primary metric.

### Features tested

**2D texture features (9):** Raw intensity at peak SNR layer, gradient 
magnitude (Sobel), structure tensor eigenvalues and coherence, Hessian 
eigenvalues and Laplacian, local variance (11×11 window).

**3D depth features (10):** Computed across all 65 depth layers at each 
(x, y) location: depth gradient magnitude, max depth gradient, depth 
variance, depth range, intensity centroid, peak depth layer, 
top/bottom ratio, adjacent layer correlation, depth skewness, depth 
kurtosis.

### Evaluation

Logistic regression trained on one fragment, tested on all others. 
Cross-fragment AUC is the mean AUC across all cross-fragment test pairs. 
Block sizes tested: 16×16, 32×32, 64×64 — results are consistent across 
all three, confirming the findings are not block-size artifacts.

## Results

### 2D texture features do not generalize at block level

All 9 tested 2D features produce cross-fragment AUC of 0.50–0.55 at 
64×64 block level. The combined 9-feature model drops from within-fragment 
AUC of 0.60 to cross-fragment AUC of 0.55 — the generalization gap 
quantifies the overfitting visible in cross-scroll inference.

Note: at pixel level, these same features show AUC of 0.74–0.75 on 
Frag1. That number is entirely spatial autocorrelation.

### 3D depth features transfer with zero generalization gap

| Feature | Within-frag AUC | Cross-frag AUC | Gap |
|---------|:-:|:-:|:-:|
| depth_gradient_mag | 0.619 | 0.619 | 0.000 |
| max_depth_gradient | 0.624 | 0.624 | 0.000 |
| depth_range | 0.616 | 0.616 | 0.000 |
| depth_variance | 0.604 | 0.604 | 0.000 |

Zero generalization gap means the classifier learned something real about 
ink's effect on depth profiles, not something specific to one fragment's 
geometry. The cross-scroll AUC (Frag3 tested against Frag1/2 trained) is 
essentially identical to the cross-fragment number.

### Paganin preprocessing raises cross-fragment AUC from 0.62 to 0.77

Applying Paganin filtering (δ/β=200) to each surface volume layer before 
computing depth features:

| Feature | Raw cross-AUC | Paganin d/b=200 | Change |
|---------|:-:|:-:|:-:|
| depth_gradient_mag | 0.619 | 0.766 | +0.147 |
| max_depth_gradient | 0.624 | 0.771 | +0.147 |
| depth_variance | 0.604 | 0.757 | +0.153 |
| depth_range | 0.616 | 0.765 | +0.148 |

Zero generalization gap is preserved. d/b=200 outperforms d/b=5000 --
consistent with DLS being in the geometric optics regime (Fresnel
number ~14). DLS reconstruction does not include Paganin filtering, so
this is first-application Paganin-form smoothing on raw
absorption-reconstructed data. The filter acts as regularized noise
suppression at these scan parameters, not phase retrieval. The AUC
improvement mechanism is noise suppression that preserves depth
structure, not fringe removal. Per-layer noise creates fragment-specific
depth profiles; mild smoothing makes depth gradients more consistent
across fragments without destroying the ink signal.

## Physical interpretation

Ink penetrates papyrus during writing and creates a characteristic 
pattern in how intensity varies across the 65 depth layers — 
specifically, a sharper gradient (higher |dI/dz|) at ink-bearing layers. 
This depth signature is a property of how ink interacts with papyrus 
during writing and carbonization, and is consistent across fragments.

2D texture features instead capture papyrus fiber arrangement, crack 
morphology, and surface topology — all scroll-specific rather than 
ink-specific.

## Practical recommendations

**Use depth features as input channels** for cross-scroll ink detection 
rather than or in addition to raw intensity layers. The zero-gap property 
means a model trained on depth features from one fragment should transfer 
to others.

**Apply mild Paganin preprocessing (d/b=200) before computing depth 
features.** This raises cross-fragment AUC by ~+0.15 while preserving 
the zero generalization gap. Single FFT pair per layer — negligible cost.

## Installation
```bash
conda create -n vesuvius python=3.10
conda activate vesuvius
conda env update -f environment.yml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage
```bash
# Compute all 10 depth features per fragment (requires downloaded data)
python scripts/run_depth_features.py

# Evaluate cross-fragment generalization at block level
python scripts/run_depth_cross_validation.py

# 2D texture feature baseline (Phase 2 comparison)
python scripts/run_cross_validation.py

# Paganin + depth features experiment
python scripts/test_paganin_plus_depth_features.py
```

Data should be placed under `data/fragments/frag{1,2,3}/54keV_exposed_surface/`.

## Key outputs

| File | Description |
|------|-------------|
| `results/tables/depth_feature_auc_summary.csv` | Block-level cross-fragment AUC at 3 block sizes |
| `results/tables/paganin_depth_feature_summary.csv` | Paganin + depth features comparison |
| `results/tables/cross_fragment_auc_summary.csv` | 2D feature baseline results |
| `results/figures/paganin_depth_feature_comparison.png` | Main result figure |
| `results/figures/cross_fragment_auc_heatmaps.png` | 2D feature generalization matrix |

## Limitations

- Three fragments only (Frags 4–6 have no available surface volumes at 
  this energy; 88 keV does not exist for these fragments)
- Logistic regression is a linear ceiling on what these features can do 
  alone — the zero-gap result is necessary but not sufficient for strong 
  ink detection
- Raw cross-AUC values of 0.61–0.62 indicate real signal but not 
  standalone detection. Paganin preprocessing raises this to 0.76–0.77.
- The Paganin improvement on DLS data is from noise suppression, not 
  phase retrieval. The effect on ESRF data (Fresnel number ~1.1) may 
  differ and is not yet validated.

## Data license

EduceLab-Scrolls Dataset (CC-BY-NC 4.0). Cite: Parsons et al. (2023), 
EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri 
using X-ray CT. arXiv:2304.02084.
