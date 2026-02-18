"""Shared feature extraction for training and inference (must stay in sync)."""
import numpy as np
import cv2
import scipy.stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label as sk_label


def preprocess(img_bgr, size=(224, 224)):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def _box_count_fd(binary_img: np.ndarray) -> float:
    """Fractal dimension via box-counting (approximation)."""
    sizes = [2, 4, 8, 16, 32]
    sizes_used, counts = [], []
    for s in sizes:
        h = binary_img.shape[0] // s
        w = binary_img.shape[1] // s
        if h == 0 or w == 0:
            continue
        blocks = binary_img[: h * s, : w * s].reshape(h, s, w, s)
        counts.append(blocks.any(axis=(1, 3)).sum())
        sizes_used.append(s)
    if len(counts) < 2:
        return 0.0
    coeffs = np.polyfit(
        np.log(sizes_used), np.log(np.array(counts) + 1e-6), 1
    )
    return float(-coeffs[0])


def extract_features(gray: np.ndarray) -> np.ndarray:
    """
    Feature vector combining:
    - Intensity statistics (brightness distribution)
    - GLCM texture features
    - LBP texture histogram
    - Morphological shape features (via nucleus segmentation)
    - Fractal dimension approximation
    """
    # -------------------------
    # 1. Intensity statistics
    # -------------------------
    mean = float(gray.mean())
    std = float(gray.std())
    p10 = float(np.percentile(gray, 10))
    p50 = float(np.percentile(gray, 50))
    p90 = float(np.percentile(gray, 90))
    skew = float(scipy.stats.skew(gray.ravel()))      # asymmetry of brightness
    kurt = float(scipy.stats.kurtosis(gray.ravel()))  # tail heaviness
    hist_vals = np.histogram(gray, bins=256, range=(0, 1), density=True)[0]
    entropy = float(
        -np.sum(hist_vals * np.log2(hist_vals + 1e-12))
    )  # complexity/randomness of pixel distribution

    # -------------------------
    # 2. GLCM texture features
    # -------------------------
    g = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(
        g, distances=[1, 2], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())

    # -------------------------
    # 3. LBP texture histogram
    # -------------------------
    # LBP captures fine local texture patterns — very effective on histology images
    lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    lbp_hist = lbp_hist.astype(np.float32)  # 10-number summary of texture pattern distribution

    # -------------------------
    # 4. Morphological shape features
    # -------------------------
    # Threshold to isolate nucleus (simple Otsu)
    thresh = threshold_otsu(g)
    binary = g > thresh
    labeled = sk_label(binary)
    regions = regionprops(labeled)

    if regions:
        # Take the largest region (assumed to be the nucleus)
        region = max(regions, key=lambda r: r.area)
        area = float(region.area)
        perimeter = float(region.perimeter)
        compactness = float((perimeter ** 2) / (area + 1e-6))  # irregularity of shape
        eccentricity = float(region.eccentricity)               # elongation
        solidity = float(region.solidity)                      # convexity of nucleus
    else:
        area = perimeter = compactness = eccentricity = solidity = 0.0

    # -------------------------
    # 5. Fractal dimension (box-counting approximation)
    # -------------------------
    # Measures boundary complexity — malignant nuclei tend to have higher fractal dimension
    fractal_dim = _box_count_fd(binary)

    # -------------------------
    # Assemble final vector
    # -------------------------
    feats = np.array(
        [
            # intensity (8)
            mean, std, p10, p50, p90, skew, kurt, entropy,
            # GLCM texture (4)
            contrast, homogeneity, energy, correlation,
            # LBP histogram (10)
            *lbp_hist,
            # shape (5)
            area, perimeter, compactness, eccentricity, solidity,
            # fractal (1)
            fractal_dim,
        ],
        dtype=np.float32,
    )
    return feats  # 28-dimensional vector
