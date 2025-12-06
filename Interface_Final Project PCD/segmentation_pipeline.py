import cv2
import numpy as np
from skimage.feature import hog
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

# ====================================================
# STAGE 1 — PREPROCESSING
# ====================================================
def stage1_preprocessing(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    target = 256
    scale = target / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target, target), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    padded[y_off:y_off+new_h, x_off:x_off+new_w] = img_small

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    padded = clahe.apply(padded)

    padded = cv2.GaussianBlur(padded, (3,3), 0)
    padded = cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX)

    return padded

# ====================================================
# STAGE 2 — OTSU + fallback
# ====================================================
def stage2_threshold(img):
    blur = cv2.GaussianBlur(img, (7,7), 0)

    _, otsu = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if np.count_nonzero(otsu) < 2000:
        _, alt = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alt = cv2.bitwise_not(alt)
        if np.count_nonzero(alt) > np.count_nonzero(otsu):
            return alt

    return otsu

# ====================================================
# STAGE 3 — Morph cleaning
# ====================================================
def _odd(x):
    x = int(max(3, round(x)))
    return x if x % 2 == 1 else x + 1

def stage3_clean(mask):
    if np.count_nonzero(mask) < 800:
        return mask

    H, W = mask.shape
    area_img = H * W

    k_close = _odd(min(H, W) * 0.02)
    k_open  = _odd(min(H, W) * 0.01)

    k_close_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    k_open_el  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_el, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k_open_el, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)

    candidates = []
    min_area_frac = 0.005
    max_area_frac = 0.45
    eps = 1e-6

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if not (min_area_frac * area_img <= area <= max_area_frac * area_img):
            continue

        cx = x + w / 2
        if cx < W * 0.05 or cx > W * 0.95:
            continue

        aspect = h / (w + eps)
        score = area * (0.8 + 0.4 * min(aspect, 2) / 2)
        candidates.append((score, i))

    if len(candidates) == 0:
        return cleaned

    candidates = sorted(candidates, reverse=True)
    chosen = [candidates[0][1]]
    if len(candidates) >= 2:
        chosen.append(candidates[1][1])

    final = np.zeros_like(mask)
    for c in chosen:
        final[labels == c] = 255

    final = cv2.medianBlur(final.astype(np.uint8), 5)

    return final

# ====================================================
# STAGE 4 — ROI 128×128
# ====================================================
def stage4_apply_mask(img_preprocessed, mask_clean):
    ys, xs = np.where(mask_clean == 255)
    if len(xs) == 0:
        return None

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    roi = img_preprocessed[y1:y2, x1:x2]

    target = 128
    h, w = roi.shape
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target, target), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    padded = cv2.GaussianBlur(padded, (3,3), 0)
    padded = cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX)
    return padded

# ====================================================
# MASK OVERLAY (warna)
# ====================================================
def colored_border(mask, color=(0,0,255), ksize=5):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    border = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, k)

    border_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    border_color[border > 0] = color
    return border_color

def colored_mask(mask, color=(0,255,255), alpha=0.35):
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_color[mask > 0] = color
    return mask_color, alpha

def overlay_on_image(img, mask_color, alpha, border_color):
    mask_idx = (mask_color.sum(axis=2) > 0)
    overlay = img.copy()

    overlay[mask_idx] = (
        (1 - alpha) * img[mask_idx] +
        alpha * mask_color[mask_idx]
    ).astype(np.uint8)

    border_idx = (border_color.sum(axis=2) > 0)
    overlay[border_idx] = border_color[border_idx]

    return overlay

# ====================================================
# FEATURE EXTRACTION
# ====================================================
def extract_features(img):
    if img is None:
        return np.zeros(120,)

    pixels = img[img > 0].ravel()
    if len(pixels) == 0:
        return np.zeros(120,)

    stats_feat = [
        np.mean(pixels),
        np.std(pixels),
        np.min(pixels),
        np.max(pixels),
        shannon_entropy(pixels),
        skew(pixels),
        kurtosis(pixels)
    ]

    hist, _ = np.histogram(pixels, bins=16, range=(0,255), density=True)

    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(64,64),
        cells_per_block=(1,1),
        block_norm='L2-Hys'
    )

    return np.hstack([stats_feat, hist, hog_feat])

# ====================================================
# FULL PIPELINE UNTUK 1 GAMBAR
# ====================================================
def process_single_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img1 = stage1_preprocessing(img)
    mask1 = stage2_threshold(img1)
    mask2 = stage3_clean(mask1)
    roi = stage4_apply_mask(img1, mask2)
    feat = extract_features(roi)

    # overlay generate
    base = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    border = colored_border(mask2, color=(0,0,255))
    maskcol, alpha = colored_mask(mask2, color=(0,255,255), alpha=0.30)
    overlay = overlay_on_image(base, maskcol, alpha, border)

    return img1, mask2, roi, feat, overlay
