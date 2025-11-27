import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.fft import fft2
from skimage.restoration import denoise_wavelet


# -------------------------------
# Feature Extractors
# -------------------------------

# PRNU
def extract_prnu(img_gray):
    denoised = denoise_wavelet(img_gray, convert2ycbcr=False, mode='soft')
    residual = img_gray - denoised
    return np.array([
        np.mean(residual),
        np.std(residual),
        np.var(residual),
        np.mean(np.abs(residual)),
        np.median(residual),
    ])

# ELA
def extract_ela(img_bgr, quality=90):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img_rgb, encode_param)
    recompressed = cv2.imdecode(enc, 1)
    diff = cv2.absdiff(img_rgb, recompressed)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    return np.array([
        np.mean(gray),
        np.std(gray),
        np.max(gray),
        np.median(gray),
        np.percentile(gray, 90),
    ])

# Sobel
def extract_sobel(img_bgr):
    sobelx = cv2.Sobel(img_bgr, cv2.CV_64F, 1, 0, ksize=3) # detect vertical edges
    sobely = cv2.Sobel(img_bgr, cv2.CV_64F, 0, 1, ksize=3) # detect horizontal edges
    sobel_mag = np.sqrt(sobelx**2 + sobely**2) # compute gradient magnitude
    return np.array([
        np.mean(sobel_mag),
        np.std(sobel_mag),
    ])

# LBP
def extract_lbp(img_gray):
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    return hist

# GLCM
def extract_glcm(img_gray):
    gl = graycomatrix((img_gray * 255).astype(np.uint8), distances=[1], angles=[0], levels=256)
    return np.array([
        graycoprops(gl, 'contrast')[0, 0],
        graycoprops(gl, 'dissimilarity')[0, 0],
        graycoprops(gl, 'homogeneity')[0, 0],
        graycoprops(gl, 'ASM')[0, 0],
        graycoprops(gl, 'energy')[0, 0],
        graycoprops(gl, 'correlation')[0, 0],
    ])

# FFT
def extract_frequency(img_gray):
    f = np.abs(fft2(img_gray))
    return np.array([
        np.mean(f),
        np.std(f),
        np.max(f),
        np.percentile(f, 99),
        np.percentile(f, 95),
    ])

# Color stats
def extract_color_stats(img_bgr):
    feats = []
    for c in cv2.split(img_bgr):
        feats += [np.mean(c), np.std(c), np.median(c)]
    return np.array(feats)