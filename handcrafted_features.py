import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.fft import fft2
from skimage.restoration import denoise_wavelet


# -------------------------------
# Feature Extractors
# -------------------------------

# PRNU
# Wavelet denoising to isolate sensor noise residual
def extract_prnu(img_gray):
    denoised = denoise_wavelet(img_gray, convert2ycbcr=False, mode='soft')
    residual = img_gray - denoised
    epsilon = 1e-8
    prnu = residual / (img_gray.astype(np.float32) + epsilon)
    return np.array([
        np.mean(prnu),
        np.std(prnu),
        np.var(prnu),
        np.mean(np.abs(prnu)),
        np.median(prnu),
    ])

# ELA
# JPEG re-encoding to reveal compression inconsistencies
def extract_ela(img_bgr, quality=90):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img_rgb, encode_param)
    recompressed = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
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
# Edge magnitude as a simple measure of global sharpness/structure
def extract_sobel(img_bgr):
    sobelx = cv2.Sobel(img_bgr, cv2.CV_64F, 1, 0, ksize=3) 
    sobely = cv2.Sobel(img_bgr, cv2.CV_64F, 0, 1, ksize=3) 
    sobel_mag = np.sqrt(sobelx**2 + sobely**2) 
    return np.array([
        np.mean(sobel_mag),
        np.std(sobel_mag),
    ])

# LBP
# Uniform LBP histogram captures local texture distribution
def extract_lbp(img_gray):
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    return hist

# GLCM
# First-order GLCM on angle 0° and distance 1 to capture global texture statistics
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
# Global frequency magnitude distribution (useful for detecting AI smoothness)
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
# Channel-level color intensity statistics (captures unnatural color patterns)
def extract_color_stats(img_bgr):
    feats = []
    for c in cv2.split(img_bgr):
        feats += [np.mean(c), np.std(c), np.median(c)]
    return np.array(feats)