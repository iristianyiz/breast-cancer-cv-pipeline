"""Shared feature extraction for training and inference (must stay in sync)."""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


def preprocess(img_bgr, size=(224, 224)):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def extract_features(gray):
    mean = float(gray.mean())
    std = float(gray.std())
    p10 = float(np.percentile(gray, 10))
    p50 = float(np.percentile(gray, 50))
    p90 = float(np.percentile(gray, 90))

    g = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(g, distances=[1, 2], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())

    return np.array([mean, std, p10, p50, p90, contrast, homogeneity, energy, correlation], dtype=np.float32)
