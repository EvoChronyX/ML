import os, math
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold
)


sns.set(style="whitegrid")

print("Libraries installed successfully..")



def image_pipeline(file_path, img_size=(128, 128), n_components=50):

    # =========================================================
    # 1. IMAGE EDA (your logic, slightly modularized)
    # =========================================================
    def image_eda(file_path):

        print(f"\n[EDA] Processing: {os.path.basename(file_path)}")

        image_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

        image_paths, labels = [], []

        # --- Scan ---
        for root, _, files in os.walk(file_path):
            for f in files:
                if f.lower().endswith(image_ext):
                    p = os.path.join(root, f)
                    image_paths.append(p)
                    labels.append(os.path.basename(os.path.dirname(p)))

        print(f"Images: {len(image_paths)} | Classes: {len(set(labels))}")
        if not image_paths:
            return None

        # --- Containers ---
        widths, heights, ratios, channels, formats = [], [], [], [], []
        corrupted = []

        brightness, contrast, sparsity = [], [], []
        hist = np.zeros(256)

        ch_sum = np.zeros(3)
        ch_sq = np.zeros(3)
        pixel_count = 0

        # --- Loop ---
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    formats.append(img.format)
                    w, h = img.size
                    widths.append(w); heights.append(h)
                    ratios.append(w/h if h else 0)
                    channels.append(len(img.getbands()))

                img_cv = cv2.imread(p)
                if img_cv is None:
                    continue

                img_f = img_cv.astype(np.float32)
                h, w, c = img_f.shape
                n = h*w

                pixel_count += n

                for i in range(c):
                    ch_sum[i] += img_f[:,:,i].sum()
                    ch_sq[i] += (img_f[:,:,i]**2).sum()

                brightness.append(img_f.mean())
                contrast.append(img_f.std())

                sparsity.append((img_f==0).sum()/(h*w*c)*100)

                hist += np.histogram(img_cv.flatten(), bins=256, range=(0,256))[0]

            except:
                corrupted.append(p)

        # --- Stats ---
        print("\nFormats:", Counter(formats))
        print("Channels:", Counter(channels))
        print("Corrupted:", len(corrupted))

        if pixel_count:
            mean = ch_sum/pixel_count
            std = np.sqrt(ch_sq/pixel_count - mean**2)

            print("\nChannel Stats:")
            for i in range(len(mean)):
                print(f"Channel {i}: Mean={mean[i]:.3f}, Std={std[i]:.3f}")

            print("\nDataset Stats:")
            print(f"Brightness: {np.mean(brightness):.2f}")
            print(f"Contrast: {np.mean(contrast):.2f}")
            print(f"Sparsity: {np.mean(sparsity):.2f}%")

        # --- Visualization ---
        fig, ax = plt.subplots(2,3, figsize=(16,9))

        cnt = Counter(labels)
        ax[0,0].bar(cnt.keys(), cnt.values())
        ax[0,0].set_title("Class Distribution")

        ax[0,1].scatter(widths, heights)
        ax[0,1].set_title("Resolution")

        sns.kdeplot(ratios, ax=ax[0,2], fill=True)
        ax[0,2].set_title("Aspect Ratio")

        ax[1,0].bar(range(256), hist)
        ax[1,0].set_title("Pixel Histogram")

        ax[1,1].scatter(brightness, contrast)
        ax[1,1].set_title("Brightness vs Contrast")

        sns.kdeplot(sparsity, ax=ax[1,2], fill=True)
        ax[1,2].set_title("Sparsity")

        plt.tight_layout()
        plt.show()

        return image_paths, labels
    # =========================================================
    # 2. IMAGE PREPROCESSING
    # =========================================================
    def preprocess_images(image_paths):
        processed_images = []
        valid_paths = []

        for path in tqdm(image_paths, desc="Preprocessing Images"):
            try:
                img = cv2.imread(path)
                if img is None:
                    continue

                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0  # normalization

                processed_images.append(img)
                valid_paths.append(path)

            except:
                continue

        return np.array(processed_images), valid_paths

    # =========================================================
    # 3. FEATURE ENGINEERING (basic image transforms)
    # =========================================================
    def feature_engineering(images):
        features = []

        for img in images:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # simple handcrafted features
            mean = np.mean(gray)
            std = np.std(gray)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            features.append([mean, std, edge_density])

        return np.array(features)

    # =========================================================
    # 4. FEATURE EXTRACTION (flatten + PCA)
    # =========================================================
    def feature_extraction(images):
        n_samples = images.shape[0]

        flat = images.reshape(n_samples, -1)

        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat)

        pca = PCA(n_components=min(n_components, flat_scaled.shape[0], flat_scaled.shape[1]))
        reduced = pca.fit_transform(flat_scaled)

        return reduced, scaler, pca

    # =========================================================
    # 5. FEATURE SELECTION (same philosophy as tabular)
    # =========================================================
    def feature_selection(X, y):
        # remove low variance
        vt = VarianceThreshold(0.01)
        X_vt = vt.fit_transform(X)

        # embedded selection
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_vt, y)

        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:min(20, len(importances))]

        return X_vt[:, idx], idx

    # =========================================================
    # 6. PIPELINE EXECUTION
    # =========================================================
    print("\n[PIPELINE] Starting Image Processing Pipeline...\n")

    image_paths, labels = image_eda(file_path)

    images, valid_paths = preprocess_images(image_paths)

    # align labels
    labels = [os.path.basename(os.path.dirname(p)) for p in valid_paths]
    labels = pd.Series(labels).astype("category").cat.codes

    engineered_features = feature_engineering(images)

    extracted_features, scaler, pca = feature_extraction(images)

    # combine features
    X_combined = np.hstack([engineered_features, extracted_features])

    X_selected, selected_idx = feature_selection(X_combined, labels)

    print("\n[PIPELINE COMPLETE]")
    print(f"Final Feature Shape: {X_selected.shape}")

    return {
        "X": X_selected,
        "y": labels,
        "scaler": scaler,
        "pca": pca,
        "selected_features_idx": selected_idx
    }