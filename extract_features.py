import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import handcrafted_features
from CNNEmbedder import CNNEmbedder

# Constants
IMAGES_NUM=10000 # must be <= 79950 total images
IMG_SIZE=256

# ---------------------------
# Full handcrafted extractor
# ---------------------------
def extract_handcrafted(img, gray, cfg=None):
    """
    Generate handcrafted features for single image
    cfg: dict controlling handcrafted features to include; if None uses defaults
    """
    if cfg is None:
        cfg = {
            'use_prnu': True,
            'use_ela': True,
            'use_sobel': True,
            'use_lbp': True,
            'use_glcm': True,
            'use_fft': True,
            'use_colorstats': True
        }
    feats = []
    if cfg.get('use_prnu', True):
        feats.append(handcrafted_features.extract_prnu(gray))
    if cfg.get('use_ela', True):
        feats.append(handcrafted_features.extract_ela(img))
    if cfg.get('use_sobel', True):
        feats.append(handcrafted_features.extract_sobel(img))
    if cfg.get('use_lbp', True):
        feats.append(handcrafted_features.extract_lbp(gray))
    if cfg.get('use_glcm', True):
        feats.append(handcrafted_features.extract_glcm(gray))
    if cfg.get('use_fft', True):
        feats.append(handcrafted_features.extract_frequency(gray))
    if cfg.get('use_colorstats', True):
        feats.append(handcrafted_features.extract_color_stats(img))
    # flatten and concat
    feats = np.concatenate([f.flatten() for f in feats], axis=0).astype(np.float32)
    return feats

# --------------------
# Dataset processing
# --------------------

def build_processed_dataset(df, images_num, cfg=None, verbose=True):
    """
    Generate dataset with handcrafted features and CNN embedded
    cfg: dict controlling handcrafted features to include; if None uses defaults
    """
    image_paths = df['file_name']
    labels = df['label']
    print("Total images to process:", images_num)

    # Resize and gray obtention
    imgs_preprocessed = []
    for path in tqdm(image_paths, desc='resizing and obtaining grays', disable=not verbose):
        try:
            path = os.path.join('c:/Users/reyin/OneDrive/Documents/Git Repositories/ProyectoIDI2/', path)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            imgs_preprocessed.append((img, gray))
        except Exception as e:
            print("Error", path, e)

    # Handcrafted extraction
    handcrafted_feats = []
    for img_cv2, gray in tqdm(imgs_preprocessed, desc='extracting handcrafted', disable=not verbose):
        feats = extract_handcrafted(img_cv2, gray, cfg=cfg)
        handcrafted_feats.append(feats)
    handcrafted_feats = np.stack(handcrafted_feats, axis=0).astype(np.float32)
    # scaling for handcrafted
    scaler = StandardScaler()
    handcrafted_feats = scaler.fit_transform(handcrafted_feats).astype(np.float32) 

    # CNN embedding extraction
    embeddings = []
    cnn_embedder = CNNEmbedder()
    for img_cv2, gray in tqdm(imgs_preprocessed, desc='extracting CNN embedding', disable=not verbose):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        emb = cnn_embedder.get(img_pil)
        embeddings.append(emb)
    embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    
    # Build processed dataset
    print("handcrafted dim:", handcrafted_feats.shape[1])
    print("embedding dim:", embeddings.shape[1])
    print("total dim handcrafted + embedding:", handcrafted_feats.shape[1]+embeddings.shape[1])
    
    img_features = np.concatenate([handcrafted_feats, embeddings], axis=1).astype(np.float32)
    
    if img_features.shape[0] == images_num:
        print("all images processed")
    else:
        print(f"{img_features.shape[0]} images processed of {images_num}")
    
    df = pd.DataFrame(img_features)
    df.insert(0, 'file_name', image_paths)
    df['label'] = labels
    return df


# -------------------------------
# Build processed dataset script
# -------------------------------

df_path = 'images_realvsAI.csv'
df_processed_path = 'images_realvsAI_processed.csv'
if os.path.exists(df_path):
    df = pd.read_csv(df_path, nrows=IMAGES_NUM)
    df_processed = build_processed_dataset(df, images_num=IMAGES_NUM)
    df_processed.to_csv(df_processed_path, index=False)
    print(f'csv file saved: {df_processed_path}')
else:
    print(f"Dataset {df_path} not found")