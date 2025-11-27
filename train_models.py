import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, recall_score, f1_score
import torch
from HybridMLP import HybridMLP

# ----------------------------------
# Models train script
# ----------------------------------
df_train_path = 'images_realvsAI_train.csv'
df_test_path = 'images_realvsAI_test.csv'
if os.path.exists(df_train_path) and os.path.exists(df_test_path):
    df_train = pd.read_csv(df_train_path)
    X_train = df_train.drop(['file_name', 'label'], axis=1).to_numpy(dtype='float32')
    y_train = df_train['label'].to_numpy()

    df_test = pd.read_csv(df_test_path)
    X_test = df_test.drop(['file_name', 'label'], axis=1).to_numpy(dtype='float32')
    y_test = df_test['label'].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TRAIN Hybrid model
    epochs = 50
    batch_size = 16

    model = HybridMLP(input_dim=X_train.shape[1], use_sigmoid=False, device=device).to(device)
            
    model.fit(
        train_data=(X_train, y_train), 
        test_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size)
    
else:
    print(f"Check the paths of datasets")