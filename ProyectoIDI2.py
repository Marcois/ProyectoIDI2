import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, recall_score, f1_score
import torch
from HybridMLP import HybridMLP

# Constants
SEED=6


def metrics_evaluation(y_val, y_prob):
    # Calculate ROC curve and AUC 
    fpr, tpr, _ = roc_curve(y_val, y_prob) 
    auc_val = auc(fpr, tpr) 

    y_pred = (y_prob > 0.5).astype(int)
    recall = recall_score(y_val, y_pred) # Calculate recall 
    f1 = f1_score(y_val, y_pred) # Calculate F1-score 
        
    print("\n--- Results evaluation ---") 
    print(f"Recall: {recall:.4f}") 
    print(f"F1-score: {f1:.4f}") 
    print(f"AUC: {auc_val:.4f}") 

    # Plot ROC curve 
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.2f})') 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    #plt.xlim([0.0, 1.0]) 
    #plt.ylim([0.0, 1.05]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve') 
    plt.legend(loc="lower right") 
    plt.show() 


# ----------------------------------
# Model train and evaluation script
# ----------------------------------
if __name__ == "__main__":
    df_path = 'images_realvsAI_processed.csv'
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        features = df.drop(['file_name', 'label'], axis=1)
        labels = df['label']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train and test split
        X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(
        features.to_numpy(dtype='float32'),
        labels.to_numpy(dtype='float32'),
        test_size=0.7,
        random_state=SEED,
        stratify=labels
        )

        # test and validation split
        X_test, X_val, y_test, y_val = train_test_split(
        X_test_tmp,
        y_test_tmp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_test_tmp
        )

        # Model
        model = HybridMLP(input_dim=X_train.shape[1], use_sigmoid=False, device=device).to(device)
        
        # TRAIN
        epochs = 50
        batch_size = 16
        model.fit(
            train_data=(X_train, y_train), 
            test_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size)
        
        # VALIDATION
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        probs = model.predict(X_val)
        # Evaluate model performance
        metrics_evaluation(y_val, probs)

    else:
        print(f"Dataset {df_path} not found")