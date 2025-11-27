import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, recall_score, f1_score
import torch
from HybridMLP import HybridMLP

# Constants
SEED=6

# Metrics to evaluate
def metrics_evaluation(y_val, y_prob, model_name):
    # Calculate ROC curve and AUC 
    fpr, tpr, _ = roc_curve(y_val, y_prob) 
    auc_val = auc(fpr, tpr) 

    y_pred = (y_prob > 0.5).astype(int)
    recall = recall_score(y_val, y_pred) # Calculate recall 
    f1 = f1_score(y_val, y_pred) # Calculate F1-score 
        
    print("\n--- Results evaluation ---") 
    print(f"{model_name} Recall: {recall:.4f}") 
    print(f"{model_name} F1-score: {f1:.4f}") 
    print(f"{model_name} AUC: {auc_val:.4f}") 

    # Plot ROC curve 
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.2f})') 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    #plt.xlim([0.0, 1.0]) 
    #plt.ylim([0.0, 1.05]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title(f'{model_name} ROC Curve') 
    plt.legend(loc="lower right") 
    plt.show() 


# ----------------------------------
# Model evaluation script
# ----------------------------------
if __name__ == "__main__":
    df_val_path = 'images_realvsAI_val.csv'
    if os.path.exists(df_val_path):
        df_val = pd.read_csv(df_val_path)
        X_val = df_val.drop(['file_name', 'label'], axis=1).to_numpy(dtype='float32')
        y_val = df_val['label'].to_numpy()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # VALIDATION Hybrid model performance
        model = HybridMLP(input_dim=X_val.shape[1], use_sigmoid=False, device=device).to(device)
        
        model.load_state_dict(torch.load("best_HybridMLP_model.pth", weights_only=True))
        probs = model.predict(X_val)
      
        metrics_evaluation(y_val, probs, model_name='Hybrid MLP')

    else:
        print(f"Dataset {df_val_path} not found")