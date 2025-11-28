import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from HybridMLP import HybridMLP
from Vit224 import ImageDataset, ViTClassifier

# Metrics to evaluate
def metrics_evaluation(y_val, y_prob, model_name, predict_time):
    # Calculate ROC curve and AUC 
    fpr, tpr, _ = roc_curve(y_val, y_prob) 
    auc_val = auc(fpr, tpr) 

    # Calculate F1-score and Recall
    y_pred = (y_prob > 0.5).astype(int)
    recall = recall_score(y_val, y_pred) 
    f1 = f1_score(y_val, y_pred) 
    
     # PRINT results
    print(f"\n--- {model_name} results evaluation ---") 
    print(f"{model_name} Recall: {recall:.4f}") 
    print(f"{model_name} F1-score: {f1:.4f}") 
    print(f"{model_name} AUC: {auc_val:.4f}")
    print(f"Prediction time: {predict_time:.4f} seconds")

    # Save log to txt
    txt_path = os.path.join("results", f"{model_name}_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"--- {model_name} results evaluation ---\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"AUC: {auc_val:.4f}\n")
        f.write(f"Prediction time: {predict_time:.4f} seconds\n")


    print(f"Metrics saved to: {txt_path}")

    # Plot ROC curve 
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.2f})') 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title(f'{model_name} ROC Curve') 
    plt.legend(loc="lower right") 

    # ---- Save ROC image ----
    img_path = os.path.join("results", f"{model_name}_roc_curve.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to: {img_path}\n")


# -------------------------------------------
# Model evaluation script
# -------------------------------------------
if __name__ == "__main__":

    df_hybrid_val_path = 'images_realvsAI_hybrid_val.csv'
    df_vit_val_path = 'images_realvsAI_vit_val.csv'

    # Check that both validation datasets exist
    if os.path.exists(df_hybrid_val_path) and os.path.exists(df_vit_val_path):
        
        # -----------------------------------
        # Validate Hybrid MLP model
        # -----------------------------------
        df_val = pd.read_csv(df_hybrid_val_path)

        X_val = df_val.drop(['file_name', 'label'], axis=1).to_numpy(dtype='float32')
        y_val = df_val['label'].to_numpy()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = HybridMLP(
            input_dim=X_val.shape[1], 
            use_sigmoid=False, 
            device=device
        ).to(device)
        
        # Load best weights saved during training
        model.load_state_dict(torch.load("best_HybridMLP_model.pth", weights_only=True))
        
        # Predict probability on the validation set and measure prediction time
        start = time.time()
        probs = model.predict(X_val)
        predict_time = time.time() - start

        # Compute metrics 
        metrics_evaluation(
            y_val, 
            probs, 
            model_name='Hybrid MLP', 
            predict_time=predict_time
        )


        # -----------------------------------
        # Validate Vision Transformer (ViT)
        # -----------------------------------
        model = ViTClassifier(
            model_name="google/vit-base-patch16-224-in21k", 
            lr=2e-5
        )
        
        # Load best weights saved during training
        model.load_state_dict(torch.load("best_ViT_model.pth", weights_only=True))

        batch_size = 16

        # Create dataset and dataloader
        df_val = pd.read_csv(df_vit_val_path)
        y_val = df_val['label'].to_numpy()

        val_dataset = ImageDataset(df_val)
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=model.collate_fn
        )

        print("\nValidation dataloader generated")

        # Predict probability on the validation set and measure prediction time
        start = time.time()
        probs = model.predict(val_dl)
        predict_time = time.time() - start

        # Compute metrics
        metrics_evaluation(
            y_val, 
            probs, 
            model_name='ViT',
            predict_time=predict_time
        )

    else:
        print(f"Check the paths of datasets")