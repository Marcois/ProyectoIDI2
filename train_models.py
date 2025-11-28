import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from HybridMLP import HybridMLP
from Vit224 import ImageDataset, ViTClassifier

# ----------------------------------
# Models train script
# ----------------------------------
if __name__ == "__main__":

    df_hybrid_train_path = 'images_realvsAI_hybrid_train.csv'
    df_hybrid_test_path = 'images_realvsAI_hybrid_test.csv'
    df_vit_train_path = 'images_realvsAI_vit_train.csv'
    df_vit_test_path = 'images_realvsAI_vit_test.csv'

    # Ensure all required dataset files exist before training
    if (os.path.exists(df_hybrid_train_path) and  
        os.path.exists(df_hybrid_test_path)  and
        os.path.exists(df_vit_test_path)     and
        os.path.exists(df_vit_test_path)):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- TRAIN Hybrid model ----------
        print('--- Hybrid model training: ---')

         # Load hybrid training features (handcrafted + CNN)
        df_train = pd.read_csv(df_hybrid_train_path)
        X_train = df_train.drop(['file_name', 'label'], axis=1).to_numpy(dtype='float32')
        y_train = df_train['label'].to_numpy()

        df_test = pd.read_csv(df_hybrid_test_path)
        X_test = df_test.drop(['file_name', 'label'], axis=1).to_numpy(dtype='float32')
        y_test = df_test['label'].to_numpy()

        # Config
        epochs = 50
        batch_size = 16

         # Hybrid classifier: MLP receiving concatenated features
        model = HybridMLP(
            input_dim=X_train.shape[1], 
            use_sigmoid=False, 
            device=device
        ).to(device)
                
        model.fit(
            train_data=(X_train, y_train), 
            test_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size)
        
        
        # ----- TRAIN ViT model "google/vit-base-patch16-224-in21k -----"
        print('\n--- ViT model "google/vit-base-patch16-224-in21k training: ---')

        # Vision Transformer fine-tuning
        model = ViTClassifier(
            model_name="google/vit-base-patch16-224-in21k", 
            lr=2e-5
        )

        # Config
        epochs = 5
        batch_size = 16

         # -------- Train dataloader --------
        df_train = pd.read_csv(df_vit_train_path)
        train_dataset = ImageDataset(df_train)
        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=model.collate_fn
        )
        print("Train dataloader generated")

        # -------- Test dataloader --------
        df_test = pd.read_csv(df_vit_test_path)
        test_dataset = ImageDataset(df_test)
        test_dl = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=model.collate_fn
        )
        print("Test dataloader generated")

        model.fit(
            train_dl=train_dl,
            test_dl=test_dl,
            epochs=epochs
        )
        
    else:
        print(f"Check the paths of datasets")