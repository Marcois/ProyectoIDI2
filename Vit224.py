import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, ViTForImageClassification
from tqdm import tqdm


# ---------------------------
# Dataset for ViT
# ---------------------------
class ImageDataset(Dataset):
    """Dataset that loads image paths and labels from a DataFrame"""

    def __init__(self, df):
        self.paths = df["file_name"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Return (PIL RGB image, label)
        """
        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")
        return img, label



# ---------------------------
# ViTClassifier
# ---------------------------
class ViTClassifier(nn.Module):
    def __init__(self, model_name, lr=2e-5):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Image processor (resize, normalization, formatting for ViT)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

        # ViT model with a single output logit for binary classification
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=1,
        ).to(self.device)

        # Binary cross-entropy loss with logits
        self.loss = nn.BCEWithLogitsLoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

    def collate_fn(self, batch):
        """
        Custom collate function: converts a batch of PIL images + labels
        into tensors expected by ViT (pixel_values + float labels)
        """
        imgs, labels = zip(*batch)

        # Processor handles resize, normalization, and channel ordering
        enc = self.processor(
            images=list(imgs),
            return_tensors="pt"
        )

        # Add labels (B,1) as float tensor
        enc["labels"] = torch.tensor(labels).float().unsqueeze(1)

        return enc

    # ---------------------------------
    # Training
    # ---------------------------------
    def fit(self, train_dl, test_dl, epochs):
        """
        Train the ViT model and evaluate on validation set each epoch
        Saves the best model based on Recall
        """
        best_recall = 0.0

        for epoch in tqdm(range(epochs), desc='Epochs'):

            # ---------------- TRAIN LOOP ----------------
            self.train()
            train_loss = 0.0

            for batch in train_dl:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits

                # Compute BCE loss
                loss = self.loss(logits, labels)

                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dl)

            # ---------------- VALIDATION LOOP ----------------
            self.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for batch in test_dl:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(pixel_values=pixel_values)
                    logits = outputs.logits

                    loss = self.loss(logits, labels)
                    val_loss += loss.item()

                    # Convert logits to probabilities
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels.cpu().numpy().flatten().tolist())

            val_loss /= len(test_dl)

            # Convert to arrays for metrics
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            # Binary predictions using threshold = 0.5
            preds = (all_probs >= 0.5).astype(int)

            # Validation metrics
            auc = roc_auc_score(all_labels, all_probs)
            recall = recall_score(all_labels, preds)

            # Save best model based on Recall
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch + 1
                torch.save(self.state_dict(), "best_ViT_model.pth")

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"AUC={auc:.4f} | Recall={recall:.4f}"
            )

        print(f"\nBest model saved with Recall = {best_recall:.4f} at epoch {best_epoch}")


    # Predict
    def predict(self, dl):
        """
        Predict probabilities for all samples in a dataloader
        Returns a NumPy vector of sigmoid outputs in the same order
        """
        self.eval()
        all_probs = []

        with torch.no_grad():
            for batch in dl:
                pixel_values = batch["pixel_values"].to(self.device)

                # Forward pass
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits

                # Convert logits to probabilities
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs.tolist())
        
        return np.array(all_probs)
        
