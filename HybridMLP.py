import numpy as np
from sklearn.metrics import roc_auc_score, recall_score
import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------
# Hybrid MLP (PyTorch)
# ---------------------------
class HybridMLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=64, dropout1=0.3, dropout2=0.2, use_sigmoid=False, device='cpu'):
        super().__init__()
        
        self.device = device

        # If using BCEWithLogitsLoss, final layer must output raw logits
        # so activation becomes identity lambda x: x
        self.use_sigmoid = use_sigmoid

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout1),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout2),

            nn.Linear(hidden2, 1)
        )
        if self.use_sigmoid:
            self.act = nn.Sigmoid()
        else:
            # if activation is in loss function e.g. BCEWithLogitsLoss
            self.act = lambda x: x

    def forward(self, x):
        # Input is [B, embedding_dim + handcrafted_dim]
        # Output is raw logits unless use_sigmoid=True
        # BCEWithLogitsLoss expects logits -> do NOT apply sigmoid here
        x = self.net(x)
        x = self.act(x)
        return x.view(-1)  # logits if not sigmoid
    
    def fit(self, train_data, test_data, epochs, batch_size):
        """
        Train the hybrid model and evaluate on validation set each epoch
        Saves the best model based on Recall
        """
        # split train and test
        X_train, y_train = train_data
        X_test, y_test = test_data

        # Convert numpy arrays → torch tensors on correct device
        X_train_t = torch.from_numpy(X_train).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        X_test_t = torch.from_numpy(X_test).to(self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        opt = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

        # BCEWithLogitsLoss = sigmoid + BCE in one stable operation
        loss_fn = nn.BCEWithLogitsLoss()

        n_train = X_train_t.size(0)
        best_recall = 0.0
        for epoch in tqdm(range(epochs), desc='Epochs'):

            # -------------- Train --------------
            self.train()
            idx = np.random.permutation(n_train) # Shuffling indices manually
            train_loss = 0.0

            for i in range(0, n_train, batch_size):
                batch_idx = idx[i:i+batch_size]

                X_b = X_train_t[batch_idx]
                y_b = y_train_t[batch_idx]

                opt.zero_grad()
                logits = self(X_b) # forward pass → logits (not probabilities)
                loss = loss_fn(logits, y_b)
                loss.backward()
                opt.step()

                train_loss += loss.item() * len(batch_idx)

            train_loss /= n_train

            # ---------- Validation with test data ----------
            self.eval()
            with torch.no_grad():
                val_logits = self(X_test_t)
                val_loss = loss_fn(val_logits, y_test_t).item()

                # Apply sigmoid manually because forward returns logits
                probs = torch.sigmoid(val_logits).cpu().numpy()

                y_true = y_test_t.cpu().numpy().astype(int)

                auc = roc_auc_score(y_true, probs)

                # preds for recall use threshold 0.5 on probabilities
                preds = (probs > 0.5).astype(int)
                recall = recall_score(y_true, preds)
            
            # Save best model based on Recall
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch + 1
                torch.save(self.state_dict(), "best_HybridMLP_model.pth")

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"AUC={auc:.4f} | Recall={recall:.4f}"
            )

        print(f"Best model saved with Recall = {best_recall:.4f} in epoch: {best_epoch}")

    def predict(self, X_val):
        """
        Predict returns probabilities
        """
        X_val_t = torch.from_numpy(X_val).to(self.device)
        self.eval()
        with torch.no_grad():
            val_logits = self(X_val_t)
            probs = torch.sigmoid(val_logits).cpu().numpy()

        return probs