from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
)
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()

    def _calculate_metrics(self, y_true, y_pred, y_probs):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_probs),
            'pr_auc': average_precision_score(y_true, y_probs),
        }
        return metrics

    def evaluate(self, loader):
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = output.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(data.y.cpu().numpy())

        return self._calculate_metrics(all_labels, all_preds, all_probs)

    def train(self, train_loader, val_loader, optimizer, epochs, patience):
        best_val_roc = 0
        trigger = 0
        best_model = None
        history = []

        for epoch in range(epochs):
            self.model.train()
            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data.y)
                loss.backward()
                optimizer.step()

            val_metrics = self.evaluate(val_loader)
            history.append({"epoch": epoch+1, **val_metrics})
            print(f"Epoch {epoch+1} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

            if val_metrics['roc_auc'] > best_val_roc:
                best_val_roc = val_metrics['roc_auc']
                trigger = 0
                best_model = self.model.state_dict()
            else:
                trigger += 1
                if trigger >= patience:
                    print("Early stopping.")
                    break

        if best_model:
            self.model.load_state_dict(best_model)
        return best_val_roc, history