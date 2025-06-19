import os
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, average_precision_score
)

from datasets.csv_descriptor_dataset import GeneratedDescToxicDataset
from models.mpnn_model import MPNNModel  # Make sure this is defined as we discussed
import utils


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MPNNTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()

    def _calculate_metrics(self, y_true, y_pred, y_probs):
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        valid = ~np.isnan(y_probs)
        y_true = y_true[valid]
        y_probs = y_probs[valid]
        y_pred = np.array(y_pred)[valid]

        labels_present = list(np.unique(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if labels_present == [0]:
                tn = cm[0][0]
            elif labels_present == [1]:
                tp = cm[0][0]

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0,
            'true_negatives': tn, 'false_positives': fp,
            'false_negatives': fn, 'true_positives': tp,
            'support_negative': tn + fp, 'support_positive': fn + tp
        }

        precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-7)
        metrics.update({
            'precision_negative': precision_neg,
            'recall_negative': recall_neg,
            'f1_negative': f1_neg
        })

        return metrics

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, data.y)
                probs = out.cpu().numpy()
                preds = (probs > 0.5).astype(int)

                total_loss += loss.item()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(data.y.cpu().numpy())

        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def train(self, train_loader, val_loader, optimizer, epochs=30, patience=5):
        best_val_pr = 0
        trigger = 0
        best_model_state = None
        history = []

        print("\n" + "=" * 60)
        print(f"🚀 Starting MPNN Training - Epochs: {epochs}, Patience: {patience}")
        print("=" * 60 + "\n")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_probs, all_preds, all_labels = [], [], []

            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, data.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                probs = out.detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(data.y.cpu().numpy())

            train_metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
            train_metrics['loss'] = total_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)

            history.append({
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            print(f"\nEpoch {epoch + 1}: Train PR-AUC: {train_metrics['pr_auc']:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f}")

            if val_metrics['pr_auc'] > best_val_pr:
                best_val_pr = val_metrics['pr_auc']
                best_model_state = self.model.state_dict()
                trigger = 0
                print("  🎉 New Best Model Found!")
            else:
                trigger += 1
                if trigger >= patience:
                    print("  🚩 Early stopping triggered")
                    break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return best_val_pr, pd.DataFrame(history)


def run_mpnn_training(endpoint_dir, output_dir='saved_models/mpnn', epoch=100, patience=10):
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    endpoint = os.path.basename(endpoint_dir)
    dataset = GeneratedDescToxicDataset(endpoint_dir, descriptor_names=utils.desc_list_12)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    model = MPNNModel(
        in_channels=8,
        edge_channels=4,
        descriptor_dim=len(train_loader.dataset[0].descriptors[0]),
        hidden_channels=64,
        num_layers=3,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = MPNNTrainer(model, device)
    best_val_pr, history_df = trainer.train(train_loader, val_loader, optimizer, epochs=epoch, patience=patience)

    # Save model
    model_path = os.path.join(output_dir, 'models', f'{endpoint}.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Save history
    hist_path = os.path.join(output_dir, 'history', f'{endpoint}_training_history.csv')
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    history_df.to_csv(hist_path, index=False)

    print(f"\nModel saved to: {model_path}")
    print(f"Training history saved to: {hist_path}")

    # Evaluate all splits
    eval_metrics = []
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        metrics = trainer.evaluate(loader)
        print(f"{name.upper()} - ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}")
        metrics.update({
            'endpoint': endpoint,
            'split': name,
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'lr': 1e-3
        })
        eval_metrics.append(metrics)

    eval_df = pd.DataFrame(eval_metrics)
    cols = ['endpoint', 'split'] + [c for c in eval_df.columns if c not in ['endpoint', 'split']]
    eval_df = eval_df[cols]

    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        updated = pd.concat([existing, eval_df], ignore_index=True)
        updated.to_csv(metrics_path, index=False)
    else:
        eval_df.to_csv(metrics_path, index=False)

    print(f"\n📊 Evaluation metrics saved to: {metrics_path}")


if __name__ == '__main__':
    base_data_dir = 'data/generated_desc'
    os.makedirs('saved_models/mpnn/models', exist_ok=True)
    os.makedirs('saved_models/mpnn/history', exist_ok=True)

    for endpoint_dir in os.listdir(base_data_dir):
        run_mpnn_training(os.path.join(base_data_dir, endpoint_dir), epoch=100, patience=10)
