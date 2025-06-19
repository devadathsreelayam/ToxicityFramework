import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn import AttentiveFP
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, log_loss
)

# === Descriptor List ===
desc_list_12 = [
    'MolWt', 'HeavyAtomCount', 'NumValenceElectrons',
    'NumAromaticRings', 'FractionCSP3', 'TPSA',
    'MolLogP', 'HallKierAlpha', 'NumHAcceptors',
    'NumHDonors', 'RingCount', 'MolMR'
]

# === EarlyStopping Class ===
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# === Evaluation Function ===
def evaluate_predictions(y_true, y_probs):
    y_pred = (np.array(y_probs) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-7)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0,
        'true_negatives': tn, 'false_positives': fp,
        'false_negatives': fn, 'true_positives': tp,
        'support_negative': tn + fp, 'support_positive': fn + tp,
        'precision_negative': precision_neg,
        'recall_negative': recall_neg,
        'f1_negative': f1_neg,
        'loss': log_loss(y_true, y_probs)
    }
    return metrics

# === Model ===
class SimpleAttentiveFP(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, descriptor_size=0):
        super(SimpleAttentiveFP, self).__init__()
        self.has_descriptors = descriptor_size > 0

        self.attentivefp = AttentiveFP(
            in_channels=node_feat_size,
            hidden_channels=128,
            out_channels=128,
            edge_dim=edge_feat_size,
            num_layers=3,
            num_timesteps=3,
            dropout=0.2
        )

        if self.has_descriptors:
            self.descriptor_processor = nn.Linear(descriptor_size, 32)
            self.final_predictor = nn.Linear(128 + 32, 1)
        else:
            self.final_predictor = nn.Linear(128, 1)

    def forward(self, data):
        x_graph = self.attentivefp(data.x, data.edge_index, data.edge_attr, data.batch)
        if self.has_descriptors and hasattr(data, 'descriptors'):
            x_desc = self.descriptor_processor(data.descriptors.squeeze(1))
            x = torch.cat([x_graph, x_desc], dim=1)
        else:
            x = x_graph
        return torch.sigmoid(self.final_predictor(x)).squeeze(-1)

# === Training Loop ===
def train_model(train_loader, val_loader, model, optimizer, criterion, early_stopper, device):
    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_probs, y_true = [], []
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                probs = model(batch).detach().cpu().numpy()
                labels = batch.y.cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(labels)

        val_metrics = evaluate_predictions(y_true, y_probs)
        val_score = val_metrics['roc_auc']
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Val ROC AUC: {val_score:.4f}")

        early_stopper(val_score, model)
        if early_stopper.early_stop:
            print("[!] Early stopping")
            break

# === Save Evaluation Results ===
def save_results_to_csv(results_dict, output_csv_all, output_csv_test):
    all_rows, test_rows = [], []
    for endpoint, result in results_dict.items():
        for split, metrics in result.items():
            row = {'endpoint': endpoint, 'split': split}
            row.update(metrics)
            all_rows.append(row)
            if split == 'test':
                test_rows.append(row)
    pd.DataFrame(all_rows).to_csv(output_csv_all, index=False)
    pd.DataFrame(test_rows).to_csv(output_csv_test, index=False)
    print(f"[✓] Saved all evaluations to:\n  {output_csv_all}\n  {output_csv_test}")


# === Main Runner for All Endpoints ===
from datasets.csv_descriptor_dataset import GeneratedDescToxicDataset  # adjust as needed

def run_training_for_all_endpoints(dataset_root, save_dir, desc_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("evaluation_logs", exist_ok=True)

    all_results = {}
    for endpoint in os.listdir(dataset_root):
        endpoint_dir = os.path.join(dataset_root, endpoint)
        if not os.path.isdir(endpoint_dir):
            continue

        print(f"\n[●] Processing: {endpoint}")
        dataset = GeneratedDescToxicDataset(endpoint_dir, descriptor_names=desc_list)
        train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=64)

        sample_batch = next(iter(train_loader))
        node_feat = sample_batch.x.shape[1]
        edge_feat = sample_batch.edge_attr.shape[1] if sample_batch.edge_attr is not None else 0
        desc_feat = sample_batch.descriptors.shape[1]

        model = SimpleAttentiveFP(node_feat, edge_feat, desc_feat).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model_path = os.path.join(save_dir, f"best_model_{endpoint}.pt")
        early_stopper = EarlyStopping(patience=10, delta=0.001, path=model_path)

        train_model(train_loader, val_loader, model, optimizer, criterion, early_stopper, device)

        # Load best model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        def eval_split(loader):
            y_probs, y_true = [], []
            for batch in loader:
                batch = batch.to(device)
                with torch.no_grad():
                    probs = model(batch).detach().cpu().numpy()
                    labels = batch.y.cpu().numpy()
                y_probs.extend(probs)
                y_true.extend(labels)
            return evaluate_predictions(y_true, y_probs)

        result = {
            'train': eval_split(train_loader),
            'val': eval_split(val_loader),
            'test': eval_split(test_loader)
        }

        all_results[endpoint] = result

    save_results_to_csv(
        results_dict=all_results,
        output_csv_all="evaluation_logs/attentivefp_full_evaluation.csv",
        output_csv_test="evaluation_logs/attentivefp_test_only.csv"
    )

# === ENTRY POINT ===
if __name__ == "__main__":
    dataset_root = "data/generated_desc"
    save_dir = "saved_models/attentive_fp"
    os.makedirs(save_dir, exist_ok=True)

    run_training_for_all_endpoints(
        dataset_root=dataset_root,
        save_dir=save_dir,
        desc_list=desc_list_12
    )
