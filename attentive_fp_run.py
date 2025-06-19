import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.optim import Adam
from torch_geometric.nn import AttentiveFP
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, log_loss
)
from datasets.csv_descriptor_dataset import GeneratedDescToxicDataset

# === Evaluation Function ===
def evaluate_model(y_true, y_pred_probs, threshold=0.5):
    import numpy as np
    y_pred_probs = np.asarray(y_pred_probs).flatten()
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision_score_val = precision_score(y_true, y_pred, zero_division=0)
    recall_score_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_probs)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    recall, precision = zip(*sorted(zip(recall, precision)))
    pr_auc = auc(recall, precision)

    loss = log_loss(y_true, y_pred_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    support_neg, support_pos = tn + fp, tp + fn

    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0

    return {
        "accuracy": acc,
        "precision": precision_score_val,
        "recall": recall_score_val,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "support_negative": support_neg,
        "support_positive": support_pos,
        "precision_negative": prec_neg,
        "recall_negative": rec_neg,
        "f1_negative": f1_neg,
        "loss": loss
    }

# === Model ===
class SimpleAttentiveFP(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, descriptor_size=0, hidden_channels=128, out_channels=128):
        super(SimpleAttentiveFP, self).__init__()
        self.has_descriptors = descriptor_size > 0

        self.attentivefp = AttentiveFP(
            in_channels=node_feat_size,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_feat_size,
            num_layers=3,
            num_timesteps=3,
            dropout=0.2
        )

        if self.has_descriptors:
            self.descriptor_processor = nn.Linear(descriptor_size, 32)
            self.final_predictor = nn.Linear(out_channels + 32, 1)
        else:
            self.final_predictor = nn.Linear(out_channels, 1)

    def forward(self, data):
        x_graph = self.attentivefp(data.x, data.edge_index, data.edge_attr, data.batch)
        if self.has_descriptors and hasattr(data, 'descriptors'):
            x_desc = self.descriptor_processor(data.descriptors.squeeze(1))
            x = torch.cat([x_graph, x_desc], dim=1)
        else:
            x = x_graph
        return torch.sigmoid(self.final_predictor(x)).squeeze(-1)

# === Training Function ===
def train_attentivefp_model(endpoint_dir, save_path, descriptor_list, max_epochs=50, patience=7, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GeneratedDescToxicDataset(endpoint_dir, descriptor_names=descriptor_list)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=64)

    sample_batch = next(iter(train_loader))
    node_dim = sample_batch.x.shape[1]
    edge_dim = sample_batch.edge_attr.shape[1] if sample_batch.edge_attr is not None else 0
    desc_dim = sample_batch.descriptors.shape[1]

    model = SimpleAttentiveFP(node_dim, edge_dim, desc_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_val_auc = 0
    patience_counter = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                probs = model(batch).detach().cpu().numpy().flatten().tolist()
                labels = batch.y.view(-1).cpu().numpy().flatten().tolist()
                all_probs.extend(probs)
                all_labels.extend(labels)

        val_result = evaluate_model(all_labels, all_probs)
        print(f"[{epoch+1}] Loss: {epoch_loss:.4f} | Val ROC-AUC: {val_result['roc_auc']:.4f}")

        if val_result["roc_auc"] > best_val_auc:
            best_val_auc = val_result["roc_auc"]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[!] Early stopping triggered at epoch {epoch+1}")
            break

    return model, val_result

# === Run All Endpoints ===
def train_all_attentivefp(parent_data_folder, output_model_folder, descriptor_list, log_csv_path):
    os.makedirs(output_model_folder, exist_ok=True)
    logs = []

    for endpoint in os.listdir(parent_data_folder):
        endpoint_dir = os.path.join(parent_data_folder, endpoint)
        if not os.path.isdir(endpoint_dir):
            continue

        save_path = os.path.join(output_model_folder, f"best_model_{endpoint}.pt")
        print(f"\n[*] Training on {endpoint}")
        try:
            _, val_metrics = train_attentivefp_model(endpoint_dir, save_path, descriptor_list)
            val_metrics.update({"endpoint": endpoint})
            logs.append(val_metrics)
        except Exception as e:
            print(f"[!] Failed on {endpoint}: {e}")

    df = pd.DataFrame(logs)
    df.to_csv(log_csv_path, index=False)
    print(f"\n[✓] Logs saved to: {log_csv_path}")

# === Descriptor List ===
desc_list_12 = [
    'MolWt', 'HeavyAtomCount', 'NumValenceElectrons',
    'NumAromaticRings', 'FractionCSP3', 'TPSA',
    'MolLogP', 'HallKierAlpha', 'NumHAcceptors',
    'NumHDonors', 'RingCount', 'MolMR'
]

# === Entry Point ===
if __name__ == "__main__":
    parent_data_folder = "../data/generated_desc"
    output_model_folder = "saved_models/attentive_fp"
    log_csv_path = "evaluation_logs/attentivefp_val_results.csv"

    train_all_attentivefp(parent_data_folder, output_model_folder, desc_list_12, log_csv_path)
