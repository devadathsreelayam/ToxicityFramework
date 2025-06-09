import os
import torch
import optuna
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from datasets.csv_descriptor_dataset import GeneratedDescToxicDataset
from models.gat_optuna import GATOptunaModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GATTrainer:
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
            'true_negatives': tn, 'false_positives': fp,
            'false_negatives': fn, 'true_positives': tp,
            'support_negative': tn + fp, 'support_positive': fn + tp
        }
        precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics.update({
            'precision_negative': precision_neg,
            'recall_negative': recall_neg,
            'f1_negative': 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-7)
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

    def train(self, train_loader, val_loader, optimizer, epochs=100, patience=10, verbose=True):
        best_val_roc = 0
        trigger = 0
        best_model_state = None
        history = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_probs, all_preds, all_labels = [], [], []
            for data in train_loader:
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
            history.append({"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

            if verbose:
                print(f"Epoch {epoch + 1:03d} | "
                      f"Train ROC AUC: {train_metrics['roc_auc']:.4f} | "
                      f"Val ROC AUC: {val_metrics['roc_auc']:.4f} | "
                      f"Loss: {train_metrics['loss']:.4f}")

            if val_metrics['roc_auc'] > best_val_roc:
                best_val_roc = val_metrics['roc_auc']
                trigger = 0
                best_model_state = self.model.state_dict()
            else:
                trigger += 1
                if trigger >= patience:
                    break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return best_val_roc, pd.DataFrame(history)


def print_study_progress(study, trial):
    print(
        f"[Trial {trial.number}] "
        f"Params: {trial.params} | "
        f"Val ROC AUC: {trial.value:.4f} | "
        f"Best ROC AUC: {study.best_value:.4f} "
        f"(Trial {study.best_trial.number})"
    )


def run_study(endpoint_dir, output_dir='saved_models/gat_optimised', n_trials=50):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    endpoint = os.path.basename(endpoint_dir)
    print(f"Running optimisation for: {endpoint}")
    dataset = GeneratedDescToxicDataset(endpoint_dir)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    def objective(trial):
        params = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'heads': trial.suggest_categorical('heads', [2, 4, 8]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        }
        model = GATOptunaModel(
            in_channels=8,
            edge_channels=4,
            descriptor_dim=len(train_loader.dataset[0].descriptors[0]),
            hidden_channels=params['hidden_channels'],
            num_layers=params['num_layers'],
            heads=params['heads'],
            dropout=params['dropout']
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        trainer = GATTrainer(model, device)
        val_roc, _ = trainer.train(train_loader, val_loader, optimizer, epochs=50, patience=7)
        trial.set_user_attr("val_roc", val_roc)
        trial.set_user_attr("params", params)
        return val_roc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, callbacks=[print_study_progress])

    print(f"Completed optimisation for {endpoint}. Best ROC: {study.best_value:.4f}")
    best_params = study.best_trial.user_attrs['params']

    # Final training with best params
    final_model = GATOptunaModel(
        in_channels=8,
        edge_channels=4,
        descriptor_dim=len(train_loader.dataset[0].descriptors[0]),
        hidden_channels=best_params['hidden_channels'],
        num_layers=best_params['num_layers'],
        heads=best_params['heads'],
        dropout=best_params['dropout']
    ).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
    trainer = GATTrainer(final_model, device)
    val_roc, history_df = trainer.train(train_loader, val_loader, optimizer, epochs=100, patience=10)

    # Save model
    model_path = os.path.join(output_dir, f'{endpoint}.pt')
    torch.save(final_model.state_dict(), model_path)

    # Save history
    hist_path = os.path.join(output_dir, f'{endpoint}_optimisation_history.csv')
    history_df.to_csv(hist_path, index=False)

    # Evaluation
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    test_metrics = trainer.evaluate(test_loader)

    metrics = {
        'endpoint': endpoint,
        **{f'train_{k}': v for k, v in train_metrics.items()},
        **{f'val_{k}': v for k, v in val_metrics.items()},
        **{f'test_{k}': v for k, v in test_metrics.items()},
        **best_params
    }

    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        updated = pd.concat([existing, pd.DataFrame([metrics])], ignore_index=True)
        updated.to_csv(metrics_path, index=False)
    else:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)



def batch_run(endpoint_dirs, output_dir='saved_models/gat_optimised', n_trials=50):
    for endpoint in endpoint_dirs:
        print(f"Running optimisation for: {os.path.basename(endpoint)}")
        run_study(endpoint, output_dir, n_trials)
    print("\nAll optimisations completed.")


if __name__ == "__main__":
    endpoints = ['data/generated_desc/skin_sens']
    batch_run(endpoints, n_trials=5)