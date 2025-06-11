import os
import torch
import optuna
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
from models.gat_optuna import GATOptunaModel
import utils


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_study_progress(study, trial):
    print(
        f"[Trial {trial.number}] "
        f"Params: {trial.params} | "
        f"Val ROC AUC: {trial.value:.4f} | "
        f"Best ROC AUC: {study.best_value:.4f} "
        f"(Trial {study.best_trial.number})"
    )


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
            'pr_auc': average_precision_score(y_true, y_probs),  # Added PR-AUC
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

        print("\n" + "=" * 60)
        print(f"🚀 Starting Training - Max Epochs: {epochs}, Patience: {patience}")
        print("=" * 60 + "\n")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_probs, all_preds, all_labels = [], [], []

            # Training loop with progress bar
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

            # Calculate metrics
            train_metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
            train_metrics['loss'] = total_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)

            # Store history
            history.append({
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1:03d}/{epochs} Results:")
            print(
                f"   Train Loss: {train_metrics['loss']:.4f} | ROC-AUC: {train_metrics['roc_auc']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}")
            print(
                f"   Val Loss: {val_metrics['loss']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"   Early Stop Counter: {trigger}/{patience}")

            # Early stopping logic
            if val_metrics['roc_auc'] > best_val_roc:
                best_val_roc = val_metrics['roc_auc']
                trigger = 0
                best_model_state = self.model.state_dict()
                print(f"🎉 New Best Validation ROC-AUC: {best_val_roc:.4f}")
            else:
                trigger += 1
                if trigger >= patience:
                    print(f"🛑 Early Stopping Triggered (No improvement for {patience} epochs)")
                    break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print("\n" + "=" * 60)
        print(f"🏁 Training Complete - Best Val ROC-AUC: {best_val_roc:.4f}")
        print("=" * 60 + "\n")

        return best_val_roc, pd.DataFrame(history)


def run_study(endpoint_dir, output_dir='saved_models/gat_optimised', n_trials=50):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    endpoint = os.path.basename(endpoint_dir)
    print("\n" + "=" * 60)
    print(f"🔍 Beginning Optimization for: {endpoint}")
    print("=" * 60 + "\n")

    # Load dataset
    print("📦 Loading and processing dataset...")
    dataset = GeneratedDescToxicDataset(endpoint_dir, descriptor_names=utils.desc_list_12)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")

    def objective(trial):
        print("\n" + "-" * 50)
        print(f"🔧 Trial {trial.number + 1}/{n_trials} - Testing New Hyperparameters")
        print("-" * 50)

        params = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'heads': trial.suggest_categorical('heads', [2, 4, 8]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        }

        # Store params in trial user attribute
        trial.set_user_attr("params", params)
        print(f"   Hyperparameters: {params}")

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
        val_roc, history_df = trainer.train(train_loader, val_loader, optimizer, epochs=30, patience=5)

        # Save trial history
        trial_history_path = os.path.join(output_dir, f'{endpoint}_optimisation_history.csv')
        if os.path.exists(trial_history_path):
            existing_df = pd.read_csv(trial_history_path)
            history_df = pd.concat([existing_df, history_df])
        history_df.to_csv(trial_history_path, index=False)

        return val_roc

    # Run optimization
    print("\n" + "=" * 60)
    print("🧪 Starting Hyperparameter Optimization")
    print("=" * 60 + "\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, callbacks=[print_study_progress])

    print("\n" + "=" * 60)
    print(f"✅ Completed Optimization for {endpoint}")
    print(f"🏆 Best ROC-AUC: {study.best_value:.4f}")
    print("=" * 60 + "\n")

    best_params = study.best_trial.user_attrs['params']

    # Final training with best params
    print("\n" + "="*60)
    print(f"✅ Completed Optimization for {endpoint}")
    print(f"⚙️ Best Parameters: {best_params}")
    print("="*60 + "\n")

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
    val_roc, history_df = trainer.train(train_loader, val_loader, optimizer, epochs=100, patience=15)

    # Save final model and training history
    print("\n" + "=" * 60)
    print("💾 Saving Final Model and Training History")
    print("=" * 60 + "\n")

    model_path = os.path.join(output_dir, 'models', f'{endpoint}.pt')
    torch.save(final_model.state_dict(), model_path)

    hist_path = os.path.join(output_dir, f'{endpoint}_training_history.csv')
    history_df.to_csv(hist_path, index=False)
    print(f"   Model saved to: {model_path}")
    print(f"   Training history saved to: {hist_path}")

    # Evaluation
    print("\n" + "=" * 60)
    print("🧪 Evaluating Final Model on All Splits")
    print("=" * 60 + "\n")

    splits = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    metrics = []
    for split_name, loader in splits.items():
        print(f"   Evaluating on {split_name} set...")
        split_metrics = trainer.evaluate(loader)
        split_metrics.update({
            'endpoint': endpoint,
            'split': split_name,
            **best_params
        })
        metrics.append(split_metrics)

        # Print key metrics
        print(
            f"     ROC-AUC: {split_metrics['roc_auc']:.4f} | PR-AUC: {split_metrics['pr_auc']:.4f} | F1: {split_metrics['f1']:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    # Reorder columns to have endpoint and split first
    cols = ['endpoint', 'split'] + [c for c in metrics_df.columns if c not in ['endpoint', 'split']]
    metrics_df = metrics_df[cols]

    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        updated = pd.concat([existing, metrics_df], ignore_index=True)
        updated.to_csv(metrics_path, index=False)
    else:
        metrics_df.to_csv(metrics_path, index=False)

    print(f"\n📊 Evaluation metrics saved to: {metrics_path}")


def batch_run(endpoint_dirs, output_dir='saved_models/gat_optimised', n_trials=50):
    print("\n" + "=" * 60)
    print("🌟 Starting Batch Optimization Process")
    print("=" * 60 + "\n")

    for endpoint in endpoint_dirs:
        print("\n" + "=" * 60)
        print(f"🔍 Processing Endpoint: {os.path.basename(endpoint)}")
        print("=" * 60 + "\n")

        try:
            run_study(endpoint, output_dir, n_trials)
        except Exception as e:
            print(f'Error with endpoint: {endpoint}')
            print(str(e))
            continue

    print("\n" + "=" * 60)
    print("🎉 All Optimizations Completed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    ran_endpoints = [
        'ames', 'avian_tox', 'bee_tox', 'biodegradation', 'carcinogenicity',
        'crustacean', 'eye_corrosion', 'eye_irritation', 'h_ht', 'herg',
        'hia', 'micronucleus_tox', 'nr_gr', 'nr_tr', 'respiratory_tox', 'skin_sens'
    ]
    
    BASE_DIR = 'data/generated_desc'
    all_endpoints = os.listdir(BASE_DIR)
    
    endpoints = [f'{BASE_DIR}/{endpoint}' for endpoint in all_endpoints if endpoint not in ran_endpoints]
    output_dir = 'saved_models/gat_optimised'
    models_output_dir = 'saved_models/gat_optimised'
    os.makedirs(models_output_dir, exist_ok=True)
    
    batch_run(endpoints, n_trials=10, output_dir='saved_models/gat_optimised')