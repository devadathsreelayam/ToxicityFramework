import os
import random

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

from datasets.csv_descriptor_dataset import GeneratedDescToxicDataset
from hyper.optimisation_for_gat import objective
from models.gat_optuna import GATOptunaModel


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.BCELoss()

    def train(self, optimizer, epochs=100, patience=10, verbose=True):
        best_val_roc = 0
        trigger = 0
        min_epoch = 30
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for data in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs}', disable=not verbose):
                data = data.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, data.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(data.y.detach().cpu().numpy())

            train_roc = roc_auc_score(train_labels, train_preds)
            avg_train_loss = train_loss / len(self.train_loader)

            # Validation phase
            metrics = self.evaluate(self.val_loader)
            val_loss = metrics['loss']
            val_roc = metrics['roc_auc']

            if verbose:
                print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train ROC: {train_roc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val ROC: {val_roc:.4f}')

            # Early stopping
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                trigger = 0
                best_model_state = self.model.state_dict()
            else:
                trigger += 1
                if trigger >= patience and epoch > min_epoch:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1}')
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_val_roc

    def evaluate(self, loader, threshold=0.5):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, data.y)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        roc = roc_auc_score(all_labels, all_preds)

        # Convert probabilities to binary predictions
        binary_preds = [1 if p >= threshold else 0 for p in all_preds]

        metrics = {
            'loss': avg_loss,
            'roc_auc': roc,
            'accuracy': accuracy_score(all_labels, binary_preds),
            'precision': precision_score(all_labels, binary_preds),
            'recall': recall_score(all_labels, binary_preds),
            'f1': f1_score(all_labels, binary_preds)
        }

        return metrics

    def test(self, threshold=0.5):
        return self.evaluate(self.test_loader, threshold)


def run_optuna_study(train_loader, val_loader, device='cuda', n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, GATOptunaModel, device),
        n_trials=n_trials
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_params


def train_final_model(train_loader, val_loader, test_loader, best_params, device='cuda'):
    model = GATOptunaModel(
        in_channels=8,
        edge_channels=4,
        descriptor_dim=len(train_loader.dataset[0].descriptors[0]),
        hidden_channels=best_params['hidden_channels'],
        num_layers=best_params['num_layers'],
        heads=best_params['heads'],
        dropout=best_params['dropout']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    trainer = Trainer(model, train_loader, val_loader, test_loader, device)

    best_val_roc = trainer.train(optimizer)
    test_metrics = trainer.test()

    print("\nFinal Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    return model, test_metrics


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load your dataset
    endpoint_dir = "data/generated_desc/skin_sens"  # Replace with your actual path
    dataset = GeneratedDescToxicDataset(endpoint_dir)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    # Run Optuna optimization
    print("Starting Optuna optimization...")
    best_params = run_optuna_study(train_loader, val_loader, device)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    model, test_metrics = train_final_model(train_loader, val_loader, test_loader, best_params, device)

    # You can save the model if needed
    torch.save(model.state_dict(), f'saved_models/gat_optimised/{os.path.basename(endpoint_dir)}.pt')


if __name__ == "__main__":
    main()