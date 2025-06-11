# run_gcn.py

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tqdm import tqdm
import yaml
import optuna
import random
import json
from datetime import datetime

from datasets.csv_descriptor_dataset import GeneratedDescToxicDataset
from models.gcn_optuna import GCNModel
from trainer.gcn_trainer import Trainer

# ===================== Utility Functions =====================
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def save_metrics_csv(metrics_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(metrics_dict)
    df.to_csv(path, index=False)

def print_study_progress(study, trial):
    print(f"[Trial {trial.number}] Params: {trial.params} | Val ROC AUC: {trial.value:.4f} | Best ROC AUC: {study.best_value:.4f}")

def get_all_endpoints(data_root, skip_list):
    return [name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and name not in skip_list]

def log_batch_status(log_path, endpoint, status, message=""):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {endpoint}: {status} {message}\n")

def get_retry_parameters(dataset_len, base_trials, retry_config):
    if dataset_len < retry_config['small_dataset_threshold']:
        return retry_config['low_roc']['max_trials']
    elif retry_config['medium_roc']['enabled']:
        return retry_config['medium_roc']['max_trials']
    return base_trials


# ===================== Main Runner with Optuna =====================
def run_optimization(config, dataset, trial_limit=None):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, _ = dataset.get_dataloaders(
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    node_feat_dim = dataset[0].x.shape[1]
    desc_feat_dim = dataset[0].descriptors.shape[1]

    def objective(trial):
        params = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', config['optuna']['hidden_channels']),
            'num_layers': trial.suggest_int('num_layers', *config['optuna']['num_layers']),
            'dropout': trial.suggest_float('dropout', *config['optuna']['dropout']),
            'lr': trial.suggest_float('lr', *config['optuna']['lr'], log=True),
        }

        model = GCNModel(
            in_channels=node_feat_dim,
            descriptor_dim=desc_feat_dim,
            hidden_channels=params['hidden_channels'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        trainer = Trainer(model, device)
        val_roc, preds, targets = trainer.train(train_loader, val_loader, optimizer, config['training']['epochs'], config['training']['patience'], return_preds=True)

        try:
            f1 = f1_score(targets, preds)
        except:
            f1 = 0.0

        score = 0.7 * f1 + 0.3 * val_roc
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trial_limit or config['optuna']['n_trials'], callbacks=[print_study_progress])

    return study.best_params, study


def train_and_evaluate_endpoint(config, endpoint):
    endpoint_path = os.path.join(config['data_root'], endpoint)
    dataset = GeneratedDescToxicDataset(endpoint_path)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    base_trials = config['optuna']['n_trials']
    best_params, study = run_optimization(config, dataset, base_trials)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel(
        in_channels=dataset[0].x.shape[1],
        descriptor_dim=dataset[0].descriptors.shape[1],
        hidden_channels=best_params['hidden_channels'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    trainer = Trainer(model, device)
    _, history = trainer.train(train_loader, val_loader, optimizer, config['training']['epochs'], config['training']['patience'])

    model_save_path = os.path.join(config['output']['model_dir'], f"{endpoint}.pt")

    save_model(model, model_save_path)
    test_metrics = trainer.evaluate(test_loader)
    test_metrics['endpoint'] = endpoint

    history_path = os.path.join(config['output']['history_dir'], f"{endpoint}_train_history.csv")
    param_path = os.path.join(config['output']['history_dir'], f"{endpoint}_optuna_trials.csv")
    save_metrics_csv(history, history_path)
    save_metrics_csv([t.params for t in study.trials], param_path)

    return test_metrics, len(dataset), history, study


# ===================== Batch Training Runner =====================
def train_and_evaluate_all(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results, retry_failed, retry_suboptimal = [], [], []
    root_out = config['output']['root']
    log_path = os.path.join(root_out, 'history', 'batch_run_log.csv')

    endpoints = get_all_endpoints(config['data_root'], config.get('skip_endpoints', []))

    for endpoint in endpoints:
        try:
            metrics, dataset_len, _, _ = train_and_evaluate_endpoint(config, endpoint)
            results.append(metrics)

            if metrics['roc_auc'] < config['retry']['low_roc']['threshold']:
                retry_failed.append((endpoint, dataset_len))
            elif config['retry']['medium_roc']['enabled'] and metrics['roc_auc'] < config['retry']['medium_roc']['threshold']:
                retry_suboptimal.append((endpoint, dataset_len))

            log_batch_status(log_path, endpoint, "SUCCESS")

        except Exception as e:
            log_batch_status(log_path, endpoint, "FAILED", str(e))
            retry_failed.append((endpoint, 0))

    save_metrics_csv(results, os.path.join(root_out, "evaluation_metrics.csv"))

    for endpoint, dataset_len in retry_failed + retry_suboptimal:
        try:
            trials = get_retry_parameters(dataset_len, config['optuna']['n_trials'], config['retry'])
            print(f"\nRetrying {endpoint} with {trials} trials due to suboptimal ROC-AUC")

            endpoint_path = os.path.join(config['data_root'], endpoint)
            dataset = GeneratedDescToxicDataset(endpoint_path)
            best_params, study = run_optimization(config, dataset, trials)

            train_loader, val_loader, test_loader = dataset.get_dataloaders(
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training'].get('num_workers', 0)
            )

            model = GCNModel(
                in_channels=dataset[0].x.shape[1],
                descriptor_dim=dataset[0].descriptors.shape[1],
                hidden_channels=best_params['hidden_channels'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
            trainer = Trainer(model, device)
            _, history = trainer.train(train_loader, val_loader, optimizer, config['training']['epochs'], config['training']['patience'])
            save_model(model, os.path.join(config['output']['model_dir'], f"{endpoint}.pt"))
            test_metrics = trainer.evaluate(test_loader)
            test_metrics['endpoint'] = endpoint
            save_metrics_csv(history, os.path.join(config['output']['history_dir'], f"{endpoint}_train_history.csv"))
            save_metrics_csv([t.params for t in study.trials], os.path.join(config['output']['history_dir'], f"{endpoint}_optuna_trials.csv"))

        except Exception as e:
            print(f"Final retry failed for {endpoint}: {e}")

# ===================== Main =====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gcn_config.yaml')
    args = parser.parse_args()
    train_and_evaluate_all(args.config)