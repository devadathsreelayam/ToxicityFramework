import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def objective(trial, train_loader, val_loader, gat_model, device):
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5])
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = gat_model(
        in_channels=8,
        edge_channels=4,
        descriptor_dim=217,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_roc = 0
    patience = 10
    trigger = 0

    for epoch in range(50):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        val_roc = roc_auc_score(all_labels, all_preds)

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                break

    return best_val_roc