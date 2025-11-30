"""
utils.py : CSV loading, splitting, and model training utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import copy

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    if np.issubdtype(y_raw.dtype, np.integer):
        task = "classification"
        y = torch.tensor(y_raw, dtype=torch.long)
    else:
        task = "regression"
        y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

    X = torch.tensor(X, dtype=torch.float32)
    return X, y, task


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, X_train, y_train,
                X_val, y_val,
                lr=1e-3,
                weight_decay=1e-5,
                batch_size=50,
                patience=100,
                max_epochs=500):

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50
    )

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss() if y_train.dim() == 2 else nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter > patience:
            break

    model.load_state_dict(best_state)
    return model
