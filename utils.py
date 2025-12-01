"""
utils.py : CSV loading, splitting, and model training utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import copy

def load_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    
    return X, y

def data_preprocess(X, y, task):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_scaled = x_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(X_train_scaled, y_train_scaled, test_size=0.25, random_state=0)

    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

    if task == "classification":
        y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.long)
        y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.long)
        y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.long)
        
    elif task == "regression":
        y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
        y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1)
        y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_scaler, y_scaler
    
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
