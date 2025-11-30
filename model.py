import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,
                 num_layers=2, hidden_units=100,
                 task="classification"):
        super().__init__()

        layers = []
        last_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_units))
            layers.append(nn.Tanh())
            last_dim = hidden_units

        layers.append(nn.Linear(last_dim, output_dim))

        if task == "classification":
            layers.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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
