"""
ttlt.py : Test-Time Local Training (TTLT) implementation
"""

import torch
import copy
import torch.nn as nn


def get_k_nearest_neighbors(x_query, X, y, k):
    distances = torch.norm(X - x_query.unsqueeze(0), dim=1)
    knn_idx = torch.topk(distances, k, largest=False).indices
    return X[knn_idx], y[knn_idx], distances[knn_idx]


def ttlt_prediction(
    x_star, model, X, y,
    task="classification",
    k=None, T=100,
    lr=1e-5,
    weight_decay=1e-5
):

    if k is None:
        k = 100 if task == "regression" else 10

    X_neighbors, y_neighbors, dists = get_k_nearest_neighbors(x_star, X, y, k)

    eps = 1e-8
    w = 1.0 / (dists + eps)
    w = w / w.sum()

    f_local = copy.deepcopy(model)
    optimizer = torch.optim.Adam(f_local.parameters(),
                                 lr=lr, weight_decay=weight_decay)

    if task == "regression":
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for _ in range(T):
        optimizer.zero_grad()
        preds = f_local(X_neighbors)
        loss_vec = criterion(preds, y_neighbors)
        loss = (w * loss_vec).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = f_local(x_star.unsqueeze(0))

    if task == "classification":
        return out.argmax(dim=1)
    return out
