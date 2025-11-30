import torch
import torch.nn as nn
import copy

def ttlt_prediction(
    x_star, model, X, y,
    task="classification",
    k=None, T=100,
    lr=1e-5,
    weight_decay=1e-5
):

    if k is None:
        k = 100 if task == "regression" else 10

    # k nearest neighbors
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

    # T iterations
    for _ in range(T):
        optimizer.zero_grad()
        preds = f_local(X_neighbors)
        loss_vec = criterion(preds, y_neighbors)
        loss = (w * loss_vec).sum()
        loss.backward()
        optimizer.step()

    # final prediction
    with torch.no_grad():
        out = f_local(x_star.unsqueeze(0))

    if task == "classification":
        return out.argmax(dim=1)
    return out
