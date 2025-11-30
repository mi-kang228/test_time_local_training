import numpy as np
import torch

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # Determine task type: classification if y is integer
    if np.issubdtype(y_raw.dtype, np.integer):
        task = "classification"
        y = torch.tensor(y_raw, dtype=torch.long)
    else:
        task = "regression"
        y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

    X = torch.tensor(X, dtype=torch.float32)
    return X, y, task

def get_k_nearest_neighbors(x_query, X_trn, y_trn, k):
    distances = torch.norm(X_trn - x_query.unsqueeze(0), dim=1)
    knn_idx = torch.topk(distances, k, largest=False).indices
    return X_trn[knn_idx], y_trn[knn_idx], distances[knn_idx]
