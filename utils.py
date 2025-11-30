import numpy as np
import torch
import torch.nn as nn

class RegDataset():

    def __init__(self, X, Y):
    
        self.X = X
        self.Y = Y

    def __len__(self):
    
        return len(self.X)

    def __getitem__(self, idx):
    
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
    
        return x, y


class RegNN(nn.Module):

    def __init__(self, dim_x, n_layers=3, dim_h=100, prob_dropout=0):
        
        super(RegNN, self).__init__()

        layers = [nn.Linear(dim_x, dim_h), nn.Tanh(), nn.Dropout(prob_dropout)]
        for _ in range(n_layers-1):
            layers += [nn.Linear(dim_h, dim_h), nn.Tanh(), nn.Dropout(prob_dropout)]
        
        layers += [nn.Linear(dim_h, 1)]

        self.predict = nn.Sequential(*layers)                           
                               
    def forward(self, x):

        y_hat = self.predict(x)

        return y_hat

      pass

def get_k_nearest_neighbors(x_query, X_trn, y_trn, k):
    distances = torch.norm(X_trn - x_query.unsqueeze(0), dim=1)
    knn_idx = torch.topk(distances, k, largest=False).indices
    return X_trn[knn_idx], y_trn[knn_idx], distances[knn_idx]
