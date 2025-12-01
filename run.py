"""
run.py : Entry point — full pipeline execution
"""

from models import MLP
from utils import load_csv, split_data, train_model
from ttlt import ttlt_prediction
import torch

def run_pipeline(csv_path, task):
  X, y = load_csv(csv_path, task)
  X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

  if task == "classification":
    output_dim = int(y.max().item() + 1)
  else:
    output_dim = 1

  model = MLP(
    input_dim=X.shape[1],
    output_dim=output_dim,
    num_layers=2,
    hidden_units=100,
    task=task
    )

  model = train_model(model, X_train, y_train, X_val, y_val)

  preds = []
  for i in range(len(X_test)):
    pred = ttlt_prediction(
      X_test[i], model,
      X_train, y_train,
      task=task,
      k=(100 if task == "regression" else 10),
      T=100
      )
    preds.append(pred)

  preds = torch.stack(preds).squeeze()
  return preds, y_test

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
  parser.add_argument('--task', type=str, choices=['regression', 'classification'], required=True, help='Task type (optional if inferable from data)')
  args = parser.parse_args()

  csv_path = args.data
  task = args.task
  
  preds, y_test = run_pipeline(csv_path, task)
