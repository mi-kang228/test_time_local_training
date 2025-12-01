"""
run.py : Entry point — full pipeline execution
"""

from model import MLP
from utils import load_csv, split_and_scale, train_model
from ttlt import ttlt_prediction
import torch

def run_pipeline(csv_path, task):
  X, y = load_csv(csv_path)
  X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_scaler, y_scaler = split_and_scale(X, y, task)

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

  model = train_model(model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

  preds = []
  for i in range(len(X_test)):
    pred = ttlt_prediction(
      X_test_scaled[i], model,
      X_train_scaled, y_train_scaled,
      task=task,
      k=(100 if task == "regression" else 10),
      T=100
      )
    preds.append(pred)

    preds = torch.stack(preds).squeeze()

    preds_inv_scaled = y_scaler.inverse_transform(preds.view(-1, 1).cpu().numpy()).squeeze()
    y_test_inv_scaled = y_scaler.inverse_transform(y_test_scaled.view(-1, 1).cpu().numpy()).squeeze()

    results_df = pd.DataFrame({
        'Predictions': preds_inv_scaled,
        'Ground truth': y_test_inv_scaled
    })

    output_file = "predictions_results.csv"
    results_df.to_csv(output_file, index=False)

    return results_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'], required=True, help='Task type (optional if inferable from data)')
    args = parser.parse_args()

    csv_path = args.data
    task = args.task
    
    results_df = run_pipeline(csv_path, task)
    print(f"Results saved to {output_file}")
