## README.md

# TTLT: Test-Time Local Training

This repository provides a clean PyTorch implementation of **Test-Time Local Training (TTLT)**, a method that improves prediction accuracy during inference by performing local fine-tuning using a query instance’s nearest neighbors.

TTLT can be applied to **any supervised learning task**, including regression and classification, and imposes **no constraints** on neural network architectures or loss functions.

---

## 📁 Project Structure

```
.
├── model.py        # Neural network architectures
├── ttlt.py         # TTLT algorithm implementation
├── utils.py        # Data loading, splitting, neighbor search
├── run.py          # Training + TTLT inference pipeline
├── data.csv        # Example dataset (last column = y)
└── README.md
```

---

## 🚀 Usage

### 1. Install requirements
```
pip install torch numpy pandas scikit-learn
```

### 2. Prepare dataset
- Provide a CSV file where:
  - All columns except the last are features `X`
  - The last column is the target `y`
- Place the file as `data.csv` or modify the path in `run.py`.

---

## 🧠 TTLT: How It Works
Given a query instance `x*`, TTLT:
1. Finds its **k nearest neighbors** in the training dataset.
2. Creates a **local copy** of the global model.
3. Optimizes it for **T steps** using a weighted loss over the neighbors.
4. Predicts by feeding `x*` into the updated model.

This allows the model to **adapt locally** to difficult or rare test instances.

---

## ⚙️ Hyperparameters (per paper settings)
### **Global training**
- Optimizer: Adam
- LR: 1e-3
- Weight decay: 1e-5
- Batch size: 50
- LR decay: ×0.1 after 50 no-improve epochs
- Early stopping: patience 100, max epochs 500

### **TTLT fine‑tuning**
| Task | k (neighbors) | T (steps) | LR | Weight Decay |
|------|----------------|-----------|----|--------------|
| Regression | 100 | 100 | 1e‑5 | 1e‑5 |
| Classification | 10 | 100 | 1e‑5 | 1e‑5 |

---

## ▶️ Running the full pipeline
```
python run.py --data data.csv --task regression
```
Or
```
python run.py --data data.csv --task classification
```

The script will:
1. Load & split the dataset
2. Train global model
3. Perform TTLT inference on the test set
4. Report performance

---

## 📄 Notes
- The model architecture is automatically selected per dataset from:
  - Hidden layers: {1, 2, 3}
  - Hidden units: {50, 100, 200, 500}
  - Activation: `tanh`
- Output layers:
  - Regression: linear unit
  - Classification: softmax layer

---
