# HybridQMedKD

A lightweight hybrid quantum-classical framework for biomedical tabular binary classification under low-resource simulation.

## Objective
We study whether a small quantum layer is useful for low-dimensional biomedical tabular classification when combined with:
- Classical teacher-student knowledge distillation
- PCA-based dimensionality reduction
- Different quantum-layer insertion positions (front / middle / tail)

## Core Research Questions
1. Does knowledge distillation improve hybrid quantum-classical student models?
2. Where should the quantum layer be inserted for best performance-cost trade-off?
3. Under low-resource simulation, is the hybrid student worth the extra cost?

## Dataset
- **Primary:** Breast Cancer Wisconsin (Diagnostic) — 569 samples, 30 continuous features, binary classification, no missing values
- Optional: any low-dimensional biomedical tabular binary classification dataset

## Models
| ID | Model | Distillation | Quantum Layer |
|----|-------|-------------|--------------|
| G1 | Teacher-Classic | No | No |
| G2 | Student-Classic | No | No |
| G3 | Student-Classic-KD | Yes | No |
| G4 | Student-Hybrid-NoKD | No | Yes |
| G5 | Student-Hybrid-KD | Yes | Yes |

## Metrics
- AUC, F1, Accuracy, MCC
- Training time, Inference time
- Qubit count, Circuit depth

## Ablation
- Quantum layer position: front / middle / tail
- PCA dimensions: 4 / 6 / 8
- Distillation temperature T: 2 / 4
- Distillation alpha: 0.3 / 0.5 / 0.7

## Project Structure
```
HybridQMedKD/
├── configs/
├── data/raw/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── trainers/
│   └── utils/
├── outputs/
└── scripts/
```

## Quick Start
```bash
pip install -r requirements.txt
python src/main.py
```

## Notes
- Keep quantum circuits shallow (4 qubits, 1-2 layers) for local simulation feasibility
- PCA / feature selection is applied within each training fold to prevent data leakage
- Goal is practical design insight, not quantum advantage claims
