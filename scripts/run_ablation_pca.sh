#!/bin/bash
# Ablation: PCA dimensions
for dim in 4 6 8; do
  echo "PCA dim: $dim"
  python -c "
from src.utils.seed import set_seed
from src.trainers.train_teacher import train_teacher_cv
from src.trainers.train_student import run_student_cv
set_seed(42)
teacher = train_teacher_cv('data/raw/wdbc.csv', pca_dim=$dim)
run_student_cv('data/raw/wdbc.csv', teacher_fold_outputs=teacher, model_type='hybrid', use_kd=True, pca_dim=$dim, exp_name='ablation_pca_$dim')
"
done
