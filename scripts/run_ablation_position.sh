#!/bin/bash
# Ablation: quantum layer position
for pos in front middle tail; do
  echo "Position: $pos"
  python -c "
from src.utils.seed import set_seed
from src.trainers.train_teacher import train_teacher_cv
from src.trainers.train_student import run_student_cv
set_seed(42)
teacher = train_teacher_cv('data/raw/wdbc.csv', pca_dim=4)
run_student_cv('data/raw/wdbc.csv', teacher_fold_outputs=teacher, model_type='hybrid', use_kd=True, alpha=0.5, T=2.0, quantum_position='$pos', exp_name='ablation_pos_$pos')
"
done
