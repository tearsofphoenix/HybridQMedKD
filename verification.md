# 验证摘要

- 日期：2026-03-30
- 执行者：Codex

## 已完成项

1. `paper/main.tex` 已扩展为论文 v2 版本风格稿。
2. 参考文献总数已从 17 条提升到 30 条。
3. 新增文献已被实质性融入以下部分：
   - Introduction
   - Related Work
   - Method
   - Discussion
   - Limitations
   - Conclusion
4. 已删除若干与本文主题贴合度较弱的旧引用，并用更直接支撑编码、可训练性、泛化和医疗 QML 语境的新文献替换。
5. 已完成本地静态检查：
   - 文献数量正确；
   - 所有引用键已定义；
   - 所有参考文献均被正文使用。
6. 已完成本地 LaTeX 编译，生成 `paper/main.pdf`。

## 环境变更

为恢复 `quantikz` 所需的标准编译依赖，向本地 TeX 用户树安装了以下标准包：

- `xargs`
- `environ`
- `xstring`
- `trimspaces`
- `tikz-cd`

## 风险说明

- 参考文献采用手工维护的 `thebibliography`，结构正确且已编译通过，但投稿前仍建议做一次人工格式审校，以核对期刊缩写、页码和标题大小写是否完全符合目标投稿模板。

## 2026-03-31 增量验证摘要

### 已完成项

1. `paper/main.tex` 已按当前审稿式建议完成定点修订，新增内容覆盖：
   - 统计功效与 fold 数解释；
   - 3-fold placement pilot 与 5-fold companion benchmarks 的区分；
   - deeper ansatz / repeated data re-uploading 的限定性讨论；
   - noiseless statevector 的 simulator-only 定位；
   - 教师置信度、温度 sweep 与软目标信息量解释；
   - Table 2、Figure 1 与 Conclusion 的表述更新。
2. 论文中的功效数字已用本地 `statsmodels` 计算复核：
   - 10-fold WDBC 关键比较 achieved power 约为 `0.43` 和 `0.51`；
   - 5-fold Pima/Heart 关键比较 achieved power 约为 `0.11` 和 `0.17`；
   - 3-/5-/10-fold 达到 80% power 所需 `d_z` 约为 `3.26` / `1.68` / `1.00`。
3. 已完成两轮本地 LaTeX 编译，生成新的 `paper/main.pdf`，当前为 15 页。
4. 已检查 `paper/main.log`，未发现 `Undefined`、`Overfull`、`Underfull` 或通用 `Warning` 告警。

### 风险说明

- 本次属于文稿增强，不包含新增实验运行，因此对 deeper ansatz、data re-uploading、shot noise 和硬件验证的内容均采用限定性讨论而非经验性结论。
- 功效说明以现有比较结果的 achieved power 与 fold-budget sensitivity 为主，属于对当前证据强度的量化解释，不替代未来更大样本或更多 folds 的前瞻性实验设计。

## 2026-03-31 placement 5-fold 补实验验证摘要

### 已完成项

1. 已修改 [scripts/run_cross_dataset_placement.py](/Users/isaac/clawd/research/HybridQMedKD/scripts/run_cross_dataset_placement.py) ，将 `Heart/Pima` placement 检查提升到 `5-fold`。
2. 已完成 `Heart` 与 `Pima` 的 5-fold placement 重跑，并生成新的 fold CSV 与 summary JSON。
3. 已完成 `outputs/tables/revision_statistics.json` 重建，确认 placement 统计已更新到 `n=5`。
4. 已根据新结果更新 [paper/main.tex](/Users/isaac/clawd/research/HybridQMedKD/paper/main.tex) 中的方法说明、placement 结果、讨论、局限性和结论。
5. 已完成两轮本地 LaTeX 编译，新的 `paper/main.pdf` 维持 15 页并通过日志检查。

### 关键结果

- Heart 5-fold placement：
  - `front` AUC `0.5647 ± 0.1159`
  - `middle` AUC `0.8842 ± 0.0604`
  - `tail` AUC `0.8844 ± 0.0723`
  - `middle_vs_tail` AUC 差 `0.0003`，`p=0.9769`
- Pima 5-fold placement：
  - `front` AUC `0.5869 ± 0.0344`
  - `middle` AUC `0.7820 ± 0.0336`
  - `tail` AUC `0.7856 ± 0.0302`
  - `middle_vs_tail` AUC 差 `0.0036`，`p=0.2628`

### 风险说明

- 本次新增实证主要强化了 `front` 明显失效这一结论；即使扩展到 `5-fold`，`middle` 与 `tail` 的差异仍然偏小，尚不足以支撑通用排序结论。
- 运行过程中出现两次单折慢运行：
  - `heart_placement_tail` 第 5 折约 `1057s`
  - `pima_placement_front` 第 1 折约 `886s`
  结果文件均已完整生成，但平均训练时长会受到这两次异常值影响，解读训练成本时应参考 fold 级日志。
