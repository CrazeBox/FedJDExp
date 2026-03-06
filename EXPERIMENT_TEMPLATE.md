# FedJD Experiment Template

## 1) Naming Convention
Use one run-id per execution:
`{date}_{exp_group}_{alpha}_{method}_{mode}_{beta}_s{seed}`

Example:
`20260228_E3_a0.1_ssjd_loss_grad_b0.5_s1`

## 2) Recommended Folder Layout
- `experiments/experiment_grid.csv`: run plan
- `results/raw/`: raw logs (one subfolder per run-id)
- `results/records.csv`: one row per completed run
- `results/summary_by_setting.csv`: aggregated mean/std table

## 3) Minimal Reporting Fields
Keep these columns in `results/records.csv`:
- `run_id`
- `exp_group` (E1/E2/E3/A1/A2/...)
- `alpha`
- `method`
- `local_moo_backend`
- `server_moo_mode`
- `server_moo_beta`
- `server_solver`
- `server_qp_steps`
- `server_qp_lr`
- `server_fair_lambda`
- `sketch_dim`
- `server_sketch_gamma`
- `loss_stat_batches`
- `grad_stat_batches`
- `stat_every_rounds`
- `seed`
- `num_rounds`
- `avg_acc`
- `worst_task_acc`
- `fairness`
- `elapsed_sec`
- `notes`

## 4) Execution Order
1. Main comparisons: E4 -> E3 -> E2 -> E1
2. Core ablations: A1 -> A2 -> A3 -> A4
3. Robustness: R1 -> R2

Suggested command flow:
- Run one planned row: `python scripts/run_experiment_grid.py --limit 1`
- Run a specific row: `python scripts/run_experiment_grid.py --exp-id 22`
- Summarize results: `python scripts/summarize_results.py --input results/records.csv --output results/summary_by_setting.csv`

## 5) Statistical Rules
- At least 3 seeds per setting
- Report `mean +- std`
- Do not compare methods with different training budgets
- Keep stat configs fixed for fair comparison:
  - `server_solver`, `server_qp_steps`, `server_qp_lr`
  - `server_fair_lambda`, `sketch_dim`, `server_sketch_gamma`
  - `loss_stat_batches`
  - `grad_stat_batches`
  - `stat_every_rounds`

## 6) Paper Figures You Can Build Directly
- Main table: mean/std of `avg_acc`, `worst_task_acc`, `fairness`
- Convergence plots: from per-round logs in `results/raw/`
- Ablation bars: group by `server_moo_mode`
- Tradeoff line: x=`server_moo_beta`, y=`avg_acc` and `worst_task_acc`
