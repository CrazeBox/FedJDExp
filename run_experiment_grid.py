import argparse
import csv
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from FedJD_v12 import Config, HP, run, set_seed
except Exception:
    from FedJD_v10 import Config, HP, run, set_seed


def _to_bool(v, default=False):
    if v is None or v == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            fixed = {}
            for k, v in row.items():
                nk = k.lstrip("\ufeff") if isinstance(k, str) else k
                if isinstance(nk, str):
                    nk = nk.strip().strip('"')
                fixed[nk] = v
            rows.append(fixed)
    return rows


def _save_rows(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _pick_rows(rows, exp_id: str, limit: int):
    selected = []
    for idx, row in enumerate(rows):
        if exp_id and str(row.get("exp_id", "")).strip() != str(exp_id).strip():
            continue
        status = str(row.get("status", "")).strip().lower()
        if not exp_id and status not in {"", "planned"}:
            continue
        selected.append((idx, row))
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _hist_metrics(histories: dict):
    for hist in histories.values():
        if hist.accs:
            return {
                "avg_acc": float(hist.accs[-1]),
                "worst_task_acc": float(min(hist.task_accs[-1])) if hist.task_accs else "",
                "fairness": float(hist.fairness[-1]) if hist.fairness else "",
            }
    return {"avg_acc": "", "worst_task_acc": "", "fairness": ""}


def _row_to_config(row: dict) -> tuple[Config, str]:
    cfg = Config()
    method = str(row.get("method", "ssjd")).strip().lower()
    cfg.exp_id = str(row.get("exp_id", "")).strip()
    cfg.exp_group = str(row.get("exp_group", "")).strip()
    cfg.alpha = _to_float(row.get("alpha"), cfg.alpha)
    cfg.server_moo_mode = str(row.get("server_moo_mode", cfg.server_moo_mode)).strip()
    if hasattr(cfg, "local_moo_backend"):
        cfg.local_moo_backend = str(row.get("local_moo_backend", cfg.local_moo_backend)).strip()
    cfg.server_moo_beta = _to_float(row.get("server_moo_beta"), cfg.server_moo_beta)
    cfg.server_solver = str(row.get("server_solver", cfg.server_solver)).strip()
    cfg.server_qp_steps = _to_int(row.get("server_qp_steps"), cfg.server_qp_steps)
    cfg.server_qp_lr = _to_float(row.get("server_qp_lr"), cfg.server_qp_lr)
    cfg.server_fair_lambda = _to_float(row.get("server_fair_lambda"), cfg.server_fair_lambda)
    cfg.sketch_dim = _to_int(row.get("sketch_dim"), cfg.sketch_dim)
    cfg.server_sketch_gamma = _to_float(row.get("server_sketch_gamma"), cfg.server_sketch_gamma)
    cfg.loss_stat_batches = _to_int(row.get("loss_stat_batches"), cfg.loss_stat_batches)
    cfg.grad_stat_batches = _to_int(row.get("grad_stat_batches"), cfg.grad_stat_batches)
    cfg.stat_every_rounds = _to_int(row.get("stat_every_rounds"), cfg.stat_every_rounds)
    cfg.ssjd_k = _to_int(row.get("ssjd_k"), cfg.ssjd_k)
    cfg.compress = _to_bool(row.get("compress"), cfg.compress)
    cfg.client_fraction = _to_float(row.get("client_fraction"), cfg.client_fraction)
    cfg.eval_freq = _to_int(row.get("eval_freq"), cfg.eval_freq)
    cfg.num_rounds = _to_int(row.get("num_rounds"), cfg.num_rounds)
    cfg.local_epochs = _to_int(row.get("local_epochs"), cfg.local_epochs)
    cfg.num_clients = _to_int(row.get("num_clients"), cfg.num_clients)
    cfg.seed = _to_int(row.get("seed"), cfg.seed)
    cfg.run_tag = str(row.get("run_id", "")).strip()
    if not cfg.run_tag:
        cfg.run_tag = f"exp{cfg.exp_id}_{cfg.exp_group}_{method}_seed{cfg.seed}"
    cfg.notes = str(row.get("notes", "")).strip()
    HP.compress_ratio = _to_float(row.get("compress_ratio"), HP.compress_ratio)
    return cfg, method


def main():
    p = argparse.ArgumentParser(description="Run FedJD experiments from experiment_grid.csv")
    p.add_argument("--grid", default="experiments/experiment_grid.csv", help="grid CSV path")
    p.add_argument("--exp-id", default="", help="run only this exp_id")
    p.add_argument("--limit", type=int, default=1, help="number of rows to run (<=0 means all)")
    args = p.parse_args()

    grid_path = Path(args.grid)
    rows = _load_rows(grid_path)
    selected = _pick_rows(rows, args.exp_id, args.limit)
    if not selected:
        print("No runnable rows found.")
        return

    for idx, row in selected:
        cfg, method = _row_to_config(row)
        rows[idx]["status"] = "running"
        rows[idx]["run_id"] = cfg.run_tag
        _save_rows(grid_path, rows)

        try:
            set_seed(cfg.seed)
            t0 = time.perf_counter()
            histories = run(cfg, [method])
            elapsed = time.perf_counter() - t0
            m = _hist_metrics(histories)
            rows[idx]["avg_acc"] = m["avg_acc"]
            rows[idx]["worst_task_acc"] = m["worst_task_acc"]
            rows[idx]["fairness"] = m["fairness"]
            rows[idx]["elapsed_sec"] = f"{elapsed:.2f}"
            rows[idx]["status"] = "completed"
            _save_rows(grid_path, rows)
            print(f"Completed exp_id={rows[idx].get('exp_id')} run_id={cfg.run_tag}")
        except Exception as e:
            rows[idx]["status"] = "failed"
            msg = str(e).replace("\n", " ").strip()
            rows[idx]["notes"] = (rows[idx].get("notes", "") + " | " + msg).strip(" |")
            _save_rows(grid_path, rows)
            print(f"Failed exp_id={rows[idx].get('exp_id')}: {e}")


if __name__ == "__main__":
    main()
