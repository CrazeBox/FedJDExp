import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def to_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def mean_std(vals):
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(var)


def main():
    p = argparse.ArgumentParser(description="Summarize FedJD experiment records")
    p.add_argument("--input", default="results/records.csv", help="input CSV path")
    p.add_argument("--output", default="results/summary_by_setting.csv", help="output CSV path")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    groups = defaultdict(list)
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("exp_group", ""),
                row.get("alpha", ""),
                row.get("method", ""),
                row.get("local_moo_backend", ""),
                row.get("server_moo_mode", ""),
                row.get("server_moo_beta", ""),
                row.get("server_solver", ""),
                row.get("server_qp_steps", ""),
                row.get("server_qp_lr", ""),
                row.get("server_fair_lambda", ""),
                row.get("sketch_dim", ""),
                row.get("server_sketch_gamma", ""),
                row.get("loss_stat_batches", ""),
                row.get("grad_stat_batches", ""),
                row.get("stat_every_rounds", ""),
            )
            groups[key].append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "exp_group",
        "alpha",
        "method",
        "local_moo_backend",
        "server_moo_mode",
        "server_moo_beta",
        "server_solver",
        "server_qp_steps",
        "server_qp_lr",
        "server_fair_lambda",
        "sketch_dim",
        "server_sketch_gamma",
        "loss_stat_batches",
        "grad_stat_batches",
        "stat_every_rounds",
        "n_runs",
        "avg_acc_mean",
        "avg_acc_std",
        "worst_task_acc_mean",
        "worst_task_acc_std",
        "fairness_mean",
        "fairness_std",
        "elapsed_sec_mean",
        "elapsed_sec_std",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, rows in sorted(groups.items()):
            avg_vals = [to_float(r.get("avg_acc")) for r in rows]
            worst_vals = [to_float(r.get("worst_task_acc")) for r in rows]
            fair_vals = [to_float(r.get("fairness")) for r in rows]
            time_vals = [to_float(r.get("elapsed_sec")) for r in rows]

            avg_vals = [x for x in avg_vals if x is not None]
            worst_vals = [x for x in worst_vals if x is not None]
            fair_vals = [x for x in fair_vals if x is not None]
            time_vals = [x for x in time_vals if x is not None]

            avg_m, avg_s = mean_std(avg_vals)
            worst_m, worst_s = mean_std(worst_vals)
            fair_m, fair_s = mean_std(fair_vals)
            time_m, time_s = mean_std(time_vals)

            w.writerow({
                "exp_group": key[0],
                "alpha": key[1],
                "method": key[2],
                "local_moo_backend": key[3],
                "server_moo_mode": key[4],
                "server_moo_beta": key[5],
                "server_solver": key[6],
                "server_qp_steps": key[7],
                "server_qp_lr": key[8],
                "server_fair_lambda": key[9],
                "sketch_dim": key[10],
                "server_sketch_gamma": key[11],
                "loss_stat_batches": key[12],
                "grad_stat_batches": key[13],
                "stat_every_rounds": key[14],
                "n_runs": len(rows),
                "avg_acc_mean": "" if avg_m is None else f"{avg_m:.4f}",
                "avg_acc_std": "" if avg_s is None else f"{avg_s:.4f}",
                "worst_task_acc_mean": "" if worst_m is None else f"{worst_m:.4f}",
                "worst_task_acc_std": "" if worst_s is None else f"{worst_s:.4f}",
                "fairness_mean": "" if fair_m is None else f"{fair_m:.6f}",
                "fairness_std": "" if fair_s is None else f"{fair_s:.6f}",
                "elapsed_sec_mean": "" if time_m is None else f"{time_m:.2f}",
                "elapsed_sec_std": "" if time_s is None else f"{time_s:.2f}",
            })

    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
