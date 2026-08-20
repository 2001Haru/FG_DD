import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_records(path, arm, seed):
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("RECOVERY_DIAG "):
                record = json.loads(line[len("RECOVERY_DIAG "):])
            elif line.startswith("{"):
                record = json.loads(line)
            else:
                continue
            iteration = int(record["iteration"])
            iterations = int(record["iterations"])
            records.append({
                "arm": arm,
                "seed": seed,
                "batch_id": int(record["batch_id"]),
                "iteration": iteration,
                "iterations": iterations,
                "progress": iteration / max(iterations - 1, 1),
                "ce": float(record["ce"]),
            })
    records = list({
        (record["seed"], record["batch_id"], record["iteration"]): record
        for record in records
    }.values())
    if not records:
        raise RuntimeError(f"no RECOVERY_DIAG records found in {path}")
    return records


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def aggregate_curve(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["arm"], record["iteration"], record["iterations"])].append(record["ce"])
    rows = []
    for (arm, iteration, iterations), values in sorted(groups.items()):
        rows.append({
            "arm": arm, "iteration": iteration, "iterations": iterations,
            "progress": iteration / max(iterations - 1, 1), "n": len(values),
            "mean_ce": statistics.fmean(values), "median_ce": statistics.median(values),
            "q25_ce": percentile(values, 0.25), "q75_ce": percentile(values, 0.75),
            "min_ce": min(values), "max_ce": max(values),
            "mean_ce_over_initial": statistics.fmean(
                record["ce_over_initial"] for record in records
                if record["arm"] == arm and record["iteration"] == iteration
                and record["iterations"] == iterations
            ),
            "median_ce_over_initial": statistics.median(
                record["ce_over_initial"] for record in records
                if record["arm"] == arm and record["iteration"] == iteration
                and record["iterations"] == iterations
            ),
            "q25_ce_over_initial": percentile([
                record["ce_over_initial"] for record in records
                if record["arm"] == arm and record["iteration"] == iteration
                and record["iterations"] == iterations
            ], 0.25),
            "q75_ce_over_initial": percentile([
                record["ce_over_initial"] for record in records
                if record["arm"] == arm and record["iteration"] == iteration
                and record["iterations"] == iterations
            ], 0.75),
        })
    return rows


def terminal_summary(records):
    trajectories = defaultdict(list)
    for record in records:
        trajectories[(record["arm"], record["seed"], record["batch_id"])].append(record)
    terminal = []
    for (arm, seed, batch_id), values in sorted(trajectories.items()):
        ordered = sorted(values, key=lambda item: item["progress"])
        last = ordered[-1]
        normalized_auc = sum(
            0.5 * (left["ce_over_initial"] + right["ce_over_initial"])
            * (right["progress"] - left["progress"])
            for left, right in zip(ordered, ordered[1:])
        )
        terminal.append({"arm": arm, "seed": seed, "batch_id": batch_id,
                         "iteration": last["iteration"], "ce": last["ce"],
                         "ce_over_initial": last["ce_over_initial"],
                         "normalized_ce_auc": normalized_auc})

    result = {"definition": "CE at the final recovery iteration; no CE matching was applied"}
    for arm in sorted({row["arm"] for row in terminal}):
        arm_rows = [row for row in terminal if row["arm"] == arm]
        all_values = [row["ce"] for row in arm_rows]
        normalized_values = [row["ce_over_initial"] for row in arm_rows]
        auc_values = [row["normalized_ce_auc"] for row in arm_rows]
        by_seed = {}
        for seed in sorted({row["seed"] for row in arm_rows}):
            values = [row["ce"] for row in arm_rows if row["seed"] == seed]
            normalized = [row["ce_over_initial"] for row in arm_rows if row["seed"] == seed]
            aucs = [row["normalized_ce_auc"] for row in arm_rows if row["seed"] == seed]
            by_seed[str(seed)] = {
                "n_batches": len(values), "mean_ce": statistics.fmean(values),
                "median_ce": statistics.median(values), "min_ce": min(values), "max_ce": max(values),
                "mean_ce_over_initial": statistics.fmean(normalized),
                "median_ce_over_initial": statistics.median(normalized),
                "mean_normalized_ce_auc": statistics.fmean(aucs),
                "median_normalized_ce_auc": statistics.median(aucs),
            }
        result[arm] = {
            "n_seed_batches": len(all_values), "mean_ce": statistics.fmean(all_values),
            "median_ce": statistics.median(all_values), "q25_ce": percentile(all_values, 0.25),
            "q75_ce": percentile(all_values, 0.75), "by_seed": by_seed,
            "mean_ce_over_initial": statistics.fmean(normalized_values),
            "median_ce_over_initial": statistics.median(normalized_values),
            "mean_normalized_ce_auc": statistics.fmean(auc_values),
            "median_normalized_ce_auc": statistics.median(auc_values),
        }
    return terminal, result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV/JSON CE outputs were still written")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colors = {
        "baseline": "#3569b0", "oracle": "#d04a35", "random": "#3b9853",
        "coarse_target": "#8b5fbf",
        "random_coarse_target": "#a66a3f",
    }
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        x = [row["iteration"] for row in selected]
        median = [max(row["median_ce"], 1e-12) for row in selected]
        q25 = [max(row["q25_ce"], 1e-12) for row in selected]
        q75 = [max(row["q75_ce"], 1e-12) for row in selected]
        axes[0].plot(x, median, label=f"{arm} median", color=colors[arm], linewidth=2)
        axes[0].fill_between(x, q25, q75, color=colors[arm], alpha=0.18)
        normalized = [max(row["median_ce_over_initial"], 1e-12) for row in selected]
        normalized_q25 = [max(row["q25_ce_over_initial"], 1e-12) for row in selected]
        normalized_q75 = [max(row["q75_ce_over_initial"], 1e-12) for row in selected]
        axes[1].plot(x, normalized, label=f"{arm} median", color=colors[arm], linewidth=2)
        axes[1].fill_between(x, normalized_q25, normalized_q75, color=colors[arm], alpha=0.18)
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Recovery iteration")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Cross-entropy (log scale)")
    axes[0].set_title("Raw recovery CE")
    axes[1].set_ylabel("CE / CE at iteration 0 (log scale)")
    axes[1].set_title("Within-trajectory normalized CE")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Export class-in-class recovery CE curves and terminal values")
    parser.add_argument("--synthetic-parent", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--random-partition-seed", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    synthetic_parent, output = Path(args.synthetic_parent), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        seed_root = synthetic_parent / f"seed{seed}"
        records.extend(load_records(
            seed_root / "baseline_coarse20_ipc25" / "recovery_diagnostics.jsonl", "baseline", seed
        ))
        records.extend(load_records(
            seed_root / "oracle_fine100_ipc5" / "recovery_diagnostics.jsonl", "oracle", seed
        ))
        if args.random_partition_seed is not None:
            random_diagnostics = seed_root / (
                f"random_pseudo100_pseed{args.random_partition_seed}_ipc5"
            ) / "recovery_diagnostics.jsonl"
            if random_diagnostics.is_file():
                records.extend(load_records(random_diagnostics, "random", seed))
        coarse_target_diagnostics = (
            seed_root / "fine100_coarse_target_ipc25" / "recovery_diagnostics.jsonl"
        )
        if coarse_target_diagnostics.is_file():
            records.extend(load_records(coarse_target_diagnostics, "coarse_target", seed))
        random_coarse_target_diagnostics = seed_root / (
            f"random100_coarse_target_pseed{args.random_partition_seed}_ipc25"
        ) / "recovery_diagnostics.jsonl"
        if random_coarse_target_diagnostics.is_file():
            records.extend(load_records(
                random_coarse_target_diagnostics, "random_coarse_target", seed
            ))
    initial_ce = {}
    for record in records:
        key = (record["arm"], record["seed"], record["batch_id"])
        if key not in initial_ce or record["iteration"] == 0:
            initial_ce[key] = record["ce"]
    for record in records:
        key = (record["arm"], record["seed"], record["batch_id"])
        record["ce_over_initial"] = record["ce"] / max(initial_ce[key], 1e-12)
    curve = aggregate_curve(records)
    terminal, summary = terminal_summary(records)
    write_csv(output / "recovery_ce_per_batch.csv", records,
              ("arm", "seed", "batch_id", "iteration", "iterations", "progress", "ce",
               "ce_over_initial"))
    write_csv(output / "recovery_ce_curve.csv", curve,
              ("arm", "iteration", "iterations", "progress", "n", "mean_ce", "median_ce",
               "q25_ce", "q75_ce", "min_ce", "max_ce", "mean_ce_over_initial",
               "median_ce_over_initial", "q25_ce_over_initial", "q75_ce_over_initial"))
    write_csv(output / "recovery_ce_terminal_per_batch.csv", terminal,
              ("arm", "seed", "batch_id", "iteration", "ce", "ce_over_initial",
               "normalized_ce_auc"))
    with (output / "recovery_ce_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_curve(curve, output / "recovery_ce_curves.png")
    print(json.dumps(summary, indent=2))
    print(f"Recovery CE analysis written to {output}")


if __name__ == "__main__":
    main()
