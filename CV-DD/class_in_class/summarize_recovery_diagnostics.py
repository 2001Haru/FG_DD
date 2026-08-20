import argparse
import json
import statistics
from pathlib import Path


FIELDS = (
    "ce",
    "bn_raw",
    "bn_weighted",
    "bn_to_ce_loss_ratio",
    "ce_grad_rms",
    "bn_grad_rms",
    "bn_to_ce_grad_ratio",
    "relative_update_rms",
)


def load(path):
    records = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("RECOVERY_DIAG "):
                record = json.loads(line[len("RECOVERY_DIAG "):])
            elif line.startswith("{"):
                record = json.loads(line)
            else:
                continue
            record["progress"] = record["iteration"] / max(record["iterations"] - 1, 1)
            records.append(record)
    deduplicated = {
        (record["seed"], record["batch_id"], record["iteration"]): record
        for record in records
    }
    return list(deduplicated.values())


def segment(record):
    if record["progress"] <= 0.1:
        return "early[0,10%]"
    if 0.45 <= record["progress"] <= 0.55:
        return "middle[45,55%]"
    if record["progress"] >= 0.9:
        return "late[90,100%]"
    return None


def summarize(name, records):
    if not records:
        raise RuntimeError(f"no RECOVERY_DIAG records found for {name}")
    print(f"\n===== {name}: {len(records)} diagnostic points =====")
    for section in ("early[0,10%]", "middle[45,55%]", "late[90,100%]"):
        selected = [record for record in records if segment(record) == section]
        if not selected:
            continue
        print(f"[{section}] n={len(selected)}")
        for field in FIELDS:
            values = [float(record[field]) for record in selected]
            print(f"  {field}: median={statistics.median(values):.6g}, "
                  f"min={min(values):.6g}, max={max(values):.6g}")


def main():
    parser = argparse.ArgumentParser("Summarize CE/BN balance and image update scale from CV-DD recovery logs")
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--oracle-log", required=True)
    parser.add_argument("--random-log")
    parser.add_argument("--coarse-target-log")
    parser.add_argument("--random-coarse-target-log")
    args = parser.parse_args()
    baseline, oracle = load(args.baseline_log), load(args.oracle_log)
    summarize("baseline-coarse20", baseline)
    summarize("oracle-fine100-stratified", oracle)
    if args.random_log:
        summarize("random-pseudo100-stratified", load(args.random_log))
    if args.coarse_target_log:
        summarize("fine100-marginalized-coarse20", load(args.coarse_target_log))
    if args.random_coarse_target_log:
        summarize("random100-marginalized-coarse20", load(args.random_coarse_target_log))
    print("\nCalibration targets:")
    print("  r_bn controls bn_to_ce_grad_ratio; compare this before scalar loss ratios.")
    print("  lr controls relative_update_rms; compare at matched optimization progress.")


if __name__ == "__main__":
    main()
