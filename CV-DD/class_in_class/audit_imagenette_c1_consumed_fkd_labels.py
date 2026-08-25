import argparse
import concurrent.futures
import json
from pathlib import Path

from audit_imagenette_consumed_fkd_labels import analyze_root, summarize_roots


def main():
    parser = argparse.ArgumentParser("Audit existing C1 student-consumed FKD labels")
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--epoch-stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.random_root)
    tasks = []
    for teacher in args.teacher_seeds:
        teacher_root = root / f"tseed{teacher}"
        for recovery in args.recovery_seeds:
            tasks.append((
                "c1", 1, teacher, recovery,
                str(teacher_root / "synthetic" / f"cic_t_c1_ipc10_rseed{recovery}"),
                str(teacher_root / "fkd" / f"cic_t_c1_rseed{recovery}_bs10_ipc10"),
                300, args.epoch_stride, 20.0, 42,
            ))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(analyze_root, tasks))
    rows.sort(key=lambda row: (row["teacher_seed"], row["recovery_seed"]))
    summary = summarize_roots(rows, args.teacher_seeds, args.recovery_seeds)
    result = {
        "protocol": "Existing ImageNette C1 consumed marg10 FKD labels at T20",
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "epoch_stride": args.epoch_stride,
        "root_metrics": rows,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
