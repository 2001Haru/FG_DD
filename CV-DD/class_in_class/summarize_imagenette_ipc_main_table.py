import argparse
import json
import statistics
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


ROWS = ("real", "c1")
COLS = ("hard", "c1", "random100")
IPCS = (1, 10, 50)


def main():
    parser = argparse.ArgumentParser("Summarize ImageNette IPC1/10/50 source-labeler table")
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--ipc-root", required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--student-seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    random_root, factorial_root, ipc_root = map(
        Path, (args.random_root, args.factorial_root, args.ipc_root)
    )

    def path_for(ipc, row, col, teacher, recovery, student):
        if ipc != 10:
            return (
                ipc_root / f"tseed{teacher}" / "per_class"
                / f"ipc{ipc}_{row}__{col}_rseed{recovery}_sseed{student}.json"
            )
        if row == "real":
            return (
                factorial_root / f"tseed{teacher}" / "per_class"
                / f"real__{col}_rseed{recovery}_sseed{student}.json"
            )
        if col == "hard":
            return random_root / f"tseed{teacher}" / "hard_per_class" / f"c1_rseed{recovery}_sseed{student}.json"
        if col == "c1":
            return random_root / f"tseed{teacher}" / "per_class" / f"c1_rseed{recovery}_sseed{student}.json"
        return (
            factorial_root / f"tseed{teacher}" / "per_class"
            / f"c1__random100_rseed{recovery}_sseed{student}.json"
        )

    summaries, matrices = {}, {}
    values = {}
    for ipc in IPCS:
        values[ipc] = {}
        summaries[str(ipc)] = {}
        matrix = []
        for row in ROWS:
            summaries[str(ipc)][row] = {}
            line = []
            for col in COLS:
                current = {}
                for teacher in args.teacher_seeds:
                    for recovery in args.recovery_seeds:
                        for student in args.student_seeds:
                            path = path_for(ipc, row, col, teacher, recovery, student)
                            record = json.loads(path.read_text(encoding="utf-8"))
                            if int(record.get("validation_images", -1)) != 3925:
                                raise ValueError(f"invalid test metadata: {path}")
                            current[(teacher, recovery, student)] = float(record["best_top1"])
                values[ipc][(row, col)] = current
                summary = three_level_summary(
                    current, args.teacher_seeds, args.recovery_seeds, args.student_seeds
                )
                summaries[str(ipc)][row][col] = summary
                line.append(summary["grand_mean"])
            matrix.append(line)
        matrices[str(ipc)] = matrix

    ipc_effects = {}
    for row in ROWS:
        for col in COLS:
            for left, right in ((1, 10), (10, 50), (1, 50)):
                paired = {
                    key: values[right][(row, col)][key] - values[left][(row, col)][key]
                    for key in values[left][(row, col)]
                }
                ipc_effects[f"{row}__{col}_ipc{right}_minus_ipc{left}"] = three_level_summary(
                    paired, args.teacher_seeds, args.recovery_seeds, args.student_seeds
                )

    column_effects = {}
    row_effects = {}
    for ipc in IPCS:
        column_effects[str(ipc)] = {}
        for col in COLS:
            averaged = {
                key: statistics.fmean([values[ipc][(row, col)][key] for row in ROWS])
                for key in values[ipc][("real", col)]
            }
            column_effects[str(ipc)][col] = three_level_summary(
                averaged, args.teacher_seeds, args.recovery_seeds, args.student_seeds
            )
        row_effects[str(ipc)] = {}
        for row in ROWS:
            averaged = {
                key: statistics.fmean([values[ipc][(row, col)][key] for col in COLS])
                for key in values[ipc][(row, "hard")]
            }
            row_effects[str(ipc)][row] = three_level_summary(
                averaged, args.teacher_seeds, args.recovery_seeds, args.student_seeds
            )

    result = {
        "protocol": (
            "ImageNette IPC1/10/50 x rows{independent stratified real,C1 synthetic} "
            "x columns{Hard,C1 Teacher,RandomC100 Teacher}; 2x3x3 cells; "
            "IPC1/50 post-eval AdamW LR5e-4 eta2; IPC10 retained historical eta1"
        ),
        "rows": ROWS,
        "columns": COLS,
        "ipcs": IPCS,
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "cell_summaries": summaries,
        "accuracy_matrices": matrices,
        "column_effects_by_ipc": column_effects,
        "row_effects_by_ipc": row_effects,
        "paired_ipc_effects": ipc_effects,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
