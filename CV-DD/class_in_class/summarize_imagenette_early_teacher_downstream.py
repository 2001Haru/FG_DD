import argparse
import json
import statistics
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42)
STUDENT_SEEDS = (42, 43)
SOURCES = ("real", "c1")
MODES = ("ref", "pred")
LABELS_BY_C = {
    1: ("v65", "v72", "v79", "v85", "v89", "final"),
    100: ("v65", "v72", "v79", "final"),
}


def load_best(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid validation set: {path}")
    return float(payload["best_top1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment_root)

    plans = {}
    selection_summary = {}
    for teacher in TEACHER_SEEDS:
        payload = json.loads(
            (experiment / f"tseed{teacher}" / "selection.json").read_text(encoding="utf-8")
        )
        for row in payload["selections"]:
            plans[(teacher, int(row["C"]), row["label"])] = row
        selection_summary[str(teacher)] = payload

    def early_path(teacher, c, label, source, mode, recovery, student):
        record = plans[(teacher, c, label)]
        epoch = int(record["epoch"])
        return (
            experiment / f"tseed{teacher}" / "per_class"
            / f"{source}__c{c}_{label}_e{epoch:03d}_{mode}_rseed{recovery}_sseed{student}.json"
        )

    values = {}
    for c in (1, 100):
        for label in LABELS_BY_C[c]:
            for source in SOURCES:
                for mode in MODES:
                    current = {}
                    for teacher in TEACHER_SEEDS:
                        for recovery in RECOVERY_SEEDS:
                            for student in STUDENT_SEEDS:
                                path = early_path(
                                    teacher, c, label, source, mode, recovery, student
                                )
                                current[(teacher, recovery, student)] = load_best(path)
                    values[(c, label, source, mode)] = current

    arms = {
        f"c{c}_{label}_{source}_{mode}": three_level_summary(
            current, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )
        for (c, label, source, mode), current in values.items()
    }
    comparisons = {}
    for label in ("v65", "v72", "v79", "final"):
        for source in SOURCES:
            for mode in MODES:
                delta = {
                    key: (
                        values[(100, label, source, mode)][key]
                        - values[(1, label, source, mode)][key]
                    )
                    for key in values[(1, label, source, mode)]
                }
                comparisons[f"same_label_{label}_{source}_{mode}_c100_minus_c1"] = (
                    three_level_summary(
                        delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
                    )
                )

    # Core matched-accuracy comparison: the C1 v79 checkpoint is selected to
    # match the same-seed C100 final validation accuracy.
    for source in SOURCES:
        for mode in MODES:
            delta = {
                key: (
                    values[(100, "final", source, mode)][key]
                    - values[(1, "v79", source, mode)][key]
                )
                for key in values[(1, "v79", source, mode)]
            }
            comparisons[f"core_c100_final_minus_c1_v79_{source}_{mode}"] = (
                three_level_summary(
                    delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
                )
            )

    checkpoint_table = []
    for teacher in TEACHER_SEEDS:
        for c in (1, 100):
            for label in LABELS_BY_C[c]:
                record = plans[(teacher, c, label)]
                checkpoint_table.append({
                    "teacher_seed": teacher,
                    "C": c,
                    "label": label,
                    "epoch": record["epoch"],
                    "train_accuracy": record["actual_train_accuracy"],
                    "val_accuracy": record["actual_val_accuracy"],
                    "sd_z": record["sd_z"],
                    "predicted_temperature": record["predicted_temperature"],
                    "metrics_source": record.get("metrics_source", "selected trajectory checkpoint"),
                    "downstream_result_source": record.get("downstream_result_source", "selected trajectory checkpoint"),
                    "trajectory_final_exactly_matches_reused_checkpoint": record.get(
                        "trajectory_final_exactly_matches_reused_checkpoint"
                    ),
                })
    core_val_mismatch = []
    for teacher in TEACHER_SEEDS:
        c1 = plans[(teacher, 1, "v79")]["actual_val_accuracy"]
        c100 = plans[(teacher, 100, "final")]["actual_val_accuracy"]
        core_val_mismatch.append(float(c1) - float(c100))

    result = {
        "protocol": (
            "ImageNette IPC10 early Teacher matched-validation experiment; native "
            "FP16 FKD, T20 and sd(z)-predicted temperature, Real/C1-synthetic sources"
        ),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "checkpoint_table": checkpoint_table,
        "core_c1_v79_minus_c100_final_val_accuracy": {
            "by_teacher_seed": dict(zip(map(str, TEACHER_SEEDS), core_val_mismatch)),
            "mean": statistics.fmean(core_val_mismatch),
            "sample_sd": statistics.stdev(core_val_mismatch),
        },
        "arms": arms,
        "comparisons": comparisons,
        "selection_manifests": selection_summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
