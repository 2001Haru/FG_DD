import argparse
import json
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42)
STUDENT_SEEDS = (42, 43)
SOURCES = ("real", "c1")
MODES = ("ref", "pred", "t8", "t46", "t100", "t200")
TRAINING_EPOCHS = (4, 8, 16, 32, 64, 100, 150, 200, 250, 300)
LABELS = tuple(f"e{epoch:03d}" for epoch in TRAINING_EPOCHS)


def modes_for_label(label):
    # Epoch 4 was added after the Tpred analysis and is intentionally evaluated
    # only at the five fixed temperatures used by the heatmaps.
    if label == "e004":
        return tuple(mode for mode in MODES if mode != "pred")
    return MODES


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
        for label in LABELS:
            for source in SOURCES:
                for mode in modes_for_label(label):
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
    for label in LABELS:
        for source in SOURCES:
            for mode in modes_for_label(label):
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

    checkpoint_table = []
    for teacher in TEACHER_SEEDS:
        for c in (1, 100):
            for label in LABELS:
                record = plans[(teacher, c, label)]
                checkpoint_table.append({
                    "teacher_seed": teacher,
                    "C": c,
                    "label": label,
                    "training_epoch": record["training_epoch"],
                    "epoch": record["epoch"],
                    "train_accuracy": record["actual_train_accuracy"],
                    "val_accuracy": record["actual_val_accuracy"],
                    "sd_z": record["sd_z"],
                    "marg_label_entropy_T20": record.get("marg_label_entropy_T20"),
                    "participation_rank": record.get("participation_rank"),
                    "lr": record.get("lr"),
                    "val_sd_z": record.get("val_sd_z"),
                    "val_marg_entropy_T20": record.get("val_marg_entropy_T20"),
                    "val_participation_rank": record.get("val_participation_rank"),
                    "predicted_temperature": record["predicted_temperature"],
                    "metrics_source": record.get("metrics_source", "selected trajectory checkpoint"),
                    "downstream_result_source": record.get("downstream_result_source", "selected trajectory checkpoint"),
                    "trajectory_final_exactly_matches_reused_checkpoint": record.get(
                        "trajectory_final_exactly_matches_reused_checkpoint"
                    ),
                })
    result = {
        "protocol": (
            "ImageNette IPC10 fixed-epoch Teacher trajectory experiment; native "
            "FP16 FKD, T20, sd(z)-predicted T, and fixed T=8/46/100/200; "
            "Real/C1-synthetic sources"
        ),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "temperature_modes": {
            "ref": 20,
            "pred": "checkpoint-specific sd(z) prediction",
            "t8": 8,
            "t46": 46,
            "t100": 100,
            "t200": 200,
        },
        "checkpoint_table": checkpoint_table,
        "training_epochs": list(TRAINING_EPOCHS),
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
