import argparse
import concurrent.futures
import json
import statistics
from pathlib import Path

import numpy as np

from audit_imagenette_consumed_fkd_labels import analyze_root, summarize_roots
from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


ROWS = ("real", "c1", "random100", "cluster100")
COLS = ("hard", "c1", "random100", "cluster100")


def main():
    parser = argparse.ArgumentParser("Summarize 4x4 ImageNette source-by-labeler factorial")
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--cluster-root", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--student-seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--rank-epoch-stride", type=int, default=10)
    parser.add_argument("--rank-workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    random_root, cluster_root, matrix_root = map(
        Path, (args.random_root, args.cluster_root, args.matrix_root)
    )

    def source(row, teacher, recovery):
        if row == "real":
            return matrix_root / "real_sets" / f"tseed{teacher}_rseed{recovery}"
        if row == "c1":
            return random_root / f"tseed{teacher}" / "synthetic" / f"cic_t_c1_ipc10_rseed{recovery}"
        if row == "random100":
            return random_root / f"tseed{teacher}" / "synthetic" / f"cic_t_c100_ipc10_rseed{recovery}"
        if row == "cluster100":
            return cluster_root / f"tseed{teacher}" / "synthetic" / f"cic_t_c100_ipc10_rseed{recovery}"
        raise ValueError(row)

    def result_path(row, col, teacher, recovery, student):
        if row == "c1" and col == "hard":
            return random_root / f"tseed{teacher}" / "hard_per_class" / f"c1_rseed{recovery}_sseed{student}.json"
        if row == "c1" and col == "c1":
            return random_root / f"tseed{teacher}" / "per_class" / f"c1_rseed{recovery}_sseed{student}.json"
        if row == "random100" and col == "hard":
            return random_root / f"tseed{teacher}" / "hard_per_class" / f"c100_rseed{recovery}_sseed{student}.json"
        if row == "random100" and col == "random100":
            return random_root / f"tseed{teacher}" / "per_class" / f"c100_rseed{recovery}_sseed{student}.json"
        if row == "cluster100" and col == "hard":
            return cluster_root / f"tseed{teacher}" / "hard_per_class" / f"c100_rseed{recovery}_sseed{student}.json"
        if row == "cluster100" and col == "cluster100":
            return cluster_root / f"tseed{teacher}" / "per_class" / f"c100_rseed{recovery}_sseed{student}.json"
        return (
            matrix_root / f"tseed{teacher}" / "per_class"
            / f"{row}__{col}_rseed{recovery}_sseed{student}.json"
        )

    def fkd_root(row, col, teacher, recovery):
        if row == "c1" and col == "c1":
            return random_root / f"tseed{teacher}" / "fkd" / f"cic_t_c1_rseed{recovery}_bs10_ipc10"
        if row == "random100" and col == "random100":
            return random_root / f"tseed{teacher}" / "fkd" / f"cic_t_c100_rseed{recovery}_bs10_ipc10"
        if row == "cluster100" and col == "cluster100":
            return cluster_root / f"tseed{teacher}" / "fkd" / f"cic_t_c100_rseed{recovery}_bs10_ipc10"
        return (
            matrix_root / f"tseed{teacher}" / "fkd"
            / f"{row}__{col}_rseed{recovery}_bs10_ipc10"
        )

    values = {(row, col): {} for row in ROWS for col in COLS}
    for row in ROWS:
        for col in COLS:
            for teacher in args.teacher_seeds:
                for recovery in args.recovery_seeds:
                    for student in args.student_seeds:
                        path = result_path(row, col, teacher, recovery, student)
                        record = json.loads(path.read_text(encoding="utf-8"))
                        if int(record.get("validation_images", -1)) != 3925:
                            raise ValueError(f"invalid test metadata: {path}")
                        values[(row, col)][(teacher, recovery, student)] = float(
                            record["best_top1"]
                        )

    cell_summaries = {
        row: {
            col: three_level_summary(
                values[(row, col)], args.teacher_seeds,
                args.recovery_seeds, args.student_seeds,
            ) for col in COLS
        } for row in ROWS
    }
    accuracy_matrix = [
        [cell_summaries[row][col]["grand_mean"] for col in COLS] for row in ROWS
    ]

    row_effects, column_effects = {}, {}
    for row in ROWS:
        averaged = {
            key: statistics.fmean([values[(row, col)][key] for col in COLS])
            for key in values[(row, "hard")]
        }
        row_effects[row] = three_level_summary(
            averaged, args.teacher_seeds, args.recovery_seeds, args.student_seeds
        )
    for col in COLS:
        averaged = {
            key: statistics.fmean([values[(row, col)][key] for row in ROWS])
            for key in values[("real", col)]
        }
        column_effects[col] = three_level_summary(
            averaged, args.teacher_seeds, args.recovery_seeds, args.student_seeds
        )

    overall = float(np.mean(accuracy_matrix))
    row_means = np.mean(np.array(accuracy_matrix), axis=1)
    col_means = np.mean(np.array(accuracy_matrix), axis=0)
    additive_residual = (
        np.array(accuracy_matrix) - row_means[:, None] - col_means[None, :] + overall
    )

    matched_keys = (("c1", "c1"), ("random100", "random100"), ("cluster100", "cluster100"))
    unmatched_keys = tuple(
        (row, col) for row in ("c1", "random100", "cluster100")
        for col in ("c1", "random100", "cluster100") if (row, col) not in matched_keys
    )
    match_contrast = {
        key: (
            statistics.fmean([values[cell][key] for cell in matched_keys])
            - statistics.fmean([values[cell][key] for cell in unmatched_keys])
        ) for key in values[("c1", "c1")]
    }
    matched_effect = three_level_summary(
        match_contrast, args.teacher_seeds, args.recovery_seeds, args.student_seeds
    )

    rank_tasks = []
    for row in ROWS:
        for col in COLS:
            if col == "hard":
                continue
            for teacher in args.teacher_seeds:
                for recovery in args.recovery_seeds:
                    rank_tasks.append((
                        f"{row}__{col}", 100, teacher, recovery,
                        str(source(row, teacher, recovery)),
                        str(fkd_root(row, col, teacher, recovery)),
                        300, args.rank_epoch_stride, 20.0, 42,
                    ))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.rank_workers) as executor:
        rank_rows = list(executor.map(analyze_root, rank_tasks))

    rank_summaries, participation_matrix = {}, []
    for row in ROWS:
        rank_summaries[row] = {}
        rank_line = []
        for col in COLS:
            if col == "hard":
                summary = {
                    "definition": "balanced 10-way one-hot hard labels",
                    "centered_spectral_effective_rank": {"mean_across_teacher_recovery_roots": 9.0},
                    "centered_covariance_participation_rank": {"mean_across_teacher_recovery_roots": 9.0},
                    "uncentered_spectral_effective_rank": {"mean_across_teacher_recovery_roots": 10.0},
                }
            else:
                current = [item for item in rank_rows if item["partition"] == f"{row}__{col}"]
                summary = summarize_roots(current, args.teacher_seeds, args.recovery_seeds)
            rank_summaries[row][col] = summary
            rank_line.append(summary["centered_covariance_participation_rank"][
                "mean_across_teacher_recovery_roots"
            ])
        participation_matrix.append(rank_line)

    flat_accuracy = np.array(accuracy_matrix).reshape(-1)
    flat_rank = np.array(participation_matrix).reshape(-1)
    rank_accuracy_correlation = float(np.corrcoef(flat_rank, flat_accuracy)[0, 1])

    def fit_model(matrix, names):
        coefficients, _, matrix_rank, _ = np.linalg.lstsq(matrix, flat_accuracy, rcond=None)
        fitted = matrix @ coefficients
        residual = flat_accuracy - fitted
        total_ss = float(((flat_accuracy - flat_accuracy.mean()) ** 2).sum())
        residual_ss = float((residual ** 2).sum())
        return {
            "coefficient_names": names,
            "coefficients": coefficients.tolist(),
            "matrix_rank": int(matrix_rank),
            "residual_df": int(len(flat_accuracy) - matrix_rank),
            "r_squared": 1.0 - residual_ss / total_ss,
            "residual_sum_of_squares": residual_ss,
            "residual_matrix": residual.reshape(4, 4).tolist(),
        }

    design_rows = []
    strength_product = []
    rank_predictor = []
    for row_id, row in enumerate(ROWS):
        for col_id, col in enumerate(COLS):
            matched = float(
                row in ("c1", "random100", "cluster100") and row == col
            )
            design_rows.append([
                1.0,
                *[float(row_id == index) for index in range(1, 4)],
                *[float(col_id == index) for index in range(1, 4)],
                matched,
            ])
            strength_product.append(
                float((row_means[row_id] - overall) * (col_means[col_id] - overall))
            )
            rank_predictor.append(float(participation_matrix[row_id][col_id] - flat_rank.mean()))
    base_design = np.asarray(design_rows)
    product_column = np.asarray(strength_product)[:, None]
    rank_column = np.asarray(rank_predictor)[:, None]
    base_names = [
        "intercept", "row_c1", "row_random100", "row_cluster100",
        "col_c1", "col_random100", "col_cluster100", "matched_diagonal",
    ]
    factorial_models = {
        "categorical_row_column_plus_match": fit_model(base_design, base_names),
        "plus_row_strength_x_column_strength": fit_model(
            np.column_stack([base_design, product_column]),
            [*base_names, "row_strength_x_column_strength"],
        ),
        "plus_participation_rank": fit_model(
            np.column_stack([base_design, rank_column]),
            [*base_names, "centered_participation_rank"],
        ),
        "plus_strength_interaction_and_rank": fit_model(
            np.column_stack([base_design, product_column, rank_column]),
            [*base_names, "row_strength_x_column_strength", "centered_participation_rank"],
        ),
    }

    result = {
        "protocol": (
            "ImageNette IPC10 4x4 source-by-labeler factorial; Teacher seed paired "
            "across source/labeler, recovery seeds41/42/43, student seeds42/43/44"
        ),
        "rows": ROWS,
        "columns": COLS,
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "cell_summaries": cell_summaries,
        "accuracy_grand_mean_matrix": accuracy_matrix,
        "row_effects": row_effects,
        "column_effects": column_effects,
        "overall_grand_mean": overall,
        "additive_residual_matrix": additive_residual.tolist(),
        "matched_minus_unmatched_soft_labeler_effect": matched_effect,
        "rank_epoch_stride": args.rank_epoch_stride,
        "participation_rank_matrix": participation_matrix,
        "rank_summaries": rank_summaries,
        "descriptive_rank_accuracy_pearson_across_16_cells": rank_accuracy_correlation,
        "factorial_cell_mean_models": factorial_models,
        "factorial_model_note": (
            "Descriptive OLS on 16 cell grand means. The base categorical row+column+match "
            "model has 8 residual degrees of freedom; it is not an 18-cell independence claim."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
