import argparse
import json
from pathlib import Path

from summarize_post_eval_seeds import hierarchical_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class-dir', required=True)
    parser.add_argument('--recovery-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--student-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--c-values', nargs='+', type=int, default=(1, 2, 5, 10))
    parser.add_argument('--recovery-iterations', type=int, required=True)
    parser.add_argument('--recovery-lr', type=float, required=True)
    parser.add_argument('--r-bn', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.per_class_dir)
    values = {c: {} for c in args.c_values}
    for c in values:
        for recovery_seed in args.recovery_seeds:
            for student_seed in args.student_seeds:
                path = root / f'c{c}_rseed{recovery_seed}_sseed{student_seed}.json'
                payload = json.loads(path.read_text(encoding='utf-8'))
                if int(payload.get('validation_images', -1)) != 3925:
                    raise ValueError(f'{path} was not evaluated on 3925 images')
                values[c][(recovery_seed, student_seed)] = float(payload['best_top1'])
    summary = {
        'protocol': (
            'ImageNette IPC10 ResNet18 CiC-T only, marg10, T20, official split; '
            f'recovery iter{args.recovery_iterations} LR{args.recovery_lr:g} '
            f'r_bn{args.r_bn:g}'
        ),
        'c_values': args.c_values,
        'recovery_seeds': args.recovery_seeds,
        'student_seeds': args.student_seeds,
        'arms': {
            f'C{c}': hierarchical_summary(
                values[c], args.recovery_seeds, args.student_seeds
            ) for c in values
        },
        'paired_vs_C1': {},
    }
    if 1 in values:
        for c in values:
            if c == 1:
                continue
            differences = {
                key: values[c][key] - values[1][key] for key in values[c]
            }
            summary['paired_vs_C1'][f'C{c}_minus_C1'] = hierarchical_summary(
                differences, args.recovery_seeds, args.student_seeds
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2)
    output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
