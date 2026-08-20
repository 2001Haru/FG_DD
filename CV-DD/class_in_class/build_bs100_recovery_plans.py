import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("Build equal-budget BS100 recovery plans")
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--random-mapping")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.mapping).open(encoding="utf-8") as handle:
        hierarchy = json.load(handle)

    baseline_batches = []
    for batch_id in range(5):
        entries = []
        for coarse in range(20):
            for local_id in range(5):
                image_id = batch_id * 5 + local_id
                entries.append({"class_id": coarse, "image_id": image_id, "patch_id": image_id})
        baseline_batches.append(entries)

    oracle_batches = []
    for batch_id in range(5):
        entries = [
            {
                "class_id": fine,
                "coarse_id": int(hierarchy["fine_to_coarse"][str(fine)]),
                "image_id": batch_id,
                "patch_id": batch_id,
            }
            for fine in range(100)
        ]
        oracle_batches.append(entries)

    random_batches = None
    if args.random_mapping:
        with Path(args.random_mapping).open(encoding="utf-8") as handle:
            random_hierarchy = json.load(handle)
        random_batches = []
        for batch_id in range(5):
            random_batches.append([
                {
                    "class_id": pseudo_id,
                    "coarse_id": int(random_hierarchy["fine_to_coarse"][str(pseudo_id)]),
                    "image_id": batch_id,
                    "patch_id": batch_id,
                }
                for pseudo_id in range(100)
            ])

    plans = [
        ("baseline_coarse20_ipc25", baseline_batches, 20, 25),
        ("oracle_fine100_ipc5", oracle_batches, 100, 5),
    ]
    if random_batches is not None:
        partition_seed = int(random_hierarchy["partition_seed"])
        plans.append((f"random_pseudo100_pseed{partition_seed}_ipc5", random_batches, 100, 5))
    for name, batches, classes, ipc in plans:
        flat = batches[0] + batches[1] + batches[2] + batches[3] + batches[4]
        counts = {class_id: 0 for class_id in range(classes)}
        for entry in flat:
            counts[entry["class_id"]] += 1
        if len(batches) != 5 or any(len(batch) != 100 for batch in batches):
            raise RuntimeError(f"{name}: expected five batches of 100")
        if set(counts.values()) != {ipc}:
            raise RuntimeError(f"{name}: per-class count mismatch: {counts}")
        payload = {
            "name": name,
            "num_classes": classes,
            "ipc": ipc,
            "batch_size": 100,
            "num_batches": 5,
            "batches": batches,
        }
        output = Path(args.output_dir) / f"{name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{name}: 5 batches x 100 images, {classes} classes x IPC{ipc}; {output}")


if __name__ == "__main__":
    main()
