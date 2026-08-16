"""Compare Stage 8 displacement against the matched Stage 4 alignment runs."""

import json
from pathlib import Path


METRICS = ("psnr", "ssim", "lpips", "boundary_l1", "boundary_f1")
HIGHER_IS_BETTER = {"psnr", "ssim", "boundary_f1"}
PAIRS = {
    "032dee9f": (
        Path("outputs/stage4_alignment/eval_200step_5scenes_fs0p35_ar2_as1p0/metrics"),
        Path("outputs/stage8_minival/eval/032dee9f/displacement_200steps/metrics"),
    ),
    "073f5a9b": (
        Path("outputs/stage5_minival/eval/073f5a9b/joint_alignment_200steps/metrics"),
        Path("outputs/stage8_minival/eval/073f5a9b/displacement_200steps/metrics"),
    ),
    "14eb48a5": (
        Path("outputs/stage5_minival/eval/14eb48a5/joint_alignment_200steps/metrics"),
        Path("outputs/stage8_minival/eval/14eb48a5/displacement_200steps/metrics"),
    ),
}


def load_values(root: Path, metric: str) -> list[float]:
    with (root / f"scores_{metric}_all.json").open() as handle:
        return json.load(handle)


def main() -> None:
    result = {"per_train_scene": {}, "aggregate_15_pairs": {}}
    all_deltas = {metric: [] for metric in METRICS}
    for scene, (stage4_root, stage8_root) in PAIRS.items():
        scene_result = {}
        for metric in METRICS:
            stage4 = load_values(stage4_root, metric)
            stage8 = load_values(stage8_root, metric)
            deltas = [new - old for old, new in zip(stage4, stage8)]
            all_deltas[metric].extend(deltas)
            scene_result[metric] = {
                "delta": sum(deltas) / len(deltas),
                "wins": sum(
                    delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
                    for delta in deltas
                ),
                "count": len(deltas),
            }
        result["per_train_scene"][scene] = scene_result

    for metric, deltas in all_deltas.items():
        result["aggregate_15_pairs"][metric] = {
            "mean_delta": sum(deltas) / len(deltas),
            "wins": sum(
                delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
                for delta in deltas
            ),
            "count": len(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
        }

    output = Path("outputs/stage8_minival/comparison_stage4_vs_stage8.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
