"""CPU-only comparison of first-frame object-query self-attention artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_condition(directory: Path, audit: Path, query_audit: Path | None = None) -> tuple[dict, torch.Tensor, dict]:
    """Load key masks independently and verify audited first-frame query geometry."""
    maps = torch.load(directory / "self_attention_viz_maps.pt", map_location="cpu", weights_only=False)
    support = torch.load(audit / "reference_support.pt", map_location="cpu", weights_only=False)
    mask = torch.as_tensor(support["support_mask_fhw"], dtype=torch.bool)
    metadata = json.loads((directory / "self_attention_viz_summary.json").read_text())
    audit_metadata = json.loads((audit / "region_audit.json").read_text())
    with (directory / "self_attention_viz_reference_support.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(mask):
        raise ValueError("Audit and self-attention reference frame counts differ")
    query_audit = query_audit or audit
    query_support = torch.load(query_audit / "reference_support.pt", map_location="cpu", weights_only=False)
    query_metadata = json.loads((query_audit / "region_audit.json").read_text())
    query_reviews = {int(row["latent_frame"]): row for row in query_metadata.get("frames", [])}
    if query_reviews.get(0, {}).get("valid_for_object_metrics") is not True:
        raise ValueError("First-frame query support requires explicit positive visual review")
    query_mask = torch.as_tensor(query_support["support_mask_fhw"][0], dtype=torch.bool)
    center = query_support["center_trajectory"][0]
    expected = [float(center[0]), float(center[1]), float(query_support["support_radius_per_frame"][0]), int(query_mask.sum())]
    actual = [float(rows[0][key]) for key in ("center_y", "center_x", "support_radius", "support_area")]
    if query_mask.shape != mask[0].shape or not np.allclose(expected, actual, atol=1e-5, rtol=1e-5):
        raise ValueError("Query audit differs from the support used to collect first-frame self-attention")
    mask = mask.clone()
    mask[0] = query_mask
    selected = {tuple(map(int, key)): value.float() for key, value in maps.items() if int(key[2]) == 0}
    if not selected:
        raise ValueError(f"No first-frame query maps in {directory}")
    for key, value in selected.items():
        if value.ndim != 4 or tuple(value.shape[1:]) != tuple(mask.shape):
            raise ValueError(f"Map/mask shape mismatch at {key}")
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"Invalid attention probabilities at {key}")
        if not torch.allclose(value.sum((1, 2, 3)), torch.ones(value.shape[0]), atol=2e-3, rtol=2e-3):
            raise ValueError(f"Attention does not sum to one at {key}")
    return selected, mask, {"self_attention": metadata, "audit": audit_metadata, "query_audit": query_metadata, "query_audit_path": str(query_audit.resolve())}


def frame_metrics(attention: torch.Tensor, mask: torch.Tensor, valid_frames: set[int] | None = None) -> list[dict]:
    """Measure global/conditional region masses and densities, excluding query frame 0."""
    rows = []
    same = mask[0]
    for frame in range(1, len(mask)):
        if valid_frames is not None and frame not in valid_frames:
            continue
        object_mask = mask[frame]
        overlap = object_mask & same
        union = object_mask | same
        regions = {"object": object_mask, "same_coordinate": same, "overlap": overlap,
                   "object_exclusive": object_mask & ~same, "same_coordinate_exclusive": same & ~object_mask}
        values = attention[frame]
        frame_mass = float(values.sum())
        row = {"key_token_frame": frame, "frame_global_mass": frame_mass,
               "region_iou": float(overlap.sum()) / max(1, int(union.sum()))}
        for name, region in regions.items():
            area = int(region.sum())
            mass = float(values[region].sum())
            conditional = mass / frame_mass if frame_mass > 0 else float("nan")
            row.update({f"{name}_area": area, f"{name}_global_mass": mass,
                        f"{name}_conditional_mass": conditional,
                        f"{name}_global_density": mass / area if area else float("nan"),
                        f"{name}_conditional_density": conditional / area if area else float("nan")})
        row["exclusive_density_contrast"] = row["object_exclusive_conditional_density"] - row["same_coordinate_exclusive_conditional_density"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty rectangular table with stable column names."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    """Average finite per-frame metrics and retain summed cross-frame probability masses."""
    result = {"key_frame_count": len(rows), "valid_exclusive_frame_count": sum(np.isfinite(row["exclusive_density_contrast"]) for row in rows)}
    for field in rows[0]:
        if field == "key_token_frame":
            continue
        values = np.asarray([row[field] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result["mean_" + field] = float(finite.mean()) if len(finite) else float("nan")
        if field.endswith("global_mass"):
            result["sum_" + field] = float(values.sum())
    return result


def plot_maps(output: Path, conditions: dict, masks: dict, keys: list[tuple], layer: int, valid_frames: set[int]) -> None:
    """Render the illustrative slice using the shared two-row, ten-frame renderer.

    Masks contain the verified query support in frame zero. Reviewed nonquery
    supports are outlined only when valid in both conditions. The full atlas
    runner separately loads the independently reviewed frame-zero key support.
    """
    from wan21_t2v_experiments.rope_posthoc_visualize import render_pair
    from wan21_t2v_experiments.utils import _resolve_wan21_t2v_viz_frame_indices
    names = ("baseline", "modified")
    frames = len(masks["baseline"])
    indices, _ = _resolve_wan21_t2v_viz_frame_indices(frames, 81, 10)
    seed = output.name.removeprefix("seed_")
    for key in keys:
        step, current_layer, _ = key
        if current_layer != layer or step not in range(1, 6):
            continue
        render_pair(dict(seed=seed, step=step, layer=layer, output=str(output),
            attention=np.stack([conditions[name][key].mean(0)[indices].numpy() for name in names]),
            object_masks=np.stack([masks[name][indices].numpy() for name in names]),
            query_masks=np.stack([masks[name][0].numpy() for name in names]),
            object_valid=np.asarray([[i == 0 or i in valid_frames for i in indices] for name in names])))


def plot_trends(output: Path, rows: list[dict], layer: int, save_name: str = "step_trends.png", paired_rows: list[dict] | None = None) -> None:
    """Plot head-mean trends without assuming the intervention improves either metric."""
    fields = [("sum_object_global_mass", "Total valid-frame object mass"),
              ("mean_object_conditional_mass", "Mean within-frame object probability"),
              ("mean_exclusive_density_contrast", "Exclusive object − same-coordinate density")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for condition in ("baseline", "modified"):
        selected = sorted((row for row in rows if row["condition"] == condition and row["head"] == "mean" and row["layer"] == layer), key=lambda row: row["step"])
        for ax, (field, label) in zip(axes, fields):
            values = [row[field] for row in selected]
            if field == "mean_exclusive_density_contrast" and paired_rows is not None:
                grouped = {}
                for paired in paired_rows:
                    if paired["layer"] == layer:
                        grouped.setdefault(paired["step"], []).append(paired[condition])
                values = [float(np.mean(grouped[row["step"]])) if row["step"] in grouped else float("nan") for row in selected]
            ax.plot([row["step"] for row in selected], values, marker="o", markersize=3, label=condition)
            ax.set(xlabel="Denoising step (1-based)", ylabel=label)
            ax.axvspan(1, 5, color="grey", alpha=0.1)
            ax.grid(alpha=0.2)
    axes[-1].axhline(0, color="black", linewidth=0.5)
    axes[0].legend()
    fig.suptitle(f"Layer {layer}; first-frame object queries; key frame 0 excluded")
    fig.tight_layout()
    fig.savefig(output / save_name, dpi=160)
    plt.close(fig)


def main() -> None:
    """Compare paired artifact directories and audited supports entirely on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "modified", "baseline-audit", "modified-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-query-audit", type=Path)
    parser.add_argument("--modified-query-audit", type=Path)
    parser.add_argument("--plot-layer", type=int, default=27)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    conditions, masks, metadata = {}, {}, {}
    for name in ("baseline", "modified"):
        conditions[name], masks[name], metadata[name] = load_condition(getattr(args, name), getattr(args, name + "_audit"), getattr(args, name + "_query_audit"))
    keys = sorted(set(conditions["baseline"]) & set(conditions["modified"]))
    if not keys:
        raise ValueError("No shared step/layer/first-frame-query keys")
    if masks["baseline"].shape != masks["modified"].shape:
        raise ValueError("Paired conditions require identical latent grid shapes")
    reviews = {name: {int(row["latent_frame"]): row for row in data["audit"].get("frames", [])} for name, data in metadata.items()}
    valid_frames = {frame for frame in range(1, len(masks["baseline"]))
                    if all(reviews[name].get(frame, {}).get("valid_for_object_metrics") is True for name in conditions)}
    if not valid_frames:
        raise ValueError("No nonquery key frames have explicit positive visual review in both conditions")
    review_rows = [{"condition": name, "key_token_frame": frame,
                    "valid_for_object_metrics": reviews[name].get(frame, {}).get("valid_for_object_metrics", False),
                    "review_status": reviews[name].get(frame, {}).get("review_status", "unreviewed"),
                    "included_in_paired_metrics": frame in valid_frames}
                   for name in conditions for frame in range(1, len(masks[name]))]
    write_csv(args.output / "frame_inclusion.csv", review_rows)
    frame_rows, summary_rows = [], []
    for key in keys:
        if conditions["baseline"][key].shape != conditions["modified"][key].shape:
            raise ValueError(f"Paired head/map shape mismatch at {key}")
        for condition, maps in conditions.items():
            for head, attention in list(enumerate(maps[key])) + [("mean", maps[key].mean(0))]:
                identity = {"condition": condition, "step": key[0], "layer": key[1], "head": head, "query_token_frame": 0}
                rows = frame_metrics(attention, masks[condition], valid_frames)
                frame_rows.extend(dict(identity, **row) for row in rows)
                summary_rows.append(dict(identity, **summarize(rows)))
    write_csv(args.output / "per_frame_metrics.csv", frame_rows)
    write_csv(args.output / "per_head_summary.csv", summary_rows)
    plot_maps(args.output, conditions, masks, keys, args.plot_layer, valid_frames)
    plot_trends(args.output, summary_rows, args.plot_layer)
    report = {"inputs": {name: str(getattr(args, name).resolve()) for name in ("baseline", "modified", "baseline_audit", "modified_audit")},
              "valid_paired_key_frames": sorted(valid_frames), "common_map_keys": keys, "unpaired_map_keys": {name: sorted(set(maps) - set(keys)) for name, maps in conditions.items()},
              "metadata": metadata, "head_indices": "zero-based; mean denotes probability maps averaged before normalization",
              "interpretation": "Each condition uses its own audited late-step reference masks and first-frame query support. Different supports, latent trajectories, and finetuned weights confound an isolated causal interpretation of RoPE. Empty exclusive regions have undefined densities and are excluded from finite frame means. Global masses include only jointly valid reviewed nonquery key frames without renormalizing away excluded-frame or query-frame mass. Key audit supports may differ from query extraction settings; the first-frame query audit must match the collected support."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
