"""Render the exact self-attention reference masks on cross-attention and video.

Run as ``python -m wan21_t2v_experiments.region_audit --help``. This CPU-only
audit reuses the production region extractor, so its overlays depict the masks
that self_attention_viz will actually use with the same configuration.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .self_attention_viz import _build_wan21_t2v_self_attention_viz_reference_support


def audit_regions(cross_dir, output_dir, reference_step, reference_layer=27,
                  center_quantile=0.8, center_mode="geometric_center",
                  support_radius_alpha=1.5, support_radius_mode="adaptive_area",
                  support_radius_fixed=2.0, video_path=None):
    """Save frame-wise overlays and a machine-readable region configuration.

    Inputs identify a cross-attention artifact directory and the production
    extraction parameters. Outputs are PNG sheets, JSON metadata, and a tensor
    cache containing the exact [F,H,W] boolean masks and reference maps. Video
    frame 4*f (zero-based) is shown for latent frame f as a temporal reference;
    VAE receptive fields mean this is not a pixel-exact latent correspondence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(
        reuse_cross_attention_dir=str(cross_dir), target_object_words=["basketball"],
        target_verb_words=["falls", "bounces", "up"], reference_step=reference_step,
        reference_layer=reference_layer, center_mode=center_mode, center_power=1.5,
        center_quantile=center_quantile, preprocess_winsorize_quantile=0.995,
        preprocess_despike_quantile=0.98, preprocess_min_component_area=2,
        support_radius_mode=support_radius_mode, support_radius_fixed=support_radius_fixed,
        support_radius_alpha=support_radius_alpha, support_radius_min=1.0,
        support_radius_max_ratio=0.25,
    )
    support = _build_wan21_t2v_self_attention_viz_reference_support(**config)
    maps = support["reference_map"].float().numpy()
    masks = support["support_mask_fhw"].numpy()
    centers = support["center_trajectory"]
    count, height, width = maps.shape
    videos = sorted(Path(cross_dir).glob("*.mp4"))
    selected_video = Path(video_path) if video_path else (videos[0] if videos else None)
    capture = cv2.VideoCapture(str(selected_video)) if selected_video else None
    frames = []
    if capture:
        for index in range(count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 4 * index)
            success, frame = capture.read()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if success else None)
        capture.release()
    rows = []
    for index, (cy, cx) in enumerate(centers):
        rows.append(dict(latent_frame=index, video_frame=4*index+1, center_y=cy,
                         center_x=cx, support_area=int(masks[index].sum()),
                         radius=float(support["support_radius_per_frame"][index]),
                         valid_for_object_metrics=False, review_status="unreviewed"))
    for kind in ("cross_attention", "video"):
        if kind == "video" and not frames:
            continue
        panel_rows = (count + 6) // 7
        fig, axes = plt.subplots(panel_rows, 7, figsize=(21, 2.7 * panel_rows), squeeze=False)
        for index, axis in enumerate(axes.flat):
            if index >= count:
                axis.axis("off")
                continue
            if kind == "cross_attention":
                axis.imshow(maps[index], cmap="viridis", origin="upper")
            elif frames[index] is not None:
                axis.imshow(frames[index], extent=(-0.5, width-0.5, height-0.5, -0.5))
            axis.contour(masks[index], levels=[0.5], colors="red", linewidths=1.1)
            cy, cx = centers[index]
            axis.plot(cx, cy, "+", color="cyan", markersize=7)
            axis.set_title(f"latent {index}; video {4*index+1}; area {rows[index]['support_area']}", fontsize=8)
            axis.set_xlim(-0.5, width-0.5)
            axis.set_ylim(height-0.5, -0.5)
            axis.axis("off")
        fig.suptitle(f"{Path(cross_dir).name}: step {reference_step}, layer {reference_layer}, q={center_quantile}, {center_mode}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{kind}_regions.png", dpi=150)
        plt.close(fig)
    torch.save(support, output_dir / "reference_support.pt")
    metadata = dict(configuration=config, video=str(selected_video) if selected_video else None,
                    shape=list(maps.shape), frames=rows)
    (output_dir / "region_audit.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(output_dir=str(output_dir), first_frame=rows[0])))
    return metadata


def main():
    """Parse the standalone audit CLI and render its requested region masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-step", type=int, required=True)
    parser.add_argument("--reference-layer", type=int, default=27)
    parser.add_argument("--center-quantile", type=float, default=0.8)
    parser.add_argument("--center-mode", choices=["geometric_center", "centroid", "peak"], default="geometric_center")
    parser.add_argument("--support-radius-alpha", type=float, default=1.5)
    parser.add_argument("--support-radius-mode", choices=["fixed", "adaptive_area"], default="adaptive_area")
    parser.add_argument("--support-radius-fixed", type=float, default=2.0)
    parser.add_argument("--video-path")
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit_regions(**vars(args))


if __name__ == "__main__":
    main()
