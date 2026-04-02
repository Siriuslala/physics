import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_NFE_DIR_RE = re.compile(r"^nfe_(\d+)$")
_FRAME_FILE_RE = re.compile(r"^frame_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def visualize_diffusion_overview(
    infer_output_dir: str,
    save_file: Optional[str] = None,
    sampling_times_sec: Optional[List[float]] = None,
    sampling_nfes: Optional[List[int]] = None,
    diffusion_sample_count: int = 5,
    diffusion_summary_every: int = 10,
    video_fps: float = 24.0,
) -> str:
    """
    Build an overview image from an infer output directory.

    Expected folder structure:
    - infer_output_dir/
      - nfe_001/frame_00000.png ...
      - nfe_002/frame_00000.png ...
      - ...

    Args:
        infer_output_dir:
            Root folder that contains per-NFE subfolders.
        save_file:
            Output path for the overview image. If None, save to
            infer_output_dir/nfe_summary_matplotlib.png.
        sampling_times_sec:
            Optional list of video times (in seconds) to visualize.
            If None, choose times using diffusion_sample_count.
        sampling_nfes:
            Optional list of NFE steps to visualize.
            If None, choose NFEs using diffusion_summary_every.
        diffusion_sample_count:
            Number of time columns when sampling_times_sec is None.
        diffusion_summary_every:
            NFE interval when sampling_nfes is None.
        video_fps:
            Video FPS for converting time (seconds) to frame index.

    Returns:
        The saved overview image path.
    """
    if diffusion_sample_count <= 0:
        raise ValueError("diffusion_sample_count must be positive.")
    if diffusion_summary_every <= 0:
        raise ValueError("diffusion_summary_every must be positive.")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive.")

    nfe_to_dir: Dict[int, str] = {}
    for name in os.listdir(infer_output_dir):
        path = os.path.join(infer_output_dir, name)
        if not os.path.isdir(path):
            continue
        match = _NFE_DIR_RE.match(name)
        if match is None:
            continue
        nfe = int(match.group(1))
        nfe_to_dir[nfe] = path

    if not nfe_to_dir:
        raise ValueError(
            f"No NFE subdirectories found under: {infer_output_dir}. "
            "Expected names like nfe_001, nfe_002, ..."
        )

    frame_maps: Dict[int, Dict[int, str]] = {}
    for nfe, folder in nfe_to_dir.items():
        frame_map: Dict[int, str] = {}
        for name in os.listdir(folder):
            match = _FRAME_FILE_RE.match(name)
            if match is None:
                continue
            frame_idx = int(match.group(1))
            frame_map[frame_idx] = os.path.join(folder, name)
        if frame_map:
            frame_maps[nfe] = frame_map

    if not frame_maps:
        raise ValueError(
            f"No frame files found under NFE folders in: {infer_output_dir}. "
            "Expected names like frame_00000.png"
        )

    available_nfes = sorted(frame_maps.keys())

    if sampling_nfes is None:
        selected_nfes = [nfe for nfe in available_nfes if nfe % diffusion_summary_every == 0]
        if not selected_nfes:
            selected_nfes = [available_nfes[-1]]
    else:
        selected_nfes = sorted(set(sampling_nfes))
        missing_nfes = [n for n in selected_nfes if n not in frame_maps]
        if missing_nfes:
            raise ValueError(
                f"Requested NFEs not found in output folder: {missing_nfes}. "
                f"Available NFEs: {available_nfes}"
            )

    first_nfe = selected_nfes[0]
    first_indices = sorted(frame_maps[first_nfe].keys())
    if not first_indices:
        raise ValueError(f"No frames found in NFE folder: {nfe_to_dir[first_nfe]}")
    frame_num = first_indices[-1] + 1

    if sampling_times_sec is None:
        sampled_frame_indices = _default_sampled_frame_indices(
            frame_num=frame_num,
            sample_count=diffusion_sample_count,
            video_fps=video_fps,
        )
        sampled_times_sec = [idx / video_fps for idx in sampled_frame_indices]
    else:
        sampled_times_sec = list(sampling_times_sec)
        sampled_frame_indices = [
            max(0, int(round(time_sec * video_fps))) for time_sec in sampled_times_sec
        ]

    rows = len(selected_nfes)
    cols = len(sampled_frame_indices)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(1, cols) * 3.2, max(1, rows) * 2.6),
        squeeze=False,
    )
    fig.suptitle("sampling timestep (NFE) vs time (s)")

    for row_idx, nfe in enumerate(selected_nfes):
        frame_map = frame_maps[nfe]
        for col_idx, target_idx in enumerate(sampled_frame_indices):
            path, resolved_idx = _resolve_frame_path(frame_map, target_idx)
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB"))

            ax = axes[row_idx, col_idx]
            ax.imshow(arr)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"time={sampled_times_sec[col_idx]:.2f}s")
            if col_idx == 0:
                ax.set_ylabel(f"NFE={nfe}")
            if resolved_idx != target_idx:
                ax.set_xlabel(f"frame={resolved_idx}", fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_file is None:
        save_file = os.path.join(infer_output_dir, "nfe_summary_matplotlib.png")
    os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
    fig.savefig(save_file, dpi=220)
    plt.close(fig)
    return save_file


def _default_sampled_frame_indices(
    frame_num: int,
    sample_count: int,
    video_fps: float,
) -> List[int]:
    if frame_num <= 0:
        return []
    if sample_count == 1:
        return [0]

    max_required = int(round((sample_count - 1) * video_fps))
    if frame_num - 1 >= max_required:
        return [min(int(round(i * video_fps)), frame_num - 1) for i in range(sample_count)]

    return (
        np.linspace(0, frame_num - 1, num=sample_count)
        .round()
        .astype(int)
        .tolist()
    )


def _resolve_frame_path(frame_map: Dict[int, str], target_idx: int) -> Tuple[str, int]:
    if target_idx in frame_map:
        return frame_map[target_idx], target_idx

    indices = sorted(frame_map.keys())
    if target_idx <= indices[0]:
        idx = indices[0]
    elif target_idx >= indices[-1]:
        idx = indices[-1]
    else:
        idx = min(indices, key=lambda x: abs(x - target_idx))
    return frame_map[idx], idx


if __name__ == "__main__":

    visualize_diffusion_overview(
        infer_output_dir="/path/to/your_infer_output",
        diffusion_sample_count=5,
        diffusion_summary_every=10,
        video_fps=24.0,
    )

    # 自定义时刻和 NFE
    visualize_diffusion_overview(
        infer_output_dir="/path/to/your_infer_output",
        sampling_times_sec=[0, 1, 2, 3, 4],
        sampling_nfes=[10, 20, 30, 40, 50],
        video_fps=24.0,
        save_file="/path/to/custom_summary.png",
    )
