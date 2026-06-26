"""Visualization helpers for Wan2.1-T2V experiment assets."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
work_dir = Path(os.getenv("WORK_DIR", PROJECT_ROOT.as_posix()))


def _ensure_dir(path: str) -> None:
    """Create `path` when it does not already exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def _load_video_frames(video_path: str) -> List[object]:
    """Load RGB frames from a video using available local backends.

    The function prefers `decord`, which is already used in the Wan2.1 codebase,
    and falls back to `torchvision.io.read_video`.
    """
    try:
        import decord

        vr = decord.VideoReader(video_path)
        if len(vr) <= 0:
            raise ValueError(f"Decoded video has no frames: {video_path}")
        batch = vr.get_batch(list(range(len(vr)))).asnumpy()
        return [batch[i] for i in range(batch.shape[0])]
    except ImportError:
        pass

    try:
        import torchvision

        frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        if frames.ndim != 4 or int(frames.shape[0]) <= 0:
            raise ValueError(
                f"Expected decoded video frames with shape [F, H, W, C], got {tuple(frames.shape)} "
                f"from {video_path}"
            )
        frames = frames.cpu().numpy()
        return [frames[i] for i in range(frames.shape[0])]
    except ImportError as exc:
        raise ImportError(
            "Neither `decord` nor `torchvision` is available for MP4 decoding."
        ) from exc


def _resolve_wan21_t2v_video_timeline_indices(
    video_frame_count: int,
    num_frames: int = 10,
) -> List[int]:
    """Resolve default 0-based video-frame indices for timeline visualization.

    The rule matches `wan21_t2v_experiments.utils._resolve_wan21_t2v_viz_frame_indices`:
    sample uniformly in real-video frame space, always include the first and
    last frame, and use integer interval `(F_video - 1) // (n - 1)`.
    """
    if video_frame_count <= 0:
        return []

    n = min(max(1, int(num_frames)), int(video_frame_count))
    if n == 1:
        return [0]

    interval = max(1, (int(video_frame_count) - 1) // (n - 1))
    indices = [i * interval for i in range(n - 1)]
    indices.append(int(video_frame_count) - 1)
    return indices


def save_wan21_t2v_video_timeline_pdf(
    video_path: str,
    save_file: str,
    num_frames: int = 10,
    title: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """Render a one-row PDF timeline by sampling frames from an MP4 video.

    Args:
        video_path: Path to the source MP4 video.
        save_file: Output PDF path.
        num_frames: Number of displayed frames. The default `10` matches the
            default cross-attention timeline configuration.
        title: Optional figure title. When omitted, the base filename is used.

    Returns:
        A tuple `(frame_indices_0based, frame_labels_1based)` describing the
        sampled real-video frames.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video_path = str(Path(video_path).expanduser().resolve())
    save_file = str(Path(save_file).expanduser().resolve())

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    frames = _load_video_frames(video_path)
    video_frame_count = len(frames)
    sampled_indices = _resolve_wan21_t2v_video_timeline_indices(
        video_frame_count=video_frame_count,
        num_frames=int(num_frames),
    )
    if not sampled_indices:
        raise ValueError(f"No frames available for visualization: {video_path}")

    sampled_labels = [int(idx) + 1 for idx in sampled_indices]
    sampled_frames = [frames[int(idx)] for idx in sampled_indices]

    num_panels = len(sampled_frames)
    fig = plt.figure(figsize=(2.25 * num_panels, 2.8))
    gs = fig.add_gridspec(1, num_panels, wspace=0.01, hspace=0.0)
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_panels)]

    for panel_idx, frame in enumerate(sampled_frames):
        ax = axes[panel_idx]
        ax.imshow(frame)
        ax.set_title(f"frame={sampled_labels[panel_idx]}", fontsize=14)
        ax.axis("off")

    figure_title = str(title).strip() if title is not None else ""
    if not figure_title:
        figure_title = os.path.basename(video_path)
    fig.suptitle(figure_title, fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.01, hspace=0.0)

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)

    return sampled_indices, sampled_labels


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    save_wan21_t2v_video_timeline_pdf(
        video_path=str(work_dir / "outputs_wan_2_1_t2v-1.3B/cross_attention_token_viz/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./seed_8_shift_5.0_guide_5.0/wan21_t2v_cross_attention_token_viz_seed_8.mp4"),
        save_file=str(work_dir / "general_viz/basketball_seed8.pdf"),
    )
