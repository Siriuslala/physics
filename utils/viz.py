"""Visualization helpers for Wan2.1-T2V experiment assets."""

import os
import re
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


def _natural_sort_key(path: Path) -> List[object]:
    """Return a filename sort key that treats digit runs as integers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _parse_last_integer(text: str) -> Optional[int]:
    """Parse the last integer in `text`, or return `None` when absent."""
    matches = re.findall(r"\d+", text)
    if not matches:
        return None
    return int(matches[-1])


def _find_diffusion_step_dirs(diffusion_frames_dir: Path) -> List[Tuple[int, Path]]:
    """Find diffusion-step subdirectories and pair each with its numeric step id."""
    step_dirs: List[Tuple[int, Path]] = []
    for child in sorted(diffusion_frames_dir.iterdir(), key=_natural_sort_key):
        if not child.is_dir():
            continue
        step_id = _parse_last_integer(child.name)
        if step_id is None:
            continue
        step_dirs.append((step_id, child))
    return step_dirs


def _find_frame_image_files(step_dir: Path) -> List[Path]:
    """Find frame images under one diffusion-step directory."""
    valid_suffixes = {".png", ".jpg", ".jpeg", ".pdf"}
    return sorted(
        [
            path
            for path in step_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_suffixes
        ],
        key=_natural_sort_key,
    )


def _load_rgb_image(image_path: Path):
    """Load a PNG/JPEG/PDF frame as an RGB array suitable for `imshow`.

    PDF support uses PyMuPDF (`fitz`) when available and falls back to
    `pypdfium2`. Only the first page is rendered, which matches the expected
    one-frame-per-file layout of latent-decoded frame exports.
    """
    suffix = image_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
            import numpy as np

            document = fitz.open(str(image_path))
            if document.page_count <= 0:
                raise ValueError(f"PDF has no pages: {image_path}")
            pixmap = document[0].get_pixmap(alpha=False)
            array = np.frombuffer(pixmap.samples, dtype=np.uint8)
            return array.reshape(pixmap.height, pixmap.width, pixmap.n)
        except ImportError:
            pass

        try:
            import numpy as np
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(str(image_path))
            if len(document) <= 0:
                raise ValueError(f"PDF has no pages: {image_path}")
            bitmap = document[0].render(scale=1.0)
            return np.asarray(bitmap.to_pil().convert("RGB"))
        except ImportError as exc:
            raise ImportError(
                "PDF frame loading requires either `PyMuPDF` (`fitz`) or `pypdfium2`."
            ) from exc

    from PIL import Image
    import numpy as np

    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def _resize_rgb_image(image, image_max_size: Optional[int] = None):
    """Downsample an RGB image array by capping its longest side."""
    if image_max_size is None:
        return image

    max_size = int(image_max_size)
    if max_size <= 0:
        raise ValueError("image_max_size must be positive when provided.")

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image

    from PIL import Image
    import numpy as np

    scale = float(max_size) / float(longest)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized_image = Image.fromarray(image).resize(
        resized_size,
        Image.Resampling.LANCZOS,
    )
    return np.asarray(resized_image)


def save_wan21_t2v_diffusion_steps_timeline_pdf(
    diffusion_frames_dir: str,
    save_file: str,
    diffusion_sample_count: int = 10,
    diffusion_summary_every: int = 10,
    image_max_size: Optional[int] = None,
    save_dpi: int = 150,
) -> Tuple[List[int], List[int]]:
    """Render a PDF grid from latent-decoded frames across diffusion steps.

    Args:
        diffusion_frames_dir: Directory whose subdirectories store per-step
            decoded frame images, such as `nfe_001/frame_00000.png`.
        save_file: Output PDF path.
        diffusion_sample_count: Number of video frames sampled uniformly from
            each selected step. The sampling rule exactly reuses
            `_resolve_wan21_t2v_video_timeline_indices`.
        diffusion_summary_every: Diffusion-step interval for row selection.
            For a 50-step export and interval 10, rows are `T10, ..., T50`.
        image_max_size: Optional cap for each embedded frame image's longest
            side, in pixels. Smaller values reduce PDF size at the cost of
            visual detail. When omitted, source image resolution is preserved.
        save_dpi: DPI passed to `matplotlib.figure.Figure.savefig`.

    Returns:
        A tuple `(sampled_diffusion_steps, sampled_frame_labels_1based)`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diffusion_dir = Path(diffusion_frames_dir).expanduser().resolve()
    save_file = str(Path(save_file).expanduser().resolve())

    if not diffusion_dir.exists():
        raise FileNotFoundError(
            f"Diffusion frame directory does not exist: {diffusion_dir}"
        )
    if not diffusion_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {diffusion_dir}")

    step_every = int(diffusion_summary_every)
    if step_every <= 0:
        raise ValueError("diffusion_summary_every must be a positive integer.")

    step_dirs = _find_diffusion_step_dirs(diffusion_dir)
    if not step_dirs:
        raise ValueError(
            f"No numeric diffusion-step subdirectories found in: {diffusion_dir}"
        )

    step_dir_by_id = {int(step_id): step_dir for step_id, step_dir in step_dirs}
    max_step = max(step_dir_by_id)
    sampled_steps = [
        step_id
        for step_id in range(step_every, max_step + 1, step_every)
        if step_id in step_dir_by_id
    ]
    if not sampled_steps:
        raise ValueError(
            f"No diffusion-step directories matched interval {step_every} under {diffusion_dir}."
        )

    reference_files = _find_frame_image_files(step_dir_by_id[sampled_steps[0]])
    if not reference_files:
        raise ValueError(
            f"No frame images found in: {step_dir_by_id[sampled_steps[0]]}"
        )

    sampled_frame_indices = _resolve_wan21_t2v_video_timeline_indices(
        video_frame_count=len(reference_files),
        num_frames=int(diffusion_sample_count),
    )
    if not sampled_frame_indices:
        raise ValueError(f"No frame images available for visualization: {diffusion_dir}")

    sampled_frame_labels = [int(idx) + 1 for idx in sampled_frame_indices]
    num_rows = len(sampled_steps)
    num_cols = len(sampled_frame_indices)

    fig = plt.figure(figsize=(2.25 * num_cols + 0.35, 1.4 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols, wspace=0.01, hspace=0.0)

    for row_idx, step_id in enumerate(sampled_steps):
        frame_files = _find_frame_image_files(step_dir_by_id[step_id])
        if not frame_files:
            raise ValueError(f"No frame images found in: {step_dir_by_id[step_id]}")
        if max(sampled_frame_indices) >= len(frame_files):
            raise ValueError(
                f"Step T{step_id} has only {len(frame_files)} frames, but the sampled "
                f"indices require frame index {max(sampled_frame_indices)}."
            )

        for col_idx, frame_index in enumerate(sampled_frame_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            frame_image = _load_rgb_image(frame_files[int(frame_index)])
            frame_image = _resize_rgb_image(
                frame_image,
                image_max_size=image_max_size,
            )
            ax.imshow(frame_image)
            ax.axis("off")
            if col_idx == 0:
                ax.text(
                    -0.035,
                    0.5,
                    f"T{step_id}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=20,
                )

    fig.subplots_adjust(
        left=0.045, right=0.995, top=0.995, bottom=0.005, wspace=0.01, hspace=0.0
    )

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf", dpi=int(save_dpi))
    plt.close(fig)

    return sampled_steps, sampled_frame_labels


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
        video_path=str(work_dir / "/work/liyueyan/Interpretability/physics/outputs_wan_2_1_t2v-1.3B/cross_attn_head_ablation/Against_a_pure_white_background,_there_is_a_wooden_horizontal_surface,_with_one_single_wooden_slope_attached_to_its_left_end._One_small_green_ball_starts_from_rest_at_the_top_of_the_slope,_slides_straight_along_the_slope_the_entire_time_with_its_speed_inc/seed_2_shift_5.0_guide_5.0/ablate_traj_new_speed_gt0p2_contri_lt0p1-steps_all_steps/wan21_t2v_cross_attn_head_ablation_steps_all_seed_2.mp4"),
        save_file=str(work_dir / "wan_eval_viz/ball_seed2_ablate_new_speed_gt0p2_contri_lt0p1.pdf"),
    )

    # save_wan21_t2v_diffusion_steps_timeline_pdf(
    #     diffusion_frames_dir=str(work_dir / "/work/liyueyan/Interpretability/physics/outputs_wan_2_1_t2v-1.3B_new/general_a800/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./t2v-1.3B_832*480_diffusion_steps"),
    #     save_file=str(work_dir / "general_viz/basketball_seed26_denoising.pdf"),
    #     diffusion_sample_count=10,
    #     diffusion_summary_every=10,
    #     image_max_size=320,
    #     save_dpi=100,
    # )
