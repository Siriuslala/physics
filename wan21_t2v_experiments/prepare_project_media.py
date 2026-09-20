"""Prepare synchronized, full-length before/after media for the public project page."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    (f"basketball_seed_{seed}", f"basketball-seed-{seed}", f"Basketball free-fall · Seed {seed}")
    for seed in (8, 20, 23, 29)
] + [
    ("videophy_Cork being twisted out of a bottle.", "videophy-cork", "Cork being twisted out of a bottle."),
    ("videophy_A large log floats downstream in a rushing river.", "videophy-log", "A large log floats downstream in a rushing river."),
    ("videophy_Refrigerator door closing after getting a soda.", "videophy-refrigerator", "Refrigerator door closing after getting a soda."),
    ("videophy_Wine pouring from a bottle into a glass.", "videophy-wine", "Wine pouring from a bottle into a glass."),
    ("videophy_Spatula flips pancake in air.", "videophy-pancake", "Spatula flips pancake in air."),
    ("videophy_A car gliding over a road slick with rainwater.”", "videophy-car", "A car gliding over a road slick with rainwater."),
]


def prepare_example(source, source_name, slug, title, ffmpeg):
    """Move originals without overwriting; export a synchronized MP4, GIF and poster.

    Inputs are the source directory, original folder name, public slug, caption,
    and FFmpeg executable. Original bytes are preserved and hashed in the returned
    manifest record. Both clips must share dimensions, frame rate, and duration.
    Videos are composed left/right at 416 pixels per panel without cropping or
    retiming. GIFs use 12 fps; the MP4 retains the original frame rate and duration.
    """
    target = ROOT / "assets" / "videos" / slug
    target.mkdir(parents=True, exist_ok=True)
    metadata = []
    originals = {}
    for name in ("before", "after"):
        src, dst = source / source_name / f"{name}.mp4", target / f"{name}.mp4"
        if not dst.exists():
            shutil.move(str(src), str(dst))
        elif src.exists() and src.read_bytes() != dst.read_bytes():
            raise ValueError(f"Refusing to overwrite a different original: {dst}")
        reader = imageio_ffmpeg.read_frames(str(dst))
        try:
            metadata.append(next(reader))
        finally:
            reader.close()
        originals[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    before, after = metadata
    if any(before[key] != after[key] for key in ("size", "fps", "duration")):
        raise ValueError(f"Mismatched comparison timing or size: {slug}: {metadata}")

    width = 416
    height = round(width * before["size"][1] / before["size"][0] / 2) * 2
    label = Image.new("RGB", (2 * width, 36), "#f4f5f7")
    draw = ImageDraw.Draw(label)
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    draw.text((16, 8), "Before · Original model", fill="#50545c", font=font)
    draw.text((width + 16, 8), "After · Ours", fill="#225d4b", font=font)
    label_path = target / "labels.png"
    label.save(label_path)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    filters = (
        f"[0:v]setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos,setsar=1[b];"
        f"[1:v]setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos,setsar=1[a];"
        "[b][a]hstack=inputs=2[pair];[2:v][pair]vstack=inputs=2[out]"
    )
    comparison = target / "comparison.mp4"
    subprocess.run(command + [
        "-i", str(target / "before.mp4"), "-i", str(target / "after.mp4"),
        "-framerate", str(before["fps"]), "-i", str(label_path),
        "-filter_complex_threads", "1", "-filter_complex", filters,
        "-map", "[out]", "-an", "-r", str(before["fps"]),
        "-c:v", "libx264", "-crf", "20", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(comparison),
    ], check=True)
    subprocess.run(command + [
        "-i", str(comparison), "-filter_complex_threads", "1", "-filter_complex",
        "fps=12,scale=640:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=192:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
        "-loop", "0", str(target / "comparison.gif"),
    ], check=True)
    subprocess.run(command + [
        "-i", str(comparison), "-frames:v", "1", str(target / "poster.jpg"),
    ], check=True)
    label_path.unlink()
    return {"id": slug, "title": title, "source_folder": source_name,
            "size": before["size"], "fps": before["fps"],
            "duration_seconds": before["duration"], "sha256": originals}


def main():
    """Parse the import directory and build all ten comparisons plus provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing before/after folders.")
    args = parser.parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records = []
    for source_name, slug, title in EXAMPLES:
        records.append(prepare_example(args.source, source_name, slug, title, ffmpeg))
        print(f"Prepared {slug}", flush=True)
    (ROOT / "assets" / "videos" / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
