#!/usr/bin/env python3
"""Prepare WISA-80K videos for DiffSynth Wan training.

This script keeps DiffSynth-Studio untouched. It extracts WISA-80K zip shards
into a flat shard layout and writes a DiffSynth-compatible metadata file with
at least two columns:

    video,prompt

The output video paths are relative to ``--extract-dir`` so they can be used as
DiffSynth ``--dataset_base_path`` entries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import posixpath
import random
import shutil
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ZIP_DIR = Path(
    "/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/videos"
)
DEFAULT_EXTRACT_DIR = Path(
    "/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data"
)
DEFAULT_META_JSON = Path(
    "/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/"
    "snapshots/66e0fd0d6963a76999d0653b5d2d0e3b5c1442f5/data/wisa-80k.json"
)
DEFAULT_METADATA_OUT = DEFAULT_EXTRACT_DIR / "metadata.csv"
DEFAULT_MANIFEST_OUT = DEFAULT_EXTRACT_DIR / "extraction_manifest.csv"
DEFAULT_CLIP_SECONDS = 5.0
DEFAULT_TARGET_NUM_FRAMES = 81


DYNAMIC_LABELS = {
    "collision",
    "deformation",
    "elastic motion",
    "gas motion",
    "liquid motion",
    "rigid body motion",
}

THERMODYNAMIC_LABELS = {
    "combustion",
    "liquefaction",
    "melting",
    "solidification",
    "vaporization",
    "explosion",
}

OPTIC_LABELS = {
    "interference and diffraction",
    "reflection",
    "refraction",
    "scattering",
    "unnatural light source",
}

PHYSICS_CATEGORY_LABELS = {
    "dynamics": DYNAMIC_LABELS,
    "thermodynamics": THERMODYNAMIC_LABELS,
    "optics": OPTIC_LABELS,
}

CATEGORY_METADATA_FILENAMES = {
    "dynamics": "metadata_dynamics.csv",
    "thermodynamics": "metadata_thermodynamics.csv",
    "optics": "metadata_optics.csv",
}

NO_OBVIOUS_DYNAMICS = "no obvious dynamic phenomenon"


CSV_FIELDNAMES = [
    "video",
    "prompt",
    "video_name",
    "label",
    "width",
    "height",
    "fps",
    "duration",
    "motion_score",
    "motion_score_v2",
    "visual_quality_score",
    "text_bbox_num",
    "text_bbox_ratio",
    "phys_law",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "n0",
    "n1",
    "n2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract WISA-80K videos and build DiffSynth metadata."
    )
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument(
        "--metadata-mode",
        choices=("all", "categories"),
        default="all",
        help=(
            "Metadata output mode. 'all' writes one metadata file for the whole "
            "dataset. 'categories' writes separate dynamics, thermodynamics, "
            "and optics CSV files under --metadata-out."
        ),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=DEFAULT_METADATA_OUT,
        help=(
            "Output CSV path when --metadata-mode all, or output directory "
            "when --metadata-mode categories."
        ),
    )
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument(
        "--inspect-meta",
        action="store_true",
        help="Print meta JSON fields and a few examples.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract zip shards into --extract-dir/<zip_stem>/.",
    )
    parser.add_argument(
        "--build-metadata",
        action="store_true",
        help="Build DiffSynth metadata from meta JSON and extracted mp4 files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run inspect, extract, and build metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned work without writing extracted videos or metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted mp4 files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of zip shards to extract in parallel.",
    )
    parser.add_argument(
        "--limit-zips",
        type=int,
        default=None,
        help="Only process this many zip shards, useful for a smoke test.",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        default=None,
        help="Only write this many metadata rows, useful for a smoke test.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help=(
            "Minimum allowed video duration in seconds. Samples with "
            "duration <= this value are filtered by the shared basic rules."
        ),
    )
    parser.add_argument(
        "--video-check",
        choices=("on", "off"),
        default="on",
        help=(
            "Whether to decode the first frame to filter corrupted videos. "
            "Use 'off' for faster metadata generation when this check is not needed."
        ),
    )
    parser.add_argument(
        "--video-check-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help=(
            "Number of parallel workers used when --video-check on. "
            "Set to 1 for serial checking."
        ),
    )
    parser.add_argument(
        "--label-filter",
        type=str,
        default=None,
        help="Comma-separated label allowlist, e.g. 'collision,rigid body motion'.",
    )
    parser.add_argument(
        "--only-dynamics",
        action="store_true",
        help=(
            "Keep strict dynamics samples. This requires a dynamics label, "
            "q0 not equal to 'no obvious dynamic phenomenon', valid duration, "
            "an openable video, and the dynamics-specific text/numeric filters."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write rows even if the mp4 is not found under --extract-dir.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write metadata as JSONL instead of CSV.",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Read a metadata CSV and export a small sample to JSONL.",
    )
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=None,
        help="Output path for --sample-csv JSONL. Defaults to <csv_stem>_sample.jsonl.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=20,
        help="Number of rows to export from --sample-csv.",
    )
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="Randomly sample rows from --sample-csv instead of taking the first n rows.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used with --sample-random.",
    )
    parser.add_argument(
        "--sample-copy-dir",
        type=Path,
        default=None,
        help="Copy sampled videos into this directory while exporting JSONL.",
    )
    parser.add_argument(
        "--sample-base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory used to resolve relative video paths in --sample-csv. "
            "Defaults to the CSV directory, with a parent-directory fallback."
        ),
    )
    parser.add_argument(
        "--sample-copy-key",
        type=str,
        default="video",
        help="Which CSV field to interpret as the relative video path for copying.",
    )
    parser.add_argument(
        "--summarize-csv",
        type=Path,
        default=None,
        help="Print row count and label distribution for a metadata CSV.",
    )
    parser.add_argument(
        "--merge-category-metadata",
        type=Path,
        default=None,
        help=(
            "Directory containing metadata_dynamics.csv, "
            "metadata_thermodynamics.csv, and metadata_optics.csv to merge."
        ),
    )
    parser.add_argument(
        "--merge-out",
        type=Path,
        default=None,
        help="Output CSV path for --merge-category-metadata.",
    )
    parser.add_argument(
        "--reflection-sample-n",
        type=int,
        default=4000,
        help="Number of reflection rows to keep when merging category metadata.",
    )
    parser.add_argument(
        "--merge-seed",
        type=int,
        default=0,
        help="Random seed used when sampling reflection rows during merge.",
    )
    return parser.parse_args()


def load_meta(meta_json: Path) -> List[Dict[str, Any]]:
    with meta_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {meta_json}, got {type(data).__name__}")
    for i, item in enumerate(data[:10]):
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict records in {meta_json}; item {i} is {type(item).__name__}")
    return data


def inspect_meta(meta_json: Path, num_examples: int = 3) -> None:
    data = load_meta(meta_json)
    key_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    q0_counter: Counter[str] = Counter()
    for item in data:
        key_counter.update(item.keys())
        label_counter[str(item.get("label", ""))] += 1
        annotation = item.get("physical_annotation") or {}
        if isinstance(annotation, dict):
            q0_counter[str(annotation.get("q0", ""))] += 1

    print(f"meta_json: {meta_json}")
    print(f"num_items: {len(data)}")
    print(f"fields: {sorted(key_counter.keys())}")
    print("label_top20:")
    for label, count in label_counter.most_common(20):
        print(f"  {count:6d}  {label}")
    print("q0_top20:")
    for label, count in q0_counter.most_common(20):
        print(f"  {count:6d}  {label}")
    print(f"examples_first_{num_examples}:")
    for item in data[:num_examples]:
        compact = {
            "video_name": item.get("video_name"),
            "label": item.get("label"),
            "width": item.get("width"),
            "height": item.get("height"),
            "fps": item.get("fps"),
            "duration": item.get("duration"),
            "caption_prefix": str(item.get("captions", ""))[:160],
            "physical_annotation": item.get("physical_annotation"),
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2)[:4000])


def sorted_zip_paths(zip_dir: Path, limit_zips: Optional[int] = None) -> List[Path]:
    zip_paths = sorted(
        zip_dir.glob("*.zip"),
        key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem),
    )
    if limit_zips is not None:
        zip_paths = zip_paths[:limit_zips]
    return zip_paths


def safe_video_name(member_name: str) -> str:
    """Return a safe file name for a zip member.

    WISA zip members contain absolute-looking historical paths such as
    ``home/jovyan/.../128_split/0/<hash>.mp4``. DiffSynth only needs a regular
    path under ``dataset_base_path``, so we intentionally keep only the basename.
    """

    normalized = member_name.replace("\\", "/")
    name = posixpath.basename(normalized)
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid zip member name: {member_name!r}")
    if posixpath.basename(name) != name:
        raise ValueError(f"Unsafe zip member name: {member_name!r}")
    return name


def unique_target_path(target_dir: Path, file_name: str, used_names: set[str]) -> Path:
    candidate = file_name
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    counter = 1
    while candidate in used_names:
        candidate = f"{stem}_dup{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return target_dir / candidate


def iter_mp4_members(zip_file: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        if info.filename.lower().replace("\\", "/").endswith(".mp4"):
            yield info


def extract_one_zip(
    zip_path: Path,
    extract_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_dir = extract_dir / zip_path.stem
    rows: List[Dict[str, Any]] = []
    stats = {
        "zip": str(zip_path),
        "members": 0,
        "written": 0,
        "skipped_existing": 0,
        "dry_run": int(dry_run),
    }

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in iter_mp4_members(zf):
            stats["members"] += 1
            video_name = safe_video_name(info.filename)
            target_path = unique_target_path(target_dir, video_name, used_names)
            rel_video = target_path.relative_to(extract_dir).as_posix()
            row = {
                "zip_file": zip_path.name,
                "zip_member": info.filename,
                "video_name": target_path.name,
                "video": rel_video,
                "output_path": str(target_path),
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
            }
            rows.append(row)

            if dry_run:
                continue
            if target_path.exists() and not overwrite:
                stats["skipped_existing"] += 1
                continue
            with zf.open(info, "r") as src, target_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            stats["written"] += 1
    return rows, stats


def write_manifest(rows: Sequence[Dict[str, Any]], manifest_out: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write manifest: {manifest_out} ({len(rows)} rows)")
        return
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "zip_file",
        "zip_member",
        "video_name",
        "video",
        "output_path",
        "compressed_size",
        "file_size",
    ]
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote manifest: {manifest_out} ({len(rows)} rows)")


def extract_zips(args: argparse.Namespace) -> None:
    zip_paths = sorted_zip_paths(args.zip_dir, args.limit_zips)
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found under {args.zip_dir}")

    print(f"zip_dir: {args.zip_dir}")
    print(f"extract_dir: {args.extract_dir}")
    print(f"num_zips: {len(zip_paths)}")
    print(f"num_workers: {args.num_workers}")
    if args.dry_run:
        print("dry_run: true")

    all_rows: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []
    workers = max(1, args.num_workers)
    if workers == 1:
        for zip_path in zip_paths:
            rows, stat = extract_one_zip(zip_path, args.extract_dir, args.overwrite, args.dry_run)
            all_rows.extend(rows)
            stats.append(stat)
            print(
                f"{zip_path.name}: members={stat['members']} "
                f"written={stat['written']} skipped_existing={stat['skipped_existing']}"
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    extract_one_zip,
                    zip_path,
                    args.extract_dir,
                    args.overwrite,
                    args.dry_run,
                ): zip_path
                for zip_path in zip_paths
            }
            for future in as_completed(futures):
                zip_path = futures[future]
                rows, stat = future.result()
                all_rows.extend(rows)
                stats.append(stat)
                print(
                    f"{zip_path.name}: members={stat['members']} "
                    f"written={stat['written']} skipped_existing={stat['skipped_existing']}"
                )

    write_manifest(all_rows, args.manifest_out, args.dry_run)
    print(
        "extract summary: "
        f"members={sum(s['members'] for s in stats)} "
        f"written={sum(s['written'] for s in stats)} "
        f"skipped_existing={sum(s['skipped_existing'] for s in stats)}"
    )


def scan_extracted_videos(extract_dir: Path) -> Dict[str, str]:
    video_to_relpath: Dict[str, str] = {}
    duplicate_names: Counter[str] = Counter()
    for path in sorted(extract_dir.glob("*/*.mp4")):
        rel_path = path.relative_to(extract_dir).as_posix()
        if path.name in video_to_relpath:
            duplicate_names[path.name] += 1
            continue
        video_to_relpath[path.name] = rel_path
    if duplicate_names:
        print(
            "warning: duplicate extracted mp4 basenames ignored: "
            f"{len(duplicate_names)} names, {sum(duplicate_names.values())} extra files",
            file=sys.stderr,
        )
    return video_to_relpath


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def safe_float(value: Any) -> Optional[float]:
    """Parse a numeric metadata field without raising on empty or invalid values."""

    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def estimated_clip_frame_count(
    duration: float,
    fps: float,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> int:
    """Estimate how many raw frames are available in the first training clip."""

    return int(math.floor(min(duration, clip_seconds) * fps)) + 1


def get_annotation(item: Dict[str, Any]) -> Dict[str, Any]:
    annotation = item.get("physical_annotation") or {}
    if not isinstance(annotation, dict):
        return {}
    return annotation


def category_from_label(label: Any) -> Optional[str]:
    normalized_label = normalize_text(label).lower()
    for category, labels in PHYSICS_CATEGORY_LABELS.items():
        if normalized_label in labels:
            return category
    return None


def can_open_video(video_path: Path) -> bool:
    """Return whether the video container can be opened and its first frame decoded."""

    if not video_path.exists():
        return False

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return True
        finally:
            cap.release()
    except Exception:
        pass

    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore

        reader = imageio.get_reader(str(video_path))
        try:
            reader.get_data(0)
        finally:
            reader.close()
    except Exception:
        return False
    return True


def progress_iter(iterable: Iterable[Any], total: Optional[int], desc: str) -> Iterable[Any]:
    """Wrap an iterable with tqdm when tqdm is available."""

    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def check_openable_videos(
    video_relpaths: Sequence[str],
    extract_dir: Path,
    num_workers: int,
    desc: str,
) -> Dict[str, bool]:
    """Decode the first frame of each unique video path, optionally in parallel."""

    unique_relpaths = sorted(set(video_relpaths))
    if not unique_relpaths:
        return {}

    workers = max(1, num_workers)
    if workers == 1:
        results: Dict[str, bool] = {}
        for relpath in progress_iter(unique_relpaths, len(unique_relpaths), desc):
            results[relpath] = can_open_video(extract_dir / relpath)
        return results

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_relpath = {
            pool.submit(can_open_video, extract_dir / relpath): relpath
            for relpath in unique_relpaths
        }
        completed = as_completed(future_to_relpath)
        for future in progress_iter(completed, len(future_to_relpath), desc):
            relpath = future_to_relpath[future]
            try:
                results[relpath] = future.result()
            except Exception:
                results[relpath] = False
    return results


def basic_rejection_reason(
    item: Dict[str, Any],
    video_relpath: str,
    extract_dir: Path,
    min_duration: float,
    openable_videos: Optional[Dict[str, bool]] = None,
) -> Optional[str]:
    """Return the basic filtering failure reason shared by all category CSVs."""

    duration = safe_float(item.get("duration"))
    if duration is None:
        return "duration_empty"
    if duration <= min_duration:
        return f"duration_le_{min_duration:.2f}"
    fps = safe_float(item.get("fps"))
    if fps is None:
        return "fps_invalid"
    if estimated_clip_frame_count(duration, fps) < DEFAULT_TARGET_NUM_FRAMES:
        return f"first_{DEFAULT_CLIP_SECONDS:.0f}s_frames_lt_{DEFAULT_TARGET_NUM_FRAMES}"
    if not video_relpath:
        return "video_missing"
    if openable_videos is None:
        if not can_open_video(extract_dir / video_relpath):
            return "video_open_failed"
    elif not openable_videos.get(video_relpath, False):
        return "video_open_failed"
    return None


def dynamics_rejection_reason(item: Dict[str, Any]) -> Optional[str]:
    """Return why a sample should be excluded from the strict dynamics subset."""

    annotation = get_annotation(item)
    q0 = normalize_text(annotation.get("q0")).lower()
    if q0 == NO_OBVIOUS_DYNAMICS:
        return "q0_no_obvious_dynamic_phenomenon"

    prompt = normalize_text(item.get("captions")).lower()
    n0 = normalize_text(annotation.get("n0")).lower()
    if "magnet" in prompt or "magetic" in prompt or "magnet" in n0 or "magetic" in n0:
        return "magnet_or_magetic_in_prompt_or_n0"
    if "metallic" in n0:
        return "metallic_in_n0"

    motion_score = safe_float(item.get("motion_score"))
    motion_score_v2 = safe_float(item.get("motion_score_v2"))
    if motion_score is None:
        return "motion_score_invalid"
    if motion_score_v2 is None:
        return "motion_score_v2_invalid"
    if motion_score <= 0.10:
        return "motion_score_le_0.10"
    if motion_score_v2 <= 0.01:
        return "motion_score_v2_le_0.01"
    if motion_score_v2 >= 6.50:
        return "motion_score_v2_ge_6.50"
    return None


def is_dynamic_sample(item: Dict[str, Any]) -> bool:
    """Return whether a meta item is a clean dynamics sample.

    This predicate contains the dynamics-specific semantic and numeric filters.
    File-level validity is checked separately because it needs the extracted
    video path.
    """

    label = normalize_text(item.get("label")).lower()
    return label in DYNAMIC_LABELS and dynamics_rejection_reason(item) is None


def row_from_meta(item: Dict[str, Any], video_relpath: str) -> Dict[str, Any]:
    annotation = get_annotation(item)
    row = {
        "video": video_relpath,
        "prompt": normalize_text(item.get("captions")),
        "video_name": normalize_text(item.get("video_name")),
        "label": normalize_text(item.get("label")),
        "width": item.get("width", ""),
        "height": item.get("height", ""),
        "fps": item.get("fps", ""),
        "duration": item.get("duration", ""),
        "motion_score": item.get("motion_score", ""),
        "motion_score_v2": item.get("motion_score_v2", ""),
        "visual_quality_score": item.get("visual_quality_score", ""),
        "text_bbox_num": item.get("text_bbox_num", ""),
        "text_bbox_ratio": item.get("text_bbox_ratio", ""),
        "phys_law": normalize_text(annotation.get("phys_law")),
        "q0": normalize_text(annotation.get("q0")),
        "q1": normalize_text(annotation.get("q1")),
        "q2": normalize_text(annotation.get("q2")),
        "q3": normalize_text(annotation.get("q3")),
        "q4": normalize_text(annotation.get("q4")),
        "n0": normalize_text(annotation.get("n0")),
        "n1": normalize_text(annotation.get("n1")),
        "n2": normalize_text(annotation.get("n2")),
    }
    return row


def passes_filters(
    item: Dict[str, Any],
    label_filter: Optional[set[str]],
    only_dynamics: bool,
) -> bool:
    if label_filter is not None and normalize_text(item.get("label")).lower() not in label_filter:
        return False
    if only_dynamics and not is_dynamic_sample(item):
        return False
    return True


def write_metadata(
    rows: Sequence[Dict[str, Any]],
    metadata_out: Path,
    as_jsonl: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] would write metadata: {metadata_out} ({len(rows)} rows)")
        for row in rows[:3]:
            print(json.dumps(row, ensure_ascii=False)[:1000])
        return

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    if as_jsonl:
        with metadata_out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with metadata_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote metadata: {metadata_out} ({len(rows)} rows)")


def write_category_metadata(
    rows_by_category: Dict[str, List[Dict[str, Any]]],
    metadata_dir: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        for category, rows in rows_by_category.items():
            out_path = metadata_dir / CATEGORY_METADATA_FILENAMES[category]
            print(f"[dry-run] would write metadata: {out_path} ({len(rows)} rows)")
            for row in rows[:2]:
                print(json.dumps(row, ensure_ascii=False)[:1000])
        return

    metadata_dir.mkdir(parents=True, exist_ok=True)
    for category, rows in rows_by_category.items():
        out_path = metadata_dir / CATEGORY_METADATA_FILENAMES[category]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote metadata: {out_path} ({len(rows)} rows)")


def append_openable_category_rows(
    category: str,
    candidates: Sequence[Tuple[Dict[str, Any], str]],
    rows: List[Dict[str, Any]],
    reject_counter: Counter[str],
    args: argparse.Namespace,
) -> None:
    """Decode candidate videos in parallel and append rows that pass video checks."""

    if not candidates:
        return

    if args.video_check == "off":
        selected = candidates
        if args.limit_items is not None:
            selected = selected[: args.limit_items]
            reject_counter["limit_items"] += max(0, len(candidates) - len(selected))
        for item, video_relpath in selected:
            rows.append(row_from_meta(item, video_relpath))
        return

    workers = max(1, args.video_check_workers)
    if args.limit_items is None:
        openable = check_openable_videos(
            [video_relpath for _, video_relpath in candidates],
            args.extract_dir,
            workers,
            f"check {category} videos",
        )
        for item, video_relpath in candidates:
            basic_reason = basic_rejection_reason(
                item, video_relpath, args.extract_dir, args.min_duration, openable_videos=openable
            )
            if basic_reason is not None:
                reject_counter[f"basic:{basic_reason}"] += 1
                continue
            rows.append(row_from_meta(item, video_relpath))
        return

    batch_size = max(args.limit_items * 4, workers * 8, 32)
    start = 0
    while start < len(candidates) and len(rows) < args.limit_items:
        batch = candidates[start : start + batch_size]
        start += batch_size
        openable = check_openable_videos(
            [video_relpath for _, video_relpath in batch],
            args.extract_dir,
            workers,
            f"check {category} videos",
        )
        for item, video_relpath in batch:
            basic_reason = basic_rejection_reason(
                item, video_relpath, args.extract_dir, args.min_duration, openable_videos=openable
            )
            if basic_reason is not None:
                reject_counter[f"basic:{basic_reason}"] += 1
                continue
            rows.append(row_from_meta(item, video_relpath))
            if len(rows) >= args.limit_items:
                break

    if len(rows) >= args.limit_items:
        reject_counter["limit_items"] += max(0, len(candidates) - start)


def build_category_metadata(args: argparse.Namespace) -> None:
    """Build one strict metadata CSV for each physics category."""

    data = load_meta(args.meta_json)
    video_to_relpath = scan_extracted_videos(args.extract_dir)
    rows_by_category: Dict[str, List[Dict[str, Any]]] = {
        category: [] for category in PHYSICS_CATEGORY_LABELS
    }
    candidates_by_category: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
        category: [] for category in PHYSICS_CATEGORY_LABELS
    }
    reject_counters: Dict[str, Counter[str]] = {
        category: Counter() for category in PHYSICS_CATEGORY_LABELS
    }
    skipped_non_category = 0

    for item in progress_iter(data, len(data), "scan metadata"):
        category = category_from_label(item.get("label"))
        if category is None:
            skipped_non_category += 1
            continue

        video_name = normalize_text(item.get("video_name"))
        video_relpath = video_to_relpath.get(video_name, "")
        duration = safe_float(item.get("duration"))
        if duration is None:
            reject_counters[category]["basic:duration_empty"] += 1
            continue
        if duration <= args.min_duration:
            reject_counters[category][f"basic:duration_le_{args.min_duration:.2f}"] += 1
            continue
        fps = safe_float(item.get("fps"))
        if fps is None:
            reject_counters[category]["basic:fps_invalid"] += 1
            continue
        if estimated_clip_frame_count(duration, fps) < DEFAULT_TARGET_NUM_FRAMES:
            reject_counters[category][
                f"basic:first_{DEFAULT_CLIP_SECONDS:.0f}s_frames_lt_{DEFAULT_TARGET_NUM_FRAMES}"
            ] += 1
            continue
        if not video_relpath:
            reject_counters[category]["basic:video_missing"] += 1
            continue

        if category == "dynamics":
            dynamics_reason = dynamics_rejection_reason(item)
            if dynamics_reason is not None:
                reject_counters[category][f"dynamics:{dynamics_reason}"] += 1
                continue

        candidates_by_category[category].append((item, video_relpath))

    for category, candidates in candidates_by_category.items():
        append_openable_category_rows(
            category, candidates, rows_by_category[category], reject_counters[category], args
        )

    write_category_metadata(rows_by_category, args.metadata_out, args.dry_run)
    print(
        "category metadata summary: "
        f"meta_items={len(data)} extracted_names={len(video_to_relpath)} "
        f"skipped_non_category={skipped_non_category}"
    )
    for category, rows in rows_by_category.items():
        label_counter = Counter(row["label"] for row in rows)
        print(f"{category}: rows={len(rows)}")
        for label, count in label_counter.most_common():
            print(f"  {count:8d}  {label}")
        if reject_counters[category]:
            print(f"{category}_rejections:")
            for reason, count in reject_counters[category].most_common():
                print(f"  {count:8d}  {reason}")


def build_metadata(args: argparse.Namespace) -> None:
    if args.metadata_mode == "categories":
        if args.jsonl:
            raise ValueError("--jsonl is only supported with --metadata-mode all")
        build_category_metadata(args)
        return

    data = load_meta(args.meta_json)
    video_to_relpath = scan_extracted_videos(args.extract_dir)
    label_filter = None
    if args.label_filter:
        label_filter = {x.strip().lower() for x in args.label_filter.split(",") if x.strip()}

    rows: List[Dict[str, Any]] = []
    skipped_missing = 0
    skipped_filter = 0
    skipped_open_failed = 0
    for item in data:
        if not passes_filters(item, label_filter, args.only_dynamics):
            skipped_filter += 1
            continue
        video_name = normalize_text(item.get("video_name"))
        video_relpath = video_to_relpath.get(video_name, "")
        if not video_relpath and not args.allow_missing:
            skipped_missing += 1
            continue
        if args.only_dynamics:
            if args.video_check == "on":
                basic_reason = basic_rejection_reason(item, video_relpath, args.extract_dir, args.min_duration)
            else:
                basic_reason = basic_rejection_reason(
                    item, video_relpath, args.extract_dir, args.min_duration, openable_videos={video_relpath: True}
                )
            if basic_reason is not None:
                if basic_reason == "video_open_failed":
                    skipped_open_failed += 1
                else:
                    skipped_filter += 1
                continue
        rows.append(row_from_meta(item, video_relpath or video_name))
        if args.limit_items is not None and len(rows) >= args.limit_items:
            break

    write_metadata(rows, args.metadata_out, args.jsonl, args.dry_run)
    print(
        "metadata summary: "
        f"meta_items={len(data)} extracted_names={len(video_to_relpath)} "
        f"rows={len(rows)} skipped_filter={skipped_filter} "
        f"skipped_missing={skipped_missing} skipped_open_failed={skipped_open_failed}"
    )


def default_sample_out(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_sample.jsonl")


def sanitize_filename_field(value: Any, max_chars: int = 120) -> str:
    """Convert a metadata value into a compact file-name-safe token."""

    text = normalize_text(value)
    if not text:
        return "NA"
    chars = []
    for char in text:
        if char.isspace() or char in "/\\:*?\"<>|,;":
            chars.append("_")
        elif char.isalnum() or char in ".-_":
            chars.append(char)
        else:
            chars.append("_")
    token = "".join(chars).strip("._-")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        token = "NA"
    if len(token) > max_chars:
        token = token[:max_chars].rstrip("._-") or "NA"
    return token


def sampled_video_name(idx: int, row: Dict[str, str]) -> str:
    """Build the copied sample video name from label and motion metadata."""

    label = sanitize_filename_field(row.get("label"), max_chars=60)
    motion_score1 = sanitize_filename_field(row.get("motion_score"), max_chars=24)
    motion_score2 = sanitize_filename_field(row.get("motion_score_v2"), max_chars=24)
    duration = sanitize_filename_field(row.get("duration"), max_chars=24)
    n0 = sanitize_filename_field(row.get("n0"), max_chars=120)
    return (
        f"{idx:05d}_"
        f"type_{label}-"
        f"motion_score1_{motion_score1}-"
        f"motion_score2_{motion_score2}-"
        f"duration_{duration}-"
        f"n0_{n0}.mp4"
    )


def sample_csv_rows(
    csv_path: Path,
    sample_n: int,
    random_sample: bool,
    seed: int,
) -> Tuple[List[Dict[str, str]], int]:
    """Read at most ``sample_n`` rows from a CSV, optionally by reservoir sampling.

    The function keeps CSV values as strings because this sample file is mainly
    for inspection. Preserving raw CSV values avoids surprising type coercions
    when users compare a JSONL sample with the original metadata CSV.
    """

    if sample_n <= 0:
        raise ValueError(f"--sample-n must be positive, got {sample_n}")

    rows: List[Dict[str, str]] = []
    total_rows = 0
    rng = random.Random(seed)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for row in reader:
            total_rows += 1
            if random_sample:
                if len(rows) < sample_n:
                    rows.append(row)
                else:
                    replace_id = rng.randint(0, total_rows - 1)
                    if replace_id < sample_n:
                        rows[replace_id] = row
            else:
                if len(rows) < sample_n:
                    rows.append(row)
                else:
                    break
    return rows, total_rows


def resolve_sample_video_path(
    csv_path: Path,
    video_rel: str,
    sample_base_dir: Optional[Path],
) -> Optional[Path]:
    """Resolve a sampled row's relative video path against likely dataset roots."""

    if not video_rel:
        return None
    video_path = Path(video_rel)
    if video_path.is_absolute():
        return video_path if video_path.exists() else None

    candidate_roots: List[Path] = []
    if sample_base_dir is not None:
        candidate_roots.append(sample_base_dir)
    candidate_roots.append(csv_path.parent)
    candidate_roots.append(csv_path.parent.parent)

    seen_roots = set()
    for root in candidate_roots:
        root_key = str(root.resolve()) if root.exists() else str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        candidate = root / video_rel
        if candidate.exists():
            return candidate
        basename_candidate = root / video_path.name
        if basename_candidate.exists():
            return basename_candidate
    return None


def export_csv_sample_to_jsonl(
    csv_path: Path,
    jsonl_path: Path,
    sample_n: int,
    random_sample: bool,
    seed: int,
    copy_dir: Optional[Path],
    sample_base_dir: Optional[Path],
    copy_key: str,
    dry_run: bool,
) -> None:
    """Export ``sample_n`` metadata CSV rows to a JSONL file for easy inspection."""

    rows, total_rows = sample_csv_rows(csv_path, sample_n, random_sample, seed)
    if dry_run:
        print(
            f"[dry-run] would write sample jsonl: {jsonl_path} "
            f"({len(rows)} rows from {total_rows} scanned rows)"
        )
        for row in rows[:3]:
            print(json.dumps(row, ensure_ascii=False)[:1000])
        return

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_dir is not None and not dry_run:
        copy_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if copy_dir is None:
                continue
            video_rel = row.get(copy_key, "")
            if not video_rel:
                continue
            src = resolve_sample_video_path(csv_path, video_rel, sample_base_dir)
            if src is None:
                print(
                    f"warning: sampled video not found: {video_rel} "
                    f"(set --sample-base-dir to the dataset root if needed)",
                    file=sys.stderr,
                )
                continue
            dst_name = sampled_video_name(idx, row)
            dst = copy_dir / dst_name
            if dry_run:
                continue
            shutil.copy2(src, dst)
    mode = "random" if random_sample else "first"
    print(
        f"wrote sample jsonl: {jsonl_path} "
        f"({len(rows)} {mode} rows from {total_rows} scanned rows)"
    )
    if copy_dir is not None:
        print(f"copied sampled videos to: {copy_dir}")


def read_metadata_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Read a metadata CSV and validate that it has a header."""

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return [dict(row) for row in reader]


def write_metadata_csv(rows: Sequence[Dict[str, str]], csv_path: Path) -> None:
    """Write rows using the standard metadata field order."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def merge_category_metadata(
    metadata_dir: Path,
    merge_out: Path,
    reflection_sample_n: int,
    seed: int,
    dry_run: bool,
) -> None:
    """Merge category metadata and downsample the reflection subtype."""

    if reflection_sample_n <= 0:
        raise ValueError(f"--reflection-sample-n must be positive, got {reflection_sample_n}")

    input_paths = [
        metadata_dir / CATEGORY_METADATA_FILENAMES["dynamics"],
        metadata_dir / CATEGORY_METADATA_FILENAMES["thermodynamics"],
        metadata_dir / CATEGORY_METADATA_FILENAMES["optics"],
    ]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing category metadata CSV: {path}")

    rows: List[Dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    for path in input_paths:
        category_rows = read_metadata_csv(path)
        source_counts[path.name] = len(category_rows)
        rows.extend(category_rows)

    reflection_rows = [row for row in rows if normalize_text(row.get("label")).lower() == "reflection"]
    other_rows = [row for row in rows if normalize_text(row.get("label")).lower() != "reflection"]
    if len(reflection_rows) > reflection_sample_n:
        reflection_rows = random.Random(seed).sample(reflection_rows, reflection_sample_n)

    merged_rows = other_rows + reflection_rows
    random.Random(seed).shuffle(merged_rows)

    label_counter = Counter(normalize_text(row.get("label")) for row in merged_rows)
    if dry_run:
        print(f"[dry-run] would write merged metadata: {merge_out} ({len(merged_rows)} rows)")
    else:
        write_metadata_csv(merged_rows, merge_out)
        print(f"wrote merged metadata: {merge_out} ({len(merged_rows)} rows)")

    print("merge source rows:")
    for name, count in source_counts.items():
        print(f"  {count:8d}  {name}")
    print(f"reflection_source_rows: {len([row for row in rows if normalize_text(row.get('label')).lower() == 'reflection'])}")
    print(f"reflection_kept_rows: {len(reflection_rows)}")
    print("merged label distribution:")
    for label, count in label_counter.most_common():
        print(f"  {count:8d}  {label}")


def summarize_csv(csv_path: Path) -> None:
    """Print a compact label distribution for a metadata CSV."""

    label_counter: Counter[str] = Counter()
    q0_counter: Counter[str] = Counter()
    total_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for row in reader:
            total_rows += 1
            label_counter[normalize_text(row.get("label"))] += 1
            if "q0" in row:
                q0_counter[normalize_text(row.get("q0"))] += 1

    print(f"csv: {csv_path}")
    print(f"rows: {total_rows}")
    print(f"num_labels: {len(label_counter)}")
    for label, count in label_counter.most_common():
        print(f"{count:8d}  {label}")
    if q0_counter:
        print(f"num_q0: {len(q0_counter)}")
        print("q0_top20:")
        for q0, count in q0_counter.most_common(20):
            print(f"{count:8d}  {q0}")


def main() -> None:
    args = parse_args()
    run_inspect = args.inspect_meta or args.all
    run_extract = args.extract or args.all
    run_metadata = args.build_metadata or args.all
    run_sample = args.sample_csv is not None
    run_summary = args.summarize_csv is not None
    run_merge = args.merge_category_metadata is not None
    if not (run_inspect or run_extract or run_metadata or run_sample or run_summary or run_merge):
        run_inspect = True
        run_metadata = True

    if run_inspect:
        inspect_meta(args.meta_json)
    if run_extract:
        extract_zips(args)
    if run_metadata:
        build_metadata(args)
    if run_sample:
        sample_out = args.sample_out or default_sample_out(args.sample_csv)
        export_csv_sample_to_jsonl(
            args.sample_csv,
            sample_out,
            args.sample_n,
            args.sample_random,
            args.sample_seed,
            args.sample_copy_dir,
            args.sample_base_dir,
            args.sample_copy_key,
            args.dry_run,
        )
    if run_summary:
        summarize_csv(args.summarize_csv)
    if run_merge:
        if args.merge_out is None:
            raise ValueError("--merge-out is required with --merge-category-metadata")
        merge_category_metadata(
            args.merge_category_metadata,
            args.merge_out,
            args.reflection_sample_n,
            args.merge_seed,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
