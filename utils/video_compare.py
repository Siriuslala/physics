"""Serve a lightweight multi-directory video comparison panel.

The panel shows one video from each input directory in a horizontal row, uses
the shared filename as the title, and lets the user move through videos in
numeric filename order with previous/next buttons or arrow keys.
"""

import argparse
import html
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote, urlparse


VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".mkv",
}


def _natural_sort_key(path: object) -> List[object]:
    """Return a sort key that treats digit runs in paths as integers."""
    text = path.as_posix() if isinstance(path, Path) else str(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _parse_last_integer_id(text: str) -> str:
    """Parse the last integer in the filename stem as a normalized string id."""
    filename_stem = Path(text).stem
    matches = re.findall(r"\d+", filename_stem)
    if not matches:
        return ""
    return str(int(matches[-1]))


def _find_videos(
    directory: Path,
    suffixes: Sequence[str],
    recursive: bool,
) -> List[Path]:
    """Find video files in `directory`, sorted by natural path order."""
    valid_suffixes = {suffix.lower() for suffix in suffixes}
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        [
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in valid_suffixes
        ],
        key=lambda path: _natural_sort_key(path.relative_to(directory)),
    )


def _pair_key(path: Path, root: Path, recursive: bool) -> str:
    """Return the matching key used to pair videos across the two directories."""
    if recursive:
        return path.relative_to(root).as_posix()
    return path.name


def _build_comparisons(
    directories: Sequence[Path],
    pair_by: str,
    suffixes: Sequence[str],
    recursive: bool,
) -> List[Tuple[str, List[Path]]]:
    """Build `(title, video_paths)` entries for the comparison panel.

    Args:
        directories: Directories whose videos are shown as comparison columns.
        pair_by: Pairing mode. `name` requires identical filenames in both
            directories. `order` pairs sorted file lists by index.
        suffixes: Video suffixes to include.
        recursive: Whether to scan nested subdirectories.

    Returns:
        A sorted list of per-title video path lists.
    """
    if len(directories) < 2:
        raise ValueError("At least two video directories are required.")

    video_lists = [_find_videos(directory, suffixes, recursive) for directory in directories]
    for directory, videos in zip(directories, video_lists):
        if not videos:
            raise ValueError(f"No video files found in directory: {directory}")

    counts = [len(videos) for videos in video_lists]
    if len(set(counts)) != 1:
        count_text = ", ".join(
            f"{directory}: {count}"
            for directory, count in zip(directories, counts)
        )
        raise ValueError(
            "All directories must contain the same number of videos. "
            + count_text
        )

    if pair_by == "order":
        return [
            (
                _pair_key(paths[0], directories[0], recursive),
                list(paths),
            )
            for paths in zip(*video_lists)
        ]

    key_lists = [
        [_pair_key(path, directory, recursive) for path in videos]
        for directory, videos in zip(directories, video_lists)
    ]
    reference_keys = set(key_lists[0])
    details = []
    for idx, keys in enumerate(key_lists[1:], start=1):
        key_set = set(keys)
        missing = sorted(reference_keys - key_set, key=_natural_sort_key)
        extra = sorted(key_set - reference_keys, key=_natural_sort_key)
        if missing:
            details.append(f"dir {idx + 1} missing: {missing[:8]}")
        if extra:
            details.append(f"dir {idx + 1} extra: {extra[:8]}")
    if details:
        raise ValueError(
            "Filename pairing requires every directory to contain the same video keys. "
            + "; ".join(details)
            + ". Use `--pair-by order` to pair sorted files by index."
        )

    path_maps = [
        {
            _pair_key(path, directory, recursive): path
            for path in videos
        }
        for directory, videos in zip(directories, video_lists)
    ]
    ordered_keys = sorted(key_lists[0], key=_natural_sort_key)
    return [(key, [path_map[key] for path_map in path_maps]) for key in ordered_keys]


def _summarize_pair_ids(comparisons: Sequence[Tuple[str, Sequence[Path]]]) -> str:
    """Summarize parsed numeric ids for startup diagnostics."""
    ids = [_parse_last_integer_id(title) for title, _ in comparisons]
    numeric_ids = [int(video_id) for video_id in ids if video_id != ""]
    if not numeric_ids:
        return "no numeric filename ids parsed"

    first_ids = ", ".join(str(video_id) for video_id in numeric_ids[:8])
    min_id = min(numeric_ids)
    max_id = max(numeric_ids)
    return f"filename id range {min_id}-{max_id}; first ids: {first_ids}"


def _parse_range_header(
    range_header: Optional[str],
    file_size: int,
) -> Optional[Tuple[int, int]]:
    """Parse a single HTTP byte range header."""
    if not range_header or not range_header.startswith("bytes="):
        return None

    range_spec = range_header.removeprefix("bytes=").strip()
    if "," in range_spec:
        return None

    start_text, separator, end_text = range_spec.partition("-")
    if separator != "-":
        return None

    try:
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None

    if start < 0 or start >= file_size or end < start:
        return None
    return start, min(end, file_size - 1)


def _make_handler(
    comparisons: Sequence[Tuple[str, Sequence[Path]]],
    labels: Sequence[str],
):
    """Create a request handler bound to one immutable comparison session."""
    indexed_paths: Dict[Tuple[int, int], Path] = {}
    videos = []
    for idx, (title, paths) in enumerate(comparisons):
        sources = []
        for column, path in enumerate(paths):
            indexed_paths[(column, idx)] = path
            sources.append(f"/video/{column}/{idx}/{quote(path.name)}")
        videos.append(
            {
                "title": title,
                "id": _parse_last_integer_id(title),
                "sources": sources,
            }
        )

    class VideoCompareHandler(BaseHTTPRequestHandler):
        """Serve the comparison HTML page and local video files."""

        server_version = "VideoCompareHTTP/1.0"

        def log_message(self, fmt: str, *args) -> None:
            if getattr(self.server, "quiet", False):
                return
            super().log_message(fmt, *args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._serve_index()
                return
            if parsed.path.startswith("/video/"):
                self._serve_video(parsed.path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def _serve_index(self) -> None:
            body = _render_html(
                videos=videos,
                labels=labels,
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_video(self, path: str) -> None:
            parts = [unquote(part) for part in path.split("/") if part]
            if len(parts) < 4 or parts[0] != "video":
                self.send_error(HTTPStatus.NOT_FOUND, "Invalid video URL")
                return

            try:
                column = int(parts[1])
                idx = int(parts[2])
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "Invalid video index")
                return

            video_path = indexed_paths.get((column, idx))
            if video_path is None or not video_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
                return

            file_size = video_path.stat().st_size
            byte_range = _parse_range_header(self.headers.get("Range"), file_size)
            content_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
            start, end = byte_range if byte_range is not None else (0, file_size - 1)
            content_length = end - start + 1

            self.send_response(
                HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK
            )
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(content_length))
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with video_path.open("rb") as video_file:
                video_file.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = video_file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return VideoCompareHandler


def _render_html(videos: Sequence[dict], labels: Sequence[str]) -> str:
    """Render the comparison panel HTML."""
    videos_json = json.dumps(videos, ensure_ascii=False)
    labels_json = json.dumps(list(labels), ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Compare</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111317;
      --panel: #1b1f27;
      --border: #333946;
      --text: #f3f5f7;
      --muted: #a8b0bd;
      --accent: #4da3ff;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 16px;
      padding: 20px;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 650;
      text-align: center;
      word-break: break-word;
    }}
    .videos {{
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(320px, 1fr);
      gap: 16px;
      min-height: 0;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .video-panel {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      min-width: 0;
      min-height: 0;
    }}
    .label {{
      color: var(--muted);
      font-size: 14px;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    video {{
      width: 100%;
      height: 100%;
      min-height: 260px;
      max-height: calc(100vh - 160px);
      background: #050608;
      border: 1px solid var(--border);
      border-radius: 8px;
      object-fit: contain;
    }}
    footer {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      min-height: 44px;
    }}
    button {{
      height: 38px;
      min-width: 84px;
      padding: 0 16px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font-size: 15px;
      cursor: pointer;
    }}
    button:hover:not(:disabled) {{
      border-color: var(--accent);
    }}
    button:disabled {{
      color: #5d6572;
      cursor: default;
    }}
    .counter {{
      min-width: 96px;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
    }}
    .jump {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    input {{
      width: 112px;
      height: 38px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #0d1015;
      color: var(--text);
      font-size: 15px;
      padding: 0 10px;
    }}
    input:focus {{
      border-color: var(--accent);
      outline: none;
    }}
    .status {{
      width: 240px;
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    @media (max-width: 820px) {{
      .app {{
        padding: 12px;
      }}
      .videos {{
        grid-auto-flow: row;
        grid-auto-columns: auto;
        grid-template-columns: 1fr;
        overflow-x: visible;
      }}
      video {{
        height: auto;
        max-height: none;
      }}
      footer {{
        flex-wrap: wrap;
      }}
      .status {{
        width: 220px;
      }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <header>
      <h1 id="title"></h1>
    </header>
    <section id="videos" class="videos" aria-label="video comparison"></section>
    <footer>
      <button id="prevButton" type="button">pre</button>
      <div id="counter" class="counter"></div>
      <button id="nextButton" type="button">next</button>
      <div class="jump">
        <input id="jumpInput" type="number" min="0" step="1" placeholder="id">
        <button id="jumpButton" type="button">jump</button>
        <div id="jumpStatus" class="status" aria-live="polite"></div>
      </div>
    </footer>
  </main>
  <script>
    const videos = {videos_json};
    const labels = {labels_json};
    let index = 0;
    const idToIndex = new Map();

    function normalizeNumericId(value) {{
      const normalized = value.trim().replace(/^0+(?=\\d)/, "");
      return normalized;
    }}

    for (let i = 0; i < videos.length; i += 1) {{
      const videoId = normalizeNumericId(videos[i].id || "");
      if (videoId !== "" && !idToIndex.has(videoId)) {{
        idToIndex.set(videoId, i);
      }}
    }}

    const title = document.getElementById("title");
    const counter = document.getElementById("counter");
    const videoGrid = document.getElementById("videos");
    const prevButton = document.getElementById("prevButton");
    const nextButton = document.getElementById("nextButton");
    const jumpInput = document.getElementById("jumpInput");
    const jumpButton = document.getElementById("jumpButton");
    const jumpStatus = document.getElementById("jumpStatus");
    const videoElements = [];

    for (let column = 0; column < labels.length; column += 1) {{
      const panel = document.createElement("div");
      panel.className = "video-panel";

      const label = document.createElement("div");
      label.className = "label";
      label.textContent = labels[column];

      const video = document.createElement("video");
      video.controls = true;
      video.muted = true;
      video.playsInline = true;
      video.preload = "metadata";

      panel.appendChild(label);
      panel.appendChild(video);
      videoGrid.appendChild(panel);
      videoElements.push(video);
    }}

    function loadVideo(nextIndex) {{
      index = Math.max(0, Math.min(videos.length - 1, nextIndex));
      const item = videos[index];
      title.textContent = item.title;
      counter.textContent = `${{index + 1}} / ${{videos.length}}`;
      prevButton.disabled = index === 0;
      nextButton.disabled = index === videos.length - 1;
      for (let column = 0; column < videoElements.length; column += 1) {{
        videoElements[column].pause();
        videoElements[column].removeAttribute("src");
        videoElements[column].load();
        videoElements[column].src = item.sources[column];
        videoElements[column].load();
      }}
      jumpInput.value = item.id || "";
      jumpStatus.textContent = "";
    }}

    function jumpToInputId() {{
      const target = normalizeNumericId(jumpInput.value);
      if (target === "") {{
        jumpStatus.textContent = "empty id";
        return;
      }}
      if (idToIndex.has(target)) {{
        loadVideo(idToIndex.get(target));
        return;
      }}

      const ordinalIndex = Number.parseInt(target, 10) - 1;
      if (Number.isInteger(ordinalIndex) && ordinalIndex >= 0 && ordinalIndex < videos.length) {{
        loadVideo(ordinalIndex);
        return;
      }}

      const examples = Array.from(idToIndex.keys()).slice(0, 6).join(", ");
      const idHint = examples ? `ids: ${{examples}}` : "no numeric ids";
      jumpStatus.textContent = `not found, loaded ${{videos.length}}; ${{idHint}}`;
    }}

    prevButton.addEventListener("click", () => loadVideo(index - 1));
    nextButton.addEventListener("click", () => loadVideo(index + 1));
    jumpButton.addEventListener("click", jumpToInputId);
    jumpInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        jumpToInputId();
      }}
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") {{
        loadVideo(index - 1);
      }} else if (event.key === "ArrowRight") {{
        loadVideo(index + 1);
      }}
    }});

    loadVideo(0);
  </script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch a local multi-column video comparison panel.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        required=True,
        help="Video directories to compare. Pass at least two paths.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Column labels. Must have the same length as --paths.",
    )
    parser.add_argument(
        "--pair-by",
        choices=["name", "order"],
        default="name",
        help="Pair videos by identical filename or by sorted order. Default: name.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host. Default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port. Use 0 for a free port.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan videos under each input directory.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=None,
        help="Video suffix to include, such as .mp4. Can be passed multiple times.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-request HTTP logs.")
    return parser.parse_args()


def main() -> None:
    """Validate inputs and start the local comparison server."""
    args = _parse_args()
    directories = [path.expanduser().resolve() for path in args.paths]
    if len(directories) < 2:
        raise ValueError("`--paths` must contain at least two directories.")
    if len(args.labels) != len(directories):
        raise ValueError(
            "`--labels` must have the same length as `--paths`: "
            f"got {len(args.labels)} labels for {len(directories)} paths."
        )
    for directory in directories:
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {directory}")

    suffixes = args.suffix if args.suffix is not None else sorted(VIDEO_SUFFIXES)
    suffixes = [suffix if suffix.startswith(".") else f".{suffix}" for suffix in suffixes]
    comparisons = _build_comparisons(directories, args.pair_by, suffixes, args.recursive)
    handler = _make_handler(
        comparisons=comparisons,
        labels=args.labels,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.quiet = args.quiet
    host, port = server.server_address[:2]
    url_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host

    print(f"Loaded {len(comparisons)} video groups across {len(directories)} directories.")
    print(_summarize_pair_ids(comparisons))
    print(f"First group: {comparisons[0][0]}")
    print(f"Last group: {comparisons[-1][0]}")
    print(f"Open http://{url_host}:{port} in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping video comparison server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
