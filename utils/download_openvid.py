import argparse
import os
import shlex
import subprocess
import time
from typing import Iterable


RETRY_SLEEP_SECONDS = 10
RETRYABLE_HTTP_STATUS = {404, 408, 429, 500, 502, 503, 504}


def append_error_log(error_log_path: str, message: str) -> None:
    """Append a timestamped message to the download log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    print(line, end="")
    with open(error_log_path, "a", encoding="utf-8") as error_log_file:
        error_log_file.write(line)


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    """Run a command and capture stdout/stderr for retry diagnostics."""
    return subprocess.run(command, check=False, text=True, capture_output=True)


def has_non_empty_file(file_path: str) -> bool:
    """Return True when ``file_path`` exists and has non-zero size."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def is_retryable_failure(stderr: str) -> bool:
    """Heuristically decide whether wget stderr indicates a transient failure."""
    normalized = stderr.lower()
    if "no data received" in normalized:
        return True
    if "timed out" in normalized:
        return True
    if "connection reset" in normalized:
        return True
    if "temporary failure" in normalized:
        return True
    for status_code in RETRYABLE_HTTP_STATUS:
        if f"error {status_code}" in normalized or f" {status_code} " in normalized:
            return True
    return False


def download_with_resume(url: str, file_path: str, error_log_path: str) -> None:
    """Download a file with resume support and unbounded retries.

    The implementation uses ``wget -c`` so a partially downloaded file is
    preserved and continued on the next attempt. A file is considered complete
    only when wget exits successfully.
    """
    attempt = 0
    while True:
        attempt += 1
        if os.path.exists(file_path):
            print(
                f"Resuming download for {file_path} "
                f"(current size: {os.path.getsize(file_path)} bytes)."
            )
        command = [
            "wget",
            "-c",
            "--tries=0",
            "--waitretry=10",
            "--read-timeout=60",
            "--timeout=60",
            "--retry-connrefused",
            "-O",
            file_path,
            url,
        ]
        result = run_command(command)
        if result.returncode == 0:
            print(f"file {url} saved to {file_path}")
            return

        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        append_error_log(
            error_log_path,
            (
                f"Attempt {attempt} failed for {url} with return code {result.returncode}. "
                f"stdout={stdout!r}, stderr={stderr!r}"
            ),
        )
        if not is_retryable_failure(stderr):
            append_error_log(
                error_log_path,
                f"Failure for {url} was not recognized as transient, but retrying anyway.",
            )
        time.sleep(RETRY_SLEEP_SECONDS)


def unzip_archive(file_path: str, video_folder: str) -> None:
    """Extract an archive into ``video_folder`` after validating the zip file."""
    extract_marker_path = file_path + ".extracted"
    if os.path.exists(extract_marker_path):
        print(f"Archive {file_path} already extracted; skipping unzip.")
        return
    if not has_non_empty_file(file_path):
        raise RuntimeError(f"Archive {file_path} does not exist or is empty.")
    print(f"Extracting archive {file_path} ...")
    subprocess.run(["unzip", "-o", "-j", file_path, "-d", video_folder], check=True)
    with open(extract_marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write("ok\n")


def concatenate_parts(part_paths: Iterable[str], output_path: str) -> None:
    """Concatenate multipart zip shards into a single zip archive."""
    quoted_parts = " ".join(shlex.quote(path) for path in part_paths)
    quoted_output = shlex.quote(output_path)
    subprocess.run(
        ["/bin/bash", "-lc", f"cat {quoted_parts} > {quoted_output}"],
        check=True,
    )


def download_zip_or_parts(part_index: int, zip_folder: str, error_log_path: str) -> str:
    """Download one OpenVid archive, falling back to multipart shards when needed.

    Returns the local path of the completed zip archive.
    """
    url = f"https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part_index}.zip"
    file_path = os.path.join(zip_folder, f"OpenVid_part{part_index}.zip")
    done_marker_path = file_path + ".downloaded"

    if os.path.exists(done_marker_path) and has_non_empty_file(file_path):
        print(f"file {file_path} already downloaded and verified.")
        return file_path

    if has_non_empty_file(file_path) and not os.path.exists(done_marker_path):
        append_error_log(
            error_log_path,
            (
                f"Found unfinished archive at {file_path} "
                f"({os.path.getsize(file_path)} bytes); resuming download."
            ),
        )
    elif os.path.exists(file_path) and os.path.getsize(file_path) == 0:
        append_error_log(
            error_log_path,
            f"Found empty archive placeholder at {file_path}; restarting download.",
        )

    try:
        print(f"Downloading archive part {part_index} from {url} ...")
        download_with_resume(url, file_path, error_log_path)
    except Exception as error:
        append_error_log(
            error_log_path,
            f"Unexpected error while downloading {url}: {error!r}",
        )

    if not has_non_empty_file(file_path):
        append_error_log(
            error_log_path,
            f"Primary archive for part {part_index} is empty after download; trying multipart shards.",
        )
        part_urls = [
            f"https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part_index}_partaa",
            f"https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part_index}_partab",
        ]
        part_paths = []
        for part_url in part_urls:
            part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
            print(f"Downloading multipart shard {part_url} ...")
            download_with_resume(part_url, part_file_path, error_log_path)
            part_paths.append(part_file_path)
        concatenate_parts(part_paths, file_path)

    if not has_non_empty_file(file_path):
        raise RuntimeError(f"Failed to produce a non-empty zip archive for part {part_index}.")

    with open(done_marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write("ok\n")
    return file_path


def download_metadata_files(data_folder: str, error_log_path: str) -> None:
    """Download OpenVid CSV metadata with the same resume-and-retry policy."""
    data_urls = [
        "https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
        "https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv",
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        download_with_resume(data_url, data_path, error_log_path)


def download_files(output_directory):
    """Download, validate, and extract OpenVid archives with resumable retries.

    Args:
        output_directory: Root directory containing ``download/``, ``video/``,
            and ``data/train/`` subdirectories.
    """
    zip_folder = os.path.join(output_directory, "download")
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    error_log_path = os.path.join(zip_folder, "download_log.txt")

    for i in range(0, 186):
        file_path = download_zip_or_parts(
            part_index=i,
            zip_folder=zip_folder,
            error_log_path=error_log_path,
        )
        unzip_archive(file_path, video_folder)

    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    download_metadata_files(data_folder, error_log_path)

    # delete zip files
    # delete_command = "rm -rf " + zip_folder
    # os.system(delete_command)


if __name__ == '__main__':

    # python download_openvid.py --output_directory /archive/public/openvid
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--output_directory', type=str, help='Path to the dataset directory', default="/path/to/dataset")
    args = parser.parse_args()
    download_files(args.output_directory)
