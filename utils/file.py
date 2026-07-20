import cv2
import csv
import json
import jsonlines

import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
root_dir = Path(os.getenv("ROOT_DIR", PROJECT_ROOT.as_posix()))
work_dir = Path(os.getenv("WORK_DIR", PROJECT_ROOT.as_posix()))


def read_video(video_path):
    """Read video frames from a video file.
    Args:
        video_path (str): Path to the video file.

    Returns:
        frames (list): List of video frames.
        sample_rate (int): Sample rate of the video.
        height (int): Height of the video frames.
        width (int): Width of the video frames.
    """
    # install cv2: pip install opencv-python
    video = cv2.VideoCapture(video_path)
    sample_rate = int(video.get(cv2.CAP_PROP_FPS))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()

    return frames, sample_rate, height, width

def csv_to_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8", newline="") as f_in, open(output_path, "wb") as f_out:
        reader = csv.DictReader(f_in)
        fields = reader.fieldnames
        print(fields)
        with jsonlines.open(output_path, "w") as writer:
            for line in reader:
                writer.write(line)

def json_to_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(list(data[0].keys()))
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(data)

def format_jsonl_line_vid_gen(input_path, output_path, original_condition_key):
    data = []
    with jsonlines.open(input_path, "r") as f:
        for line in f:
            data.append(line)
    with jsonlines.open(output_path, "w") as f:
        for i, line in enumerate(data):
            new_line = {"id": i}
            new_line["prompt"] = line[original_condition_key]
            del line[original_condition_key]
            new_line.update(line)
            f.write(new_line)

def match_jsonl(full_path, subset_path, output_path):
    full_map = {}
    with jsonlines.open(full_path, "r") as reader:
        for item in reader:
            full_map[item["caption"]] = item
    match_captions = set()
    with jsonlines.open(subset_path, "r") as reader:
        for item in reader:
            match_captions.add(item["caption"])
    with jsonlines.open(output_path, "w") as writer:
        for cap in match_captions:
            if cap in full_map:
                writer.write(full_map[cap])

def get_solid_dynamics_cases_from_videophy(input_path, output_path):
    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for item in reader:
            if item["states_of_matter"] == "solid_solid":
                writer.write(item)

def get_cases_from_videophy2_by_category(input_path, output_path, category):
    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for item in reader:
            if item["category"] == category:
                writer.write(item)

def get_liquid_dynamics_cases_from_phygenbench(input_path, output_path):
    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for item in reader:
            if item["states_of_matter"] == "liquid_liquid":
                writer.write(item)

def find_categories(input_path):
    ret = set()
    with jsonlines.open(input_path, "r") as f:
        for line in f:
            ret.add(line["category"])
    print(ret)
    return ret


if __name__ == "__main__":
    pass

    # video_path = ""
    # frames, sample_rate, height, width = read_video(video_path)
    # print(f"Number of frames: {len(frames)}\nSample rate: {sample_rate}\nHeight: {height}\nWidth: {width}")

    # csv_path = str(root_dir / "wan_eval/datasets/videophy/videophy_test_public.csv")
    # jsonl_path = str(root_dir / "wan_eval/datasets/videophy/videophy_test_public.jsonl")
    # csv_to_jsonl(csv_path, jsonl_path)

    # json_path = str(root_dir / "wan_eval/datasets/phygenbench/prompts.json")
    # jsonl_path = str(root_dir / "wan_eval/datasets/phygenbench/prompts.jsonl")
    # json_to_jsonl(json_path, jsonl_path)

    # full_path = str(root_dir / "wan_eval/datasets/videophy2/videophy2_test.jsonl")
    # test_path = str(root_dir / "wan_eval/datasets/videophy2/prompt-upsampled-test.jsonl")
    # output_path = str(root_dir / "wan_eval/datasets/videophy2/prompt-upsampled-test-w_meta.jsonl")
    # match_jsonl(full_path, test_path, output_path)

    # old_jsonl_path = str(root_dir / "wan_eval/datasets/videophy/videophy_test_public_344.jsonl")
    # new_jsonl_path = str(root_dir / "wan_eval/datasets/videophy/prompts.jsonl")
    # condition_key = "caption"
    # format_jsonl_line_vid_gen(old_jsonl_path, new_jsonl_path, condition_key)
    
    jsonl_path = str(root_dir / "wan_eval/datasets/videophy2/prompts.jsonl")
    category = "Sports and Physical Activities"
    output_path = str(root_dir / f"wan_eval/datasets/videophy2/prompts-{category}.jsonl")
    # find_categories(jsonl_path)
    get_cases_from_videophy2_by_category(jsonl_path, output_path, category)
