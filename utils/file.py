import cv2
import csv
import json
import jsonlines


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


if __name__ == "__main__":
    pass

    # video_path = "/home/liyueyan/Interpretability/physics/ti2v-5B_1280*704_A man running in a desert.mp4"
    # frames, sample_rate, height, width = read_video(video_path)
    # print(f"Number of frames: {len(frames)}\nSample rate: {sample_rate}\nHeight: {height}\nWidth: {width}")

    # csv_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2/prompt-upsampled-test.csv"
    # jsonl_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2/prompt-upsampled-test.jsonl"
    # csv_to_jsonl(csv_path, jsonl_path)

    # json_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/phygenbench/prompts.json"
    # jsonl_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/phygenbench/prompts.jsonl"
    # json_to_jsonl(json_path, jsonl_path)

    # full_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2/videophy2_test.jsonl"
    # test_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2/prompt-upsampled-test.jsonl"
    # output_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2/prompt-upsampled-test-w_meta.jsonl"
    # match_jsonl(full_path, test_path, output_path)

    old_jsonl_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2/prompt-upsampled-test-w_meta.jsonl"
    new_jsonl_path = "/home/liyueyan/Interpretability/physics/wan_eval/datasets/videophy2_rewrite/prompts.jsonl"
    condition_key = "upsampled_caption"
    format_jsonl_line_vid_gen(old_jsonl_path, new_jsonl_path, condition_key)
    