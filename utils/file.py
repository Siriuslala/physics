import cv2


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


if __name__ == "__main__":
    video_path = "/home/liyueyan/Interpretability/physics/ti2v-5B_1280*704_A man running in a desert.mp4"
    frames, sample_rate, height, width = read_video(video_path)
    print(f"Number of frames: {len(frames)}\nSample rate: {sample_rate}\nHeight: {height}\nWidth: {width}")