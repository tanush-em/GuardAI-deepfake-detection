import glob
import cv2
import numpy as np

def find_video_files(patterns):
    """
    Given a list of glob patterns, return a list of matching video file paths.
    """
    video_files = []
    for pattern in patterns:
        video_files += glob.glob(pattern)
    return video_files


def filter_videos_by_frame_count(video_files, min_frames=150):
    """
    Remove videos with fewer than min_frames frames.
    Returns the filtered list and frame counts.
    """
    filtered = []
    frame_counts = []
    for video_file in list(video_files):
        cap = cv2.VideoCapture(video_file)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count >= min_frames:
            filtered.append(video_file)
            frame_counts.append(count)
    return filtered, frame_counts

def print_frame_stats(frame_counts):
    print("Total number of videos:", len(frame_counts))
    print("Average frame per video:", np.mean(frame_counts) if frame_counts else 0) 