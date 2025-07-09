import os
from .video_utils import find_video_files, filter_videos_by_frame_count, print_frame_stats
from .face_cropper import create_face_videos

def main():
    # Example patterns (update as needed)
    video_patterns = [
        '/content/Real videos/*.mp4',
        # Add more patterns as needed
    ]
    out_dir = '/content/drive/My Drive/FF_REAL_Face_only_data/'
    min_frames = 150

    # 1. Find video files
    video_files = find_video_files(video_patterns)

    # 2. Filter videos by frame count
    filtered_videos, frame_counts = filter_videos_by_frame_count(video_files, min_frames=min_frames)
    print("frames", frame_counts)
    print_frame_stats(frame_counts)

    # 3. Create face-cropped videos
    create_face_videos(filtered_videos, out_dir)

if __name__ == '__main__':
    main() 