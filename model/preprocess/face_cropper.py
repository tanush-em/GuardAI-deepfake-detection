import os
import glob
import cv2
import face_recognition
from tqdm.autonotebook import tqdm
from .frame_extractor import frame_extract


def create_face_videos(path_list, out_dir, max_frames=150, batch_size=4, output_size=(112, 112)):
    """
    Processes videos in path_list, extracts faces from frames, and saves cropped face videos to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    already_present_count = glob.glob(os.path.join(out_dir, '*.mp4'))
    print("No of videos already present", len(already_present_count))
    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File Already exists:", out_path)
            continue
        frames = []
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, output_size)
        for idx, frame in enumerate(frame_extract(path)):
            if idx > max_frames:
                break
            frames.append(frame)
            if len(frames) == batch_size:
                faces = face_recognition.batch_face_locations(frames)
                for i, face in enumerate(faces):
                    if len(face) != 0:
                        top, right, bottom, left = face[0]
                        try:
                            cropped = frames[i][top:bottom, left:right, :]
                            out.write(cv2.resize(cropped, output_size))
                        except Exception:
                            pass
                frames = []
        out.release() 