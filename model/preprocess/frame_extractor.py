import cv2

def frame_extract(path):
    """
    Generator that yields frames from a video file at the given path.
    """
    vidObj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image 