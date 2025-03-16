# %%
import cv2
import pickle
import matplotlib.pyplot as plt
from src.preprocess.crop_main_face import get_video_frame_data, get_video_frames


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.release()
    return fps, frames


def get_video_frame_data(fdata_path):
    with open(fdata_path, "rb") as f:
        frame_data = pickle.load(f)
        frame_landmarks = [frame["landmarks"] if not frame is None else [] for frame in frame_data]
        frame_bboxes = [frame["bboxes"] if not frame is None else [] for frame in frame_data]

    assert len(frame_landmarks) == len(frame_bboxes)

    if (len(frame_bboxes[0][0].shape) == 1):
        frame_bboxes = [[bbox.reshape((2, 2)) for bbox in bboxes]for bboxes in frame_bboxes]

    return frame_landmarks, frame_bboxes


video_path = "/home/od/stock/FaceForensicC23/videos/real/950.mp4"
fdata_path = "/home/od/stock/FaceForensicC23/frame_data/real/950.pickle"

fps, frames = get_video_frames(video_path)
frame_landmarks, frame_bboxes = get_video_frame_data(fdata_path)

import random

idx = random.randrange(0, len(frames))
plt.imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
for lm, bbox in zip(frame_landmarks[idx], frame_bboxes[idx]):
    plt.scatter(lm[:, 0], lm[:, 1], s=1, c="r")
    plt.scatter(bbox[:, 0], bbox[:, 1], s=1, c="g")
