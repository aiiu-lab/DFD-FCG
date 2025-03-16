import os
import cv2
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import List
from os.path import exists
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count


def load_args(args):
    parser = argparse.ArgumentParser(
        description='Pre-processing'
    )
    parser.add_argument(
        '--root-dir', default=None, help='video directory'
    )
    parser.add_argument(
        '--mean-face', default='./misc/20words_mean_face.npy', help='mean face path'
    )
    parser.add_argument(
        '--crop-size', default=150, type=int, help='width of face crop'
    )
    parser.add_argument(
        '--target-size', default=256, type=int, help='the target width of affined faces.'
    )
    parser.add_argument(
        '--start-idx', default=15, type=int, help='start of landmark frame_idx'
    )
    parser.add_argument(
        '--stop-idx', default=68, type=int, help='end of landmark frame_idx'
    )
    parser.add_argument(
        '--window-margin', default=12, type=int, help='window margin for smoothed landmarks'
    )
    parser.add_argument(
        '--video-dir', default="videos", type=str, help='video folder'
    )
    parser.add_argument(
        '--fdata-dir', default="frame_data", type=str, help='frame data folder'
    )
    parser.add_argument(
        '--glob-exp', default="*/*", type=str, help='additional glob expressions.'
    )
    parser.add_argument(
        '--crop-dir', default="cropped", type=str, help="folder destination to save the process results."
    )
    parser.add_argument(
        '--max-pad-secs', default=3, type=int, help="maximum seconds to pad for the untrack faces."
    )
    parser.add_argument(
        '--min-crop-rate', default=0.9, type=float, help="minimum ratio of duration with tracked faces."
    )

    parser.add_argument(
        '--d-rate', type=float, default=0.65, help="the maximum distance between the landmarks according to the ratio of face size."
    )

    parser.add_argument(
        '--replace', action="store_true", default=False
    )

    parser.add_argument(
        '--workers', default=int(cpu_count() / 2), type=int
    )

    args = parser.parse_args(args)

    return args


class FaceData:
    def __init__(self, _lm, _bbox, _idx):
        self.ema_lm = _lm  # shape = (68, 2)
        self.ema_bbox = _bbox  # shape = (2, 2)
        self.lm = [_lm]
        self.bbox = [_bbox]
        self.idx = [_idx]
        self.paddings = 0

    def last_landmark(self):
        return self.ema_lm

    def last_bbox(self):
        return self.ema_bbox

    def face_size(self):
        bbox = self.last_bbox()
        return np.linalg.norm(bbox[0] - bbox[1], axis=-1)

    def d_lm(self, landmarks):
        return np.mean(
            np.linalg.norm(landmarks - self.last_landmark(), axis=-1),
            axis=1
        )

    def d_bbox(self, bboxes):
        return np.mean(
            np.linalg.norm(bboxes - self.last_bbox(), axis=-1),
            axis=1
        )

    def pad(self):
        self.paddings += 1

    def add(self, _lm, _bbox, _idx):
        self.ema_lm = self.ema_lm * 0.5 + _lm * 0.5
        self.ema_bbox = self.ema_bbox * 0.5 + _bbox * 0.5
        self.lm.append(_lm)
        self.bbox.append(_bbox)
        self.idx.append(_idx)

        if (self.paddings > 0):
            self.paddings = 0

    def __len__(self):
        return len(self.lm)


def get_main_face_data(frame_landmarks, frame_bboxes, d_rate, max_paddings):
    # post-process the extracted frame faces.
    # create face identity database to track landmark motion.
    face_dbs = []
    num_frames = len(frame_landmarks)
    for frame_idx, landmarks, bboxes in zip(range(num_frames), frame_landmarks, frame_bboxes):

        if (
            landmarks == None or len(landmarks) == 0 or
            bboxes == None or len(bboxes) == 0
        ):
            for face in face_dbs:
                face.pad()

        else:
            assert len(landmarks) == len(bboxes), "length of landmark and bbox in frame mismatch."
            num_faces = len(landmarks)
            landmarks = np.stack(landmarks)
            bboxes = np.stack(bboxes)

            matched_indices = {}

            # find and connect with the closest face in the database.
            for db_idx, db_face in enumerate(face_dbs):
                # face landmark and bbox motion distance.
                d = db_face.d_bbox(bboxes) + db_face.d_lm(landmarks)

                # the motion continues if the landmark motion distance is lower than 100.
                if (np.min(d) > db_face.face_size() * d_rate * 2):
                    continue
                # get the closest face in the database.
                closest_idx = np.argmin(d)
                proximity = d[closest_idx]

                if (
                    (not closest_idx in matched_indices) or
                    (matched_indices[closest_idx]["d"] > proximity)
                ):
                    matched_indices[closest_idx] = dict(d=proximity, db_idx=db_idx)

            # (hacky!) pad current frame in advance, in further process, tracked faces will reset the padding.
            for db_face in face_dbs:
                db_face.pad()

            # finalize and update the database entity.
            for face_idx, save_data in matched_indices.items():
                face_dbs[save_data["db_idx"]].add(landmarks[face_idx], bboxes[face_idx], frame_idx)

            # create new database entity for untracked landmarks.
            for face_idx, landmark, bbox in zip(range(num_faces), landmarks, bboxes):
                if face_idx in matched_indices:
                    continue
                else:
                    face_dbs.append(FaceData(landmark, bbox, frame_idx))

    # report only the most consistant face in the video.
    main_face = sorted(face_dbs, key=lambda x: len(x), reverse=True)[0]

    return main_face.lm, main_face.bbox, main_face.idx


def save_video(
    filename,
    frames,
    fps
):
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename, fourcc, fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(frame)
    writer.release()  # close the writer


def affine_transform(
    frame,
    bboxes,
    landmarks,
    reference,
    target_size,
    stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0
):
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] *= (target_size / 256)
    stable_reference[:, 1] *= (target_size / 256)

    # Warp the face patch and the landmarks
    transform = cv2.estimateAffinePartial2D(
        np.vstack([landmarks[x] for x in stable_points]),
        stable_reference, method=cv2.LMEDS
    )[0]

    transformed_frame = cv2.warpAffine(
        frame,
        transform,
        dsize=(target_size, target_size),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value
    )
    transformed_landmarks = np.matmul(
        landmarks,
        transform[:, :2].transpose()
    ) + transform[:, 2].transpose()

    transformed_bboxes = np.matmul(
        bboxes,
        transform[:, :2].transpose()
    ) + transform[:, 2].transpose()

    return transformed_frame, transformed_landmarks, transformed_bboxes


def crop_driver(
    img,
    bboxes,
    landmarks,
    size,
    start_idx,
    stop_idx
):
    center_x, center_y = np.mean(landmarks[start_idx:stop_idx], axis=0)

    if center_y - size < 0:
        center_y = size + 1
    elif (center_y + size) > img.shape[0]:
        center_y = img.shape[0] - size - 1

    if center_x - size < 0:
        center_x = size + 1
    elif (center_x + size) > img.shape[1]:
        center_x = img.shape[1] - size - 1

    uy, by = int(center_y - size), int(center_y + size)
    lx, rx = int(center_x - size), int(center_x + size)
    cutted_img = np.copy(img[uy:by, lx:rx])
    cutted_landmarks = np.copy(landmarks) - [lx, uy]
    cutted_bboxes = np.copy(bboxes) - [lx, uy]

    return cutted_img, cutted_landmarks, cutted_bboxes


def crop_patch(
    frames,
    landmarks,
    bboxes,
    indices,
    reference,
    window_margin,
    start_idx,
    stop_idx,
    crop_size,
    target_size,
):
    assert len(landmarks) == len(bboxes), f"length of landmarks and bboxes mismatch."

    crop_frames = []
    crop_bboxes = []
    crop_landmarks = []

    length = len(frames)

    # preprocess for window margin
    _landmarks = [None for _ in range(length)]
    _bboxes = [None for _ in range(length)]
    for i, idx in enumerate(indices):
        _landmarks[idx] = landmarks[i]
        _bboxes[idx] = bboxes[i]

    for frame_idx in range(length):
        # check landmark exists
        if (not frame_idx in indices):
            crop_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop_landmark = None
            crop_bbox = None
        else:
            frame = frames[frame_idx]
            margin = min(window_margin // 2, frame_idx, length - 1 - frame_idx)

            # smoothed landmarks
            smoothed_landmarks = np.mean(
                [
                    _landmarks[i]
                    for i in range(frame_idx - margin, frame_idx + margin + 1)
                    if (not _landmarks[i] is None)
                ],
                axis=0
            )
            smoothed_landmarks += (_landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0))
            # smoothed bboxes
            smoothed_bboxes = np.mean(
                [
                    _bboxes[i]
                    for i in range(frame_idx - margin, frame_idx + margin + 1)
                    if (not _bboxes[i] is None)
                ],
                axis=0
            )
            smoothed_bboxes += (_bboxes[frame_idx].mean(axis=0) - smoothed_bboxes.mean(axis=0))
            # affine transform
            transformed_frame, transformed_landmarks, transformed_bboxes = affine_transform(
                frame,
                smoothed_bboxes,
                smoothed_landmarks,
                reference,
                target_size=target_size
            )
            crop_frame, crop_landmark, crop_bbox = crop_driver(
                transformed_frame,
                transformed_bboxes,
                transformed_landmarks,
                crop_size // 2,
                start_idx=start_idx,
                stop_idx=stop_idx
            )

        assert crop_frame.shape[0] == crop_frame.shape[1] == crop_size, "crop size doesn't match."

        crop_frames.append(crop_frame)
        crop_landmarks.append(crop_landmark)
        crop_bboxes.append(crop_bbox)

    # convert to numpy array for better extensibility.
    crop_frames = np.array(crop_frames)
    crop_landmarks = crop_landmarks
    crop_bboxes = crop_bboxes

    return crop_frames, crop_landmarks, crop_bboxes


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
        frame_landmarks = [[] if frame is None else frame["landmarks"] for frame in frame_data]
        frame_bboxes = [[] if frame is None else frame["bboxes"] for frame in frame_data]

    assert len(frame_landmarks) == len(frame_bboxes), f"length of landmark and bbox mismatch."

    _98_to_68_mapping = [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
        26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44,
        45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
        77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95
    ]

    frame_landmarks = [
        [
            (lm[_98_to_68_mapping] if len(lm) == 98 else lm)
            for lm in landmarks
        ]
        for landmarks in frame_landmarks
    ]

    # assert len(frame_landmarks[0][0]) == 68, "landmark should be 68 points."

    frame_bboxes = [
        [
            (
                bbox.reshape((2, 2))
                if len(bbox.shape) == 1 else
                bbox
            )
            for bbox in bboxes
        ]
        for bboxes in frame_bboxes
    ]

    return frame_landmarks, frame_bboxes


@dataclass
class RunnerParams:
    video_path: str
    args: argparse.Namespace


def runner(params: RunnerParams):
    try:
        args = params.args
        video_path = params.video_path

        rel_video_path = os.path.splitext(os.path.relpath(video_path, args.video_root))[0]
        fdata_path = os.path.join(args.fdata_root, rel_video_path) + ".pickle"
        crop_video_path = os.path.join(args.crop_root, args.video_dir, rel_video_path) + ".avi"
        crop_fdata_path = os.path.join(args.crop_root, args.fdata_dir, rel_video_path) + ".pickle"

        if (exists(f"{crop_video_path}") and exists(f"{crop_fdata_path}") and not args.replace):
            return

        fps, frames = get_video_frames(video_path)

        frame_landmarks, frame_bboxes = get_video_frame_data(fdata_path)

        assert len(frames) == len(frame_landmarks) == len(frame_bboxes)

        landmarks, bboxes, indices = get_main_face_data(
            frame_landmarks=frame_landmarks,
            frame_bboxes=frame_bboxes,
            d_rate=args.d_rate,
            max_paddings=fps * args.max_pad_secs
        )

        if (len(landmarks) < len(frames) * args.min_crop_rate):
            raise Exception("number of tracked landmarks below the minimum ratio of frames.")

        crop_frames, crop_landmarks, crop_bboxes = crop_patch(
            frames,
            landmarks,
            bboxes,
            indices,
            args.reference,
            window_margin=args.window_margin,
            start_idx=args.start_idx,
            stop_idx=args.stop_idx,
            crop_size=args.crop_size,
            target_size=args.target_size
        )

        # save video
        os.makedirs(os.path.dirname(crop_video_path), exist_ok=True)

        save_video(crop_video_path, crop_frames, fps)

        # save frame data
        os.makedirs(os.path.dirname(crop_fdata_path), exist_ok=True)

        with open(crop_fdata_path, "wb") as f:
            assert crop_bboxes[0].shape == (2, 2)
            pickle.dump(
                [
                    dict(landmarks=[landmarks], bboxes=[bboxes])
                    for landmarks, bboxes in zip(crop_landmarks, crop_bboxes)
                ],
                f
            )
    except Exception as e:
        print("Video Process Error:", video_path, e)


def main(args=None):
    args = load_args(args)
    args.reference = np.load(args.mean_face)

    args.video_root = os.path.join(args.root_dir, args.video_dir)
    args.fdata_root = os.path.join(args.root_dir, args.fdata_dir)
    args.crop_root = os.path.join(args.root_dir, args.crop_dir)

    video_files = sorted(glob(os.path.join(args.video_root, args.glob_exp)))

    if (args.workers == 0):
        for video_path in tqdm(video_files):
            runner(RunnerParams(args=args, video_path=video_path))

    else:
        with Pool(args.workers) as p:
            for _ in tqdm(
                p.imap_unordered(
                    runner,
                    [
                        RunnerParams(
                            args=args,
                            video_path=video_path
                        )
                        for video_path in video_files
                    ]
                ),
                total=len(video_files)
            ):
                continue


if __name__ == "__main__":
    main()
