import os
import cv2
import math
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import face_alignment


def load_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--fdata-dir", type=str, default="frame_data")
    parser.add_argument("--glob-exp", type=str, default="*/*")
    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--part-num", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-res", type=int, default=800)
    args = parser.parse_args(args)

    assert args.part_num > 0 and args.split_num > 0, "split and part value should be > 0"

    args.part_num = args.part_num - 1

    return args


@torch.inference_mode()
def landmark_extract(fn, org_path, batch_size, max_res):

    cap_org = cv2.VideoCapture(org_path)

    try:
        width = cap_org.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap_org.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frames = []

        # determine the scaling factor to shrink the size of input image(for efficiency).
        if (max(height, width) > max_res):
            scale = max_res / max(height, width)
        else:
            scale = 1

        while (1):
            ret_org, frame_org = cap_org.read()
            if (not ret_org):
                break
            frame_org = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
            frame_org = cv2.resize(frame_org, None, fx=scale, fy=scale)
            frames.append(frame_org)

        frame_count = len(frames)
        frame_faces = [None for _ in range(frame_count)]
        batch_indices = []
        batch_frames = []

        for cnt_frame in range(frame_count):
            batch_frames.append(frames[cnt_frame])
            batch_indices.append(cnt_frame)

            if (len(batch_frames) == batch_size or (cnt_frame == (frame_count - 1) and len(batch_frames) > 0)):

                results = fn(torch.tensor(np.stack(batch_frames).transpose((0, 3, 1, 2))))
                batch_size = len(results[0])

                batch_landmarks = results[0]
                batch_bboxes = results[2]

                for index, frame_landmarks, frame_bboxes in zip(batch_indices, batch_landmarks, batch_bboxes):
                    if (len(frame_landmarks) > 0):
                        frame_landmarks = frame_landmarks.reshape(-1, 68, 2) / scale
                        frame_landmarks = [lm for lm in frame_landmarks]
                        frame_bboxes = [bbox[:-1] / scale for bbox in frame_bboxes]
                        frame_faces[index] = {
                            "landmarks": frame_landmarks,
                            "bboxes": frame_bboxes
                        }

                batch_frames.clear()
                batch_indices.clear()

        return frame_faces

    except Exception as e:
        raise e

    finally:
        cap_org.release()


def main(args=None):
    # This file extract video landmarks from a given folder.
    # In addition, the landmarks are tracked with landmarks from previous frames.
    # By doing so, we expect to extract the most consistently appeared faces from a given video.
    # Note that under 'pack' save mode, the extracted faces must match the length of the video.
    # That's to say, if there exists a single frame without appearing faces in the video, the extract operation fails.

    args = load_args(args=args)

    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector='sfd',
        dtype=torch.float16,  # float16 to boost efficiency.
        flip_input=False,
        device="cuda",
    )

    def driver(x): return model.get_landmarks_from_batch(x, return_bboxes=True)

    if (not args.root_dir[-1] == "/"):
        args.root_dir += "/"

    video_files = sorted(glob(os.path.join(args.root_dir, args.video_dir, args.glob_exp)))
    _, video_ext = os.path.splitext(video_files[0])

    # splitting
    split_size = math.ceil(len(video_files) / args.split_num)
    video_files = video_files[args.part_num * split_size:(args.part_num + 1) * split_size]
    n_videos = len(video_files)

    print("{} videos in {}".format(n_videos, args.root_dir))
    print("path sample:{}".format(video_files[0]))

    cont = input(f"Processing Part {args.part_num+1}/{args.split_num}, Confirm?(y/n)")
    if (not cont.lower() == "y"):
        print("abort.")
        return

    for i in tqdm(range(n_videos)):
        lm_path = video_files[i].replace(args.video_dir, args.fdata_dir).replace(video_ext, '.pickle')

        if (os.path.exists(lm_path)):
            continue

        datas = landmark_extract(
            fn=driver,
            org_path=video_files[i],
            batch_size=args.batch,
            max_res=args.max_res
        )

        os.makedirs(os.path.split(lm_path)[0], exist_ok=True)
        with open(lm_path, "wb") as f:
            pickle.dump(datas, f)


if __name__ == '__main__':
    main()
