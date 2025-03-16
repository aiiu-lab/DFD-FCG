import os
import math
import argparse
from multiprocessing import Process, Pool
from glob import glob


import argparse
import copy
import os
import random

import cv2
from tqdm import tqdm
from dataclasses import dataclass
from .distortions import (block_wise, color_contrast, color_saturation,
                          gaussian_blur, gaussian_noise_color, jpeg_compression,
                          video_compression)


def parse_args():
    parser = argparse.ArgumentParser(description='Add a distortion to video.')
    parser.add_argument(
        '--dts-root',
        type=str,
        default="/scratch1/users/od/FaceForensicC23/"
    )
    parser.add_argument(
        '--vid-dir',
        type=str,
        default="videos"
    )
    parser.add_argument(
        '--glob-exp',
        type=str,
        default="*/*.mp4"
    )

    parser.add_argument(
        '--rob-dir',
        type=str,
        default="robustness"
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1
    )
    parser.add_argument(
        '--split',
        type=int,
        default=1
    )
    parser.add_argument(
        '--part',
        type=int,
        default=1
    )

    args = parser.parse_args()

    return args


def get_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
    param_dict['BW'] = [16, 32, 48, 64, 80]  # larger, worse
    param_dict['GNC'] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse
    param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse
    param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse
    param_dict['VC'] = [30, 32, 35, 38, 40]  # larger, worse

    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]


def get_distortion_function(type):
    func_dict = dict()  # a dict of function
    func_dict['CS'] = color_saturation
    func_dict['CC'] = color_contrast
    func_dict['BW'] = block_wise
    func_dict['GNC'] = gaussian_noise_color
    func_dict['GB'] = gaussian_blur
    func_dict['JPEG'] = jpeg_compression
    func_dict['VC'] = video_compression

    return func_dict[type]


def apply_distortion_log(type, level):
    if type == 'CS':
        print(f'Apply level-{level} color saturation change distortion...')
    elif type == 'CC':
        print(f'Apply level-{level} color contrast change distortion...')
    elif type == 'BW':
        print(f'Apply level-{level} local block-wise distortion...')
    elif type == 'GNC':
        print(f'Apply level-{level} white Gaussian noise in color components '
              'distortion...')
    elif type == 'GB':
        print(f'Apply level-{level} Gaussian blur distortion...')
    elif type == 'JPEG':
        print(f'Apply level-{level} JPEG compression distortion...')
    elif type == 'VC':
        print(f'Apply level-{level} video compression distortion...')


@dataclass
class Params:
    src: str
    dts_root: str
    video_root: str
    rob_dir: str


def main(params: Params):
    src, dts_root, video_root, rob_dir = params.src, params.dts_root, params.video_root, params.rob_dir

    type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC']
    level_list = [1, 2, 3, 4, 5]

    if ("FSh" in src):
        return

    frame_list = None

    for type in type_list:
        for level in level_list:
            tgt = os.path.join(
                dts_root,
                rob_dir,
                f"{type}/{level}",
                os.path.relpath(src, video_root)
            )

            if (os.path.exists(tgt)):
                continue

            if (frame_list is None):
                # extract frames
                vid = cv2.VideoCapture(src)
                fps = vid.get(cv2.CAP_PROP_FPS)
                fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
                w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f'Input video fps: {fps}')
                print(f'Input video fourcc: {fourcc}')
                print(f'Input video frame size: {w} * {h}')
                print(f'Input video frame count: {frame_count}')
                print('Extracting frames...')
                frame_list = []
                while True:
                    success, frame = vid.read()
                    if not success:
                        break
                    frame_list.append(frame)
                vid.release()
                assert len(frame_list) == frame_count

            # create output root
            root = os.path.split(tgt)[0]
            root = '.' if root == '' else root
            os.makedirs(root, exist_ok=True)

            # get distortion parameter
            dist_param = get_distortion_parameter(type, level)

            # get distortion function
            dist_function = get_distortion_function(type)

            # apply distortion
            if type == 'VC':
                apply_distortion_log(type, level)
                dist_function(src, tgt, dist_param)
            else:

                # add distortion to the frame and write to the new video at 'tgt'
                writer = cv2.VideoWriter(
                    f'{tgt[:-4]}_tmp.avi',
                    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                    fps,
                    (w, h)
                )

                apply_distortion_log(type, level)

                for frame in tqdm(frame_list):
                    new_frame = dist_function(frame.copy(), dist_param)
                    writer.write(new_frame)

                writer.release()

                cmd = f'ffmpeg -hide_banner -loglevel error -i {tgt[:-4]}_tmp.avi -y {tgt}'
                os.system(cmd)

            if os.path.exists(f'{tgt[:-4]}_tmp.avi'):
                os.remove(f'{tgt[:-4]}_tmp.avi')

            print('Finished.')


if __name__ == "__main__":
    args = parse_args()

    dts_root = args.dts_root
    vid_dir = args.vid_dir
    rob_dir = args.rob_dir

    video_root = os.path.join(dts_root, vid_dir)
    glob_exp = args.glob_exp

    videos = sorted(glob(os.path.join(video_root, glob_exp)))

    part_vids = math.ceil(len(videos) / args.split)
    start = (args.part - 1) * part_vids
    end = start + part_vids

    videos = videos[start:end]

    worker_vids = math.ceil(len(videos) / args.workers)

    with Pool(args.workers) as p:
        for _ in tqdm(
            p.imap_unordered(main, [
                Params(
                    src=src,
                    dts_root=dts_root,
                    video_root=video_root,
                    rob_dir=rob_dir
                )
                for src in videos
            ])
        ):
            continue

    print("done")
