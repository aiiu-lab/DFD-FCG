import os
import argparse
from glob import glob
from src.preprocess.crop_main_face import main as crop_entrance


def parse_args():
    parser = argparse.ArgumentParser(description='Add a distortion to video.')

    parser.add_argument("action", type=str)

    parser.add_argument(
        '--dts-root',
        type=str,
        default="/scratch1/users/od/FaceForensicC23/"
    )

    parser.add_argument(
        '--glob-exp',
        type=str,
        default="*/*/*/*.mp4"
    )

    parser.add_argument(
        '--rob-dir',
        type=str,
        default="robustness"
    )

    parser.add_argument(
        '--fd-dir',
        type=str,
        default="frame_data"
    )

    parser.add_argument(
        '--crop-dir',
        type=str,
        default="cropped_robust"
    )

    parser.add_argument(
        '--mean-face',
        type=str,
        default="./misc/20words_mean_face.npy"
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    rob_root = os.path.join(args.dts_root, args.rob_dir)
    fd_root = os.path.join(args.dts_root, args.fd_dir)

    type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC']
    level_list = ['1', '2', '3', '4', '5']

    if args.action == "setup":
        for t in type_list:
            for l in level_list:
                p1 = os.path.join(fd_root)
                p2 = os.path.join(fd_root, t)
                os.makedirs(p2, exist_ok=True)
                p2 = os.path.join(p2, l)
                os.system(f"ln -s {p1} {p2}")
    elif args.action == "run":
        crop_entrance(
            [
                "--root-dir", args.dts_root,
                "--video-dir", args.rob_dir,
                "--mean-face", args.mean_face,
                "--glob-exp", args.glob_exp,
                "--crop-dir", args.crop_dir
            ]
        )
    elif args.action == "clean":
        for t in type_list:
            p = os.path.join(fd_root, t)
            os.system(f"rm -rf {p}")
