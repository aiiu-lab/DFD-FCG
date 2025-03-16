import cv2
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# VIDEO_DIR = "/scratch1/users/od/CelebDF/Real/videos/*.mp4"
VIDEO_DIR = "/scratch1/users/od/FaceForensicC23/videos/*/*.mp4"
ROB_DIR = "robustness"

videos = glob(VIDEO_DIR)

print("Total videos:", len(videos))

type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC']
level_list = [1, 2, 3, 4, 5]


def runner(video):
    unmatch = []

    cap = cv2.VideoCapture(video)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    for t in type_list:
        for l in level_list:
            rob_video = video.replace("videos", f"robustness/{t}/{l}")
            cap = cv2.VideoCapture(rob_video)
            if (not frames == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                unmatch.append(rob_video)
            cap.release()

    return unmatch


results = []
with Pool(10) as p:

    for result in tqdm(
        p.imap_unordered(
            runner,
            videos
        ),
        total=len(videos)
    ):
        results.extend(result)


for i in results:
    print(i)
