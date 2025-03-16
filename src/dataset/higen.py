from .base import *


from .base import *


class HiGen(DeepFakeDataset):

    def __init__(self, n_px, *args, **kargs):
        super().__init__(*args, **kargs)
        self._build_video_table()
        self._build_video_list()
        self.n_px = n_px
        augmentations = [
            alb.HorizontalFlip(),
            alb.Resize(
                self.n_px, self.n_px, cv2.INTER_CUBIC
            ),
            alb.RGBShift(
                (-20, 20), (-20, 20), (-20, 20), p=0.3
            ),
            alb.HueSaturationValue(
                hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3
            ),
            alb.RandomResizedCrop(
                self.n_px, self.n_px, scale=(0.6, 1.0), ratio=(1, 1), p=1.0
            ),
            alb.OneOf(
                [
                    alb.ImageCompression(
                        quality_lower=40, quality_upper=100, p=1
                    ),
                    alb.Compose(
                        [
                            alb.RandomScale(
                                (-0.4, -0.2), always_apply=True
                            ),
                            alb.Resize(
                                self.n_px, self.n_px, cv2.INTER_CUBIC, always_apply=True
                            )
                        ], p=1
                    ),
                    alb.Blur(
                        [3, 5],
                        p=1
                    )
                ],
                p=0.5
            )
        ]

        self.video_augmentation = alb.ReplayCompose(
            augmentations,
            p=1.
        )

        augmentations = [
            alb.Resize(
                self.n_px, self.n_px, cv2.INTER_CUBIC
            ),
            alb.RGBShift(
                (-5, 5), (-5, 5), (-5, 5), p=0.1
            ),
            alb.HueSaturationValue(
                hue_shift_limit=(-0.05, 0.05), sat_shift_limit=(-0.05, 0.05), val_shift_limit=(-0.05, 0.05), p=0.1
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.1
            ),
            alb.ImageCompression(
                quality_lower=80, quality_upper=100, p=0.1
            ),
        ]

        self.frame_augmentation = alb.ReplayCompose(
            augmentations,
            p=1.0
        )

        def driver(x, replay=None):
            # transform to numpy, the alb required format
            x = [_x.numpy().transpose((1, 2, 0)) for _x in x]

            # initialize replay data
            if (replay == None):
                replay = {}

            # frame augmentation
            if (not self.frame_augmentation == None):
                if ("frame" in replay):
                    assert len(replay["frame"]) == len(x), "Error! frame replay should match the number of frames"
                    x = [
                        alb.ReplayCompose.replay(
                            _r,
                            image=_x
                        )["image"]
                        for _x, _r in zip(x, replay["frame"])
                    ]

                else:
                    replay["frame"] = [None for _ in x]
                    for i, _x in enumerate(x):
                        result = self.frame_augmentation(image=_x)
                        x[i] = result["image"]
                        replay["frame"][i] = result["replay"]
            # sequence augmentation
            if (not self.video_augmentation == None):
                if ("video" in replay):
                    x = [
                        alb.ReplayCompose.replay(
                            replay["video"],
                            image=_x
                        )["image"]
                        for _x in x
                    ]
                else:
                    replay["video"] = self.video_augmentation(image=x[0])["replay"]
                    x = [
                        alb.ReplayCompose.replay(
                            replay["video"],
                            image=_x
                        )["image"]
                        for _x in x
                    ]
            # revert to tensor
            x = [torch.from_numpy(_x.transpose((2, 0, 1))) for _x in x]
            return x, replay

        self.training_augmentations = driver

    @classmethod
    def prepare_data(cls, data_dir, vid_ext):
        # compose the path for metadata cache
        meta_cache_path = path.expanduser(cls.get_cache_dir())

        # exit if cache exists
        if path.exists(meta_cache_path):
            return

        # video directory for df_type
        video_dir = path.join(data_dir, 'videos')

        # fetch video metadatas
        video_metas = cls.build_metadata(data_dir, video_dir, vid_ext)

        # cache the metadata
        makedirs(path.dirname(meta_cache_path), exist_ok=True)
        with open(meta_cache_path, 'wb') as f:
            pickle.dump(video_metas, f)

    def _build_video_table(self):
        self.video_table = {}

        # compose cache directory path
        meta_cache_path = path.expanduser(self.get_cache_dir())

        # load metadatas
        with open(meta_cache_path, 'rb') as f:
            video_metas = pickle.load(f)

        # post process the video path
        for idx in video_metas:
            video_metas[idx]["path"] = path.join(self.data_dir, video_metas[idx]["path"]) + self.vid_ext

        # store in video table.
        self.video_table = video_metas

    def _build_video_list(self):
        video_table = pd.read_csv(
            path.join(self.data_dir, 'csv_files', f'{self.split}.csv'),
            sep=' ',
            header=None,
            names=["name", "label"]
        )

        self.video_list = []
        label_videos = {
            "REAL": [],
            "FAKE": []
        }
        for index, row in video_table.iterrows():
            filename = row["name"]
            name, ext = path.splitext(filename)
            label = "REAL" if row["label"] == 0 else "FAKE"
            if name in self.video_table:
                clips = int(self.video_table[name]["duration"] // self.clip_duration)
                if (clips > 0):
                    clips = min(clips, self.max_clips)
                    label_videos[label].append((label, name, clips))
            else:
                name = f"{label}/{name}"
                logging.warning(
                    f'Video {path.join(self.data_dir, "videos", name)} does not present in the processed dataset.'
                )
                self.stray_videos[name] = (0 if label == "REAL" else 1)
        for label in label_videos:
            _videos = label_videos[label]
            self.video_list += _videos[:int(len(_videos) * self.ratio)]

        # permanant shuffle
        random.Random(1019).shuffle(self.video_list)

        # stacking up the amount of data clips for further usage
        self.stack_video_clips = [0]
        for _, _, i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
        self.stack_video_clips.pop(0)

    def __len__(self):
        if (self.pack):
            return len(self.video_list)
        else:
            return self.stack_video_clips[-1]

    def __getitem__(self, idx):
        item_entities = self.get_item(idx)
        return [[entity[k] for entity in item_entities] for k in item_entities[0].keys()]

    def get_item(self, idx, with_entity_info=False):
        item_entities = [
            self.get_entity(
                idx,
                with_entity_info=with_entity_info
            )
        ]
        return item_entities

    def get_entity(self, idx, with_entity_info=False):
        video_idx, df_type, video_name, num_clips = self.video_info(idx)
        video_meta = self.video_table[video_name]
        logging.debug(f"Entity/Video Index:{idx}/{video_idx}")
        logging.debug(f"Entity DF:{df_type}")

        # - video path
        vid_path = video_meta["path"]
        # - create video reader
        vid_reader = VideoReader(vid_path, "video")
        # - frames per second
        video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]

        entity_clips = []
        entity_masks = []

        # desire all clips in the video under pack mode, else fetch only the clip of index.
        if (self.pack):
            clips_desire = range(num_clips)
        else:
            clips_desire = [idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx - 1])]

        for clip_of_video in clips_desire:
            # video frame processing
            frames = []

            # derive the video offset
            video_offset_duration = clip_of_video * self.clip_duration

            # augment the data only while training.
            video_speed_factor = random.random() * 0.5 + 0.5
            video_shift_factor = random.random() * (1 - video_speed_factor)
            logging.debug(f"Video Speed Motion Factor: {video_speed_factor}")
            logging.debug(f"Video Shift Factor: {video_shift_factor}")

            # the amount of frames to skip
            video_sample_offset = int(
                video_offset_duration + self.clip_duration * video_shift_factor
            )
            # the amount of frames for the duration of a clip
            video_clip_samples = int(
                video_sample_freq * self.clip_duration * video_speed_factor
            )
            # the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
            if (self.num_frames == 1):
                video_sample_stride = 0
            else:
                video_sample_stride = (
                    (video_clip_samples - 1) / (self.num_frames - 1)
                ) / video_sample_freq

            logging.debug(f"Loading Video: {vid_path}")
            logging.debug(f"Sample Offset: {video_sample_offset}")
            logging.debug(f"Sample Stride: {video_sample_stride}")

            # fetch frames of clip duration
            for sample_idx in range(self.num_frames):
                vid_reader.seek(video_sample_offset + sample_idx * video_sample_stride)
                frame = next(vid_reader)
                frames.append(frame["data"])

            # augment the data only while training.
            frames, _ = self.training_augmentations(frames)
            logging.debug("Augmentations Applied.")

            # stack list of torch frames to tensor
            frames = torch.stack(frames)

            # transformation
            frames = self.transform(frames)

            # padding and masking missing frames.
            mask = torch.tensor(
                [1.] * len(frames) +
                [0.] * (self.num_frames - len(frames)),
                dtype=torch.bool
            )
            if frames.shape[0] < self.num_frames:
                diff = self.num_frames - len(frames)
                padding = torch.zeros(
                    (diff, *frames.shape[1:]),
                    dtype=frames.dtype
                )
                frames = torch.concatenate(
                    frames,
                    padding
                )

            entity_clips.append(frames)
            entity_masks.append(mask)
            logging.debug(
                "Video Clip: {}({}s~{}s), Completed!".format(
                    vid_path,
                    self.clip_duration * clip_of_video,
                    (self.clip_duration + 1) * clip_of_video
                )
            )

        del vid_reader

        entity_clips = torch.stack(entity_clips)
        entity_masks = torch.stack(entity_masks)

        entity_info = {
            "video_name": video_name,
            "df_type": df_type,
            "vid_path": vid_path
        }
        entity_data = {
            "clips": entity_clips,
            "label": 0 if (df_type == "REAL") else 1,
            "masks": entity_masks,
            "idx": idx
        }

        if with_entity_info:
            return {**entity_data, **entity_info}
        else:
            return entity_data

    def video_info(self, idx):
        if (self.pack):
            video_idx = idx
        else:
            video_idx = next(i for i, x in enumerate(self.stack_video_clips) if idx < x)
        return video_idx, *self.video_list[video_idx]

    def video_repr(self, idx):
        return '/'.join([str(i) for i in self.video_info(idx)[1:-1]])

    def video_meta(self, idx):
        df_type, name = self.video_info(idx)[1:3]
        return self.video_table[name]


class HiGenDataModule(DeepFakeDataModule):
    def __init__(
        self,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)

    def prepare_data(self):
        HiGen.prepare_data(self.data_dir, self.vid_ext)

    def affine_model(self, model):
        super().affine_model(model)
        self.n_px = model.n_px

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        data_cls = partial(
            HiGen,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            n_px=self.n_px,
            ratio=self.ratio,
            split="train",
            pack=self.pack
        )

        if (stage == "fit"):
            self._train_dataset = data_cls(
                max_clips=self.max_clips
            )


if __name__ == "__main__":
    from src.utility.visualize import dataset_entity_visualize

    class Dummy():
        pass

    dtm = HiGenDataModule(
        data_dir="datasets/higen/",
        vid_ext=".avi",
        batch_size=1,
        num_workers=4,
        num_frames=10,
        clip_duration=1,
        ratio=1.0
    )

    model = Dummy()
    model.transform = lambda x: x
    model.n_px = 224
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")

    # # iterate the whole dataset for visualization and sanity check
    iterable = dtm._train_dataset
    save_folder = f"./misc/extern/dump_dataset/higen/train/"
    # entity dump
    for entity_idx in tqdm(range(len(iterable))):
        if (entity_idx > 100):
            break
        dataset_entity_visualize(iterable.get_entity(entity_idx, with_entity_info=True), base_dir=save_folder)

    # iterate the all dataloaders for debugging.
    # for fn in [dtm.train_dataloader]:
    #     iterable = fn()
    #     for batch in tqdm(iterable):
    #         pass
