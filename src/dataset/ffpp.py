from .base import *


class FFPPSampleStrategy(IntEnum):
    NORMAL = auto()
    CONTRAST_RAND = auto()
    QUALITY_PAIR = auto()
    CONTRAST_PAIR = auto()
    FORCE_PAIR = auto()


class FFPPAugmentation(IntFlag):
    NONE = auto()
    DEV = auto()
    NORMAL = auto()
    ROBUSTNESS = auto()
    PERTURBATION = auto()
    # > NORMAL
    VIDEO = auto()
    VIDEO_RRC = auto()
    FRAME = auto()
    FRAME_NOISE = auto()
    # > DEV
    RGB = auto()
    HUE = auto()
    BRIGHT = auto()
    COMP = auto()
    DSCALE = auto()
    SHARPEN = auto()
    RRC = auto()
    BLUR = auto()


class FFPP(DeepFakeDataset):
    TYPE_DIRS = {
        'REAL': 'real/',
        'DF': 'DF/',
        'FS': 'FS/',
        'F2F': 'F2F/',
        'NT': 'NT/'
    }

    COMPRESSIONS = {'c23', 'raw', 'c40'}

    @classmethod
    def prepare_data(cls, data_dir, compressions, vid_ext):
        progress_bar = tqdm(cls.TYPE_DIRS.keys())
        for df_type in progress_bar:
            for comp in compressions:
                # description for progress bar
                progress_bar.set_description(f"{df_type}: {comp}/videos")

                # compose the path for metadata cache
                meta_cache_path = path.expanduser(cls.get_cache_dir(df_type, comp))

                # next entity if cache exists
                if path.exists(meta_cache_path):
                    continue

                # video directory for df_type of compression
                video_dir = path.join(data_dir, cls.TYPE_DIRS[df_type], f'{comp}/videos')

                # build metadata
                video_metas = cls.build_metadata(data_dir, video_dir, vid_ext)

                # cache the metadata
                makedirs(path.dirname(meta_cache_path), exist_ok=True)
                with open(meta_cache_path, 'wb') as f:
                    pickle.dump(video_metas, f)

    def __init__(
        self,
        df_types: List[str],
        compressions: List[str],
        n_px: int,
        strategy: FFPPSampleStrategy,
        augmentations: FFPPAugmentation,
        *args,
        force_random_speed: Optional[bool] = None,
        **kargs
    ):
        super().__init__(*args, **kargs)
        # configurations
        self.df_types = df_types
        self.compressions = compressions
        self.n_px = n_px
        self.strategy = strategy
        self.augmentations = augmentations
        self.train = (True if self.split == "train" else False)
        self.random_speed = (
            force_random_speed
            if not force_random_speed == None else
            (
                True
                if self.train else
                False
            )
        )

        # record missing videos in the csv file for further usage.
        self.stray_videos = {}

        # stacking data clips
        self.stack_video_clips = []

        # build video metadata structure for fast retreival
        self._build_video_table()
        self._build_video_list()

        # augmentation selections
        logging.debug(f"Augmentations: {str(self.augmentations)}")
        self.frame_augmentation = None
        self.video_augmentation = None
        if FFPPAugmentation.NONE in self.augmentations:
            pass

        elif FFPPAugmentation.DEV in self.augmentations:
            self.frame_augmentation = None

            if FFPPAugmentation.RGB in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=1.)
                    ],
                    p=1.
                )
            elif FFPPAugmentation.HUE in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.HueSaturationValue(
                            hue_shift_limit=(-0.3, 0.3),
                            sat_shift_limit=(-0.3, 0.3),
                            val_shift_limit=(-0.3, 0.3),
                            p=1.
                        ),
                    ],
                    p=1.
                )
            elif FFPPAugmentation.BRIGHT in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.RandomBrightnessContrast(
                            brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.
                        ),
                    ],
                    p=1.
                )
            elif FFPPAugmentation.COMP in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.ImageCompression(
                            quality_lower=40, quality_upper=100, p=1.
                        )
                    ],
                    p=1.
                )
            elif FFPPAugmentation.DSCALE in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.RandomScale(
                            (-0.6, -0.3), always_apply=True
                        ),
                        alb.Resize(
                            self.n_px, self.n_px, cv2.INTER_CUBIC, always_apply=True
                        )
                    ],
                    p=1.
                )
            elif FFPPAugmentation.SHARPEN in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
                    ],
                    p=1.
                )
            elif FFPPAugmentation.RRC in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.RandomResizedCrop(
                            self.n_px, self.n_px, scale=(0.5, 0.75), ratio=(1, 1), p=1.0
                        )
                    ]
                )
            elif FFPPAugmentation.BLUR in self.augmentations:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.Blur(p=1.0)
                    ]
                )
            else:
                raise NotImplementedError()

        elif FFPPAugmentation.ROBUSTNESS in self.augmentations:
            self.frame_augmentation = None
            self.video_augmentation = alb.ReplayCompose(
                [
                    alb.Resize(
                        self.n_px, self.n_px, cv2.INTER_CUBIC
                    ),
                    alb.RandomResizedCrop(
                        self.n_px, self.n_px, scale=(0.7, 0.9), ratio=(1, 1)
                    ),
                    alb.HorizontalFlip()
                ],
                p=1.
            )

        elif FFPPAugmentation.PERTURBATION in self.augmentations:
            augmentations = [
                alb.Resize(
                    self.n_px, self.n_px, cv2.INTER_CUBIC
                ),
                alb.RGBShift(
                    (-20, 20), (-20, 20), (-20, 20), p=0.7
                ),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.7
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.7
                ),
                alb.ImageCompression(
                    quality_lower=40, quality_upper=100, p=0.7
                ),
                alb.OneOf([
                    RandomDownScale((2, 3), p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ], p=0.5)
            ]
            self.video_augmentation = alb.ReplayCompose(
                augmentations,
                p=1.
            )

        elif FFPPAugmentation.NORMAL in self.augmentations:

            if FFPPAugmentation.VIDEO in self.augmentations:
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
                    )
                ]

                if FFPPAugmentation.VIDEO_RRC in self.augmentations:
                    augmentations += [
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

            if FFPPAugmentation.FRAME in self.augmentations:
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
                if FFPPAugmentation.FRAME_NOISE in self.augmentations:
                    augmentations += [
                        alb.OneOf(
                            [
                                alb.GaussNoise(
                                    per_channel=True,
                                    p=1.0
                                ),
                                alb.MultiplicativeNoise(
                                    per_channel=False,
                                    elementwise=True,
                                    always_apply=False,
                                    p=1.0
                                ),
                            ],
                            p=0.3
                        )
                    ]

                self.frame_augmentation = alb.ReplayCompose(
                    augmentations,
                    p=1.0
                )

            if (self.frame_augmentation == None and self.video_augmentation == None):
                raise NotImplementedError()

        else:
            raise NotImplementedError()

        # construct augmentation driver
        if (self.video_augmentation == None and self.frame_augmentation == None):
            def driver(x, replay=None):
                return x, replay

        else:
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

        # item to entity mapping list
        self.item_entity_list = []

        # construct item-entity mapping
        self._build_item_entity_list()

    def _build_video_table(self):
        self.video_table = {}
        for df_type in self.df_types:
            self.video_table[df_type] = {}
            for comp in self.compressions:
                # compose cache directory path
                meta_cache_path = path.expanduser(self.get_cache_dir(df_type, comp))

                # load metadatas
                with open(meta_cache_path, 'rb') as f:
                    video_metas = pickle.load(f)

                # post process the video path
                for idx in video_metas:
                    video_metas[idx]["path"] = path.join(self.data_dir, video_metas[idx]["path"]) + self.vid_ext

                # store in video table.
                self.video_table[df_type][comp] = video_metas

    def _build_video_list(self):
        self.video_list = []

        with open(path.join(self.data_dir, 'csv_files', f'{self.split}.json')) as f:
            idxs = json.load(f)

        logging.debug(f"DF TYPES:{self.df_types}")
        logging.debug(f"DF TYPES:{self.compressions}")

        for df_type in self.df_types:
            for comp in self.compressions:
                comp_videos = []
                adj_idxs = sorted(
                    [i for inner in idxs for i in inner]
                    if df_type == 'REAL' else
                    ['_'.join(idx) for idx in idxs] + ['_'.join(reversed(idx)) for idx in idxs]
                )

                for idx in adj_idxs:
                    if idx in self.video_table[df_type][comp]:
                        clips = int(self.video_table[df_type][comp][idx]["duration"] // self.clip_duration)
                        if (clips > 0):
                            clips = min(clips, self.max_clips)
                            comp_videos.append((df_type, comp, idx, clips))
                    else:
                        logging.warning(
                            f'Video {path.join(self.data_dir, self.TYPE_DIRS[df_type], comp, "videos", idx)} does not present in the processed dataset.'
                        )
                        self.stray_videos[f"{df_type}/{comp}/{idx}"] = (0 if df_type == "REAL" else 1)
                self.video_list += comp_videos[:int(len(comp_videos) * self.ratio)]

        # permanant shuffle
        random.Random(1019).shuffle(self.video_list)

        # stacking up the amount of data clips for further usage
        self.stack_video_clips = [0]
        self.real_clip_idx = {}
        self.fake_clip_idx = {
            'back': {},
            'fore': {}
        }
        for df_type, _, idx, i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
            interval = [self.stack_video_clips[-2], self.stack_video_clips[-1] - 1]
            if df_type == "REAL":
                self.real_clip_idx[idx] = interval
            else:
                back_idx, fore_idx = idx.split('_')
                if (not back_idx in self.fake_clip_idx['back']):
                    self.fake_clip_idx['back'][back_idx] = []
                if (not fore_idx in self.fake_clip_idx['fore']):
                    self.fake_clip_idx['fore'][fore_idx] = []
                self.fake_clip_idx['back'][back_idx].append(interval)
                self.fake_clip_idx['fore'][fore_idx].append(interval)

        self.stack_video_clips.pop(0)

    def _build_item_entity_list(self):
        self.item_entity_list = []
        seen_entity_set = set()
        for idx in range(self.num_entities):
            # if the idx is allocated, ignore
            if (idx in seen_entity_set):
                continue

            logging.debug(f"Sample strategy:{self.strategy}")
            if self.strategy == FFPPSampleStrategy.NORMAL:
                desire_entity_indices = [idx]
            elif self.strategy == FFPPSampleStrategy.CONTRAST_RAND:
                _, video_df_type, _, _, _ = self.video_info(idx)
                logging.debug(f"Source Index/DF_TYPE: {idx}/{video_df_type}")
                if (video_df_type == "REAL"):
                    logging.debug(f"Seek for a Fake Entity...")
                    ground = random.choice(list(self.fake_clip_idx.keys()))
                    video_name = random.choice(list(self.fake_clip_idx[ground].keys()))
                    interval = random.choice(self.fake_clip_idx[ground][video_name])
                    logging.debug(f"Pair with {video_name} at {ground}-ground ...")
                else:
                    logging.debug(f"Seek for a Real Entity...")
                    video_name = random.choice(list(self.real_clip_idx.keys()))
                    interval = self.real_clip_idx[video_name]
                    logging.debug(f"Pair with {video_name}...")

                c_idx = random.randint(*interval)
                desire_entity_indices = [idx, c_idx]
            elif self.strategy == FFPPSampleStrategy.CONTRAST_PAIR:
                video_idx, video_df_type, _, video_idx_name, _ = self.video_info(idx)
                offset_clip = (
                    idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx - 1])
                )
                logging.debug(f"Source Index/DF_TYPE: {idx}/{video_df_type}")
                if (video_df_type == "REAL"):
                    logging.debug(f"Seek for a Fake Entity...")
                    try:
                        ground = "back"
                        video_name = video_idx_name
                        if (not video_name in self.fake_clip_idx[ground]):
                            raise Exception("unable to pair video.")
                        interval = random.choice(self.fake_clip_idx[ground][video_name])
                        if (offset_clip > (interval[1] - interval[0])):
                            raise Exception("unable to pair video clip.")
                        c_idx = interval[0] + offset_clip
                    except Exception as e:
                        ground = random.choice(list(self.fake_clip_idx.keys()))
                        video_name = random.choice(list(self.fake_clip_idx[ground].keys()))
                        interval = random.choice(self.fake_clip_idx[ground][video_name])
                        c_idx = random.randint(*interval)
                    logging.debug(f"Pair with {video_name} at {ground}-ground ...")
                else:
                    logging.debug(f"Seek for a Real Entity...")
                    try:
                        video_name = video_idx_name.split("_")[0]
                        if (not video_name in self.real_clip_idx):
                            raise Exception("unable to pair video.")
                        interval = self.real_clip_idx[video_name]
                        if (offset_clip > (interval[1] - interval[0])):
                            raise Exception("unable to pair video clip.")
                        c_idx = interval[0] + offset_clip
                    except Exception as e:
                        video_name = random.choice(list(self.real_clip_idx.keys()))
                        interval = self.real_clip_idx[video_name]
                        c_idx = random.randint(*interval)
                    logging.debug(f"Pair with {video_name}...")

                desire_entity_indices = [idx, c_idx]
            elif self.strategy == FFPPSampleStrategy.FORCE_PAIR:
                video_idx, video_df_type, _, video_idx_name, _ = self.video_info(idx)
                offset_clip = (
                    idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx - 1])
                )
                logging.debug(f"Source Index/DF_TYPE: {idx}/{video_df_type}")
                if (video_df_type == "REAL"):
                    logging.debug(f"Seek for a Fake Entity...")
                    try:
                        ground = "back"
                        video_name = video_idx_name
                        if (not video_name in self.fake_clip_idx[ground]):
                            raise Exception("unable to pair video.")
                        interval = random.choice(self.fake_clip_idx[ground][video_name])
                        if (offset_clip > (interval[1] - interval[0])):
                            raise Exception("unable to pair video clip.")
                        c_idx = interval[0] + offset_clip
                    except Exception as e:
                        continue
                    else:
                        logging.debug(f"Pair with {video_name} at {ground}-ground ...")
                else:
                    logging.debug(f"Seek for a Real Entity...")
                    try:
                        video_name = video_idx_name.split("_")[0]
                        if (not video_name in self.real_clip_idx):
                            raise Exception("unable to pair video.")
                        interval = self.real_clip_idx[video_name]
                        if (offset_clip > (interval[1] - interval[0])):
                            raise Exception("unable to pair video clip.")
                        c_idx = interval[0] + offset_clip
                    except Exception as e:
                        continue
                    else:
                        logging.debug(f"Pair with {video_name}...")

                desire_entity_indices = [idx, c_idx]
            elif self.strategy == FFPPSampleStrategy.QUALITY_PAIR:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            logging.debug(f"Item: {idx} desires the entities: {desire_entity_indices}")

            self.item_entity_list.append(desire_entity_indices)
            seen_entity_set.update(desire_entity_indices)

    @property
    def num_entities(self):
        if (self.pack):
            return len(self.video_list)
        else:
            return self.stack_video_clips[-1]

    def __len__(self):
        return len(self.item_entity_list)

    def __getitem__(self, idx):
        item_entities = self.get_item(idx)
        return [[entity[k] for entity in item_entities] for k in item_entities[0].keys()]

    def get_item(self, idx, with_entity_info=False):
        desire_entity_indices = self.item_entity_list[idx]
        item_entities = []
        replay = None
        for desire_item_index in desire_entity_indices:
            result = self.get_entity(
                desire_item_index,
                replay=replay,
                with_entity_info=with_entity_info
            )

            if (
                self.strategy == FFPPSampleStrategy.CONTRAST_PAIR or
                self.strategy == FFPPSampleStrategy.FORCE_PAIR
            ):
                replay = result.pop("replay")
            else:
                result.pop("replay")

            item_entities.append(result)
        return item_entities

    def get_entity(self, idx, replay=None, with_entity_info=False):
        video_idx, df_type, comp, video_name, num_clips = self.video_info(idx)
        video_meta = self.video_table[df_type][comp][video_name]
        logging.debug(f"Entity/Video Index:{idx}/{video_idx}")
        logging.debug(f"Entity DF/COMP:{df_type}/{comp}")

        # - video path
        vid_path = video_meta["path"]
        # - create video reader
        vid_reader = VideoReader(vid_path, "video", num_threads=1)
        # - frames per second
        video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]

        entity_clips = []
        entity_masks = []
        entity_indices = []

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
            if (self.random_speed):
                # the slow motion factor for video data augmentation
                video_speed_factor = random.random() * 0.5 + 0.5
                video_shift_factor = random.random() * (1 - video_speed_factor)
            else:
                video_speed_factor = 1
                video_shift_factor = 0
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
            indices = []
            for sample_idx in range(self.num_frames):
                vid_reader.seek(video_sample_offset + sample_idx * video_sample_stride)
                frame = next(vid_reader)
                frames.append(frame["data"])
                fps = vid_reader.get_metadata()['video']['fps'][0]
                pts = frame["pts"]
                indices.append(round(pts * fps))

            # augment the data only while training.
            frames, replay = self.training_augmentations(frames, replay)
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
                indices += ([-1] * diff)

            entity_clips.append(frames)
            entity_masks.append(mask)
            entity_indices.append(indices)
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
            "comp": comp,
            "video_name": video_name,
            "df_type": df_type,
            "vid_path": vid_path,
            "indices": entity_indices
        }
        entity_data = {
            "clips": entity_clips,
            "label": 0 if (df_type == "REAL") else 1,
            "df_label": list(self.TYPE_DIRS.keys()).index(df_type),
            "masks": entity_masks,
            "idx": idx,
            "replay": replay
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

    def video_meta(self, idx):
        df_type, comp, name = self.video_info(idx)[1:4]
        return self.video_table[df_type][comp][name]

    def video_repr(self, idx):
        return '/'.join([str(i) for i in self.video_info(idx)[1:-1]])

    def collate_fn(self, batch):
        item_videos, item_labels, item_df_labels, item_masks, item_entity_indices = list(zip(*batch))

        batch_entity_clips = [i for l in item_videos for i in l]
        batch_entity_label = [i for l in item_labels for i in l]
        batch_entity_df_label = [i for l in item_df_labels for i in l]
        batch_entity_masks = [i for l in item_masks for i in l]
        batch_entity_indices = [i for l in item_entity_indices for i in l]

        clips = torch.cat(batch_entity_clips)
        masks = torch.cat(batch_entity_masks)

        # post-process the label & index to match the shape of corresponding clips
        num_clips_per_entity = torch.tensor([entity_clips.shape[0] for entity_clips in batch_entity_clips])
        labels = torch.tensor(batch_entity_label).repeat_interleave(num_clips_per_entity)
        df_labels = torch.tensor(batch_entity_df_label).repeat_interleave(num_clips_per_entity)
        indices = torch.tensor(batch_entity_indices).repeat_interleave(num_clips_per_entity)

        assert clips.shape[0] == masks.shape[0] == labels.shape[0] == indices.shape[0]

        dts_name = self.cls_name
        names = [self.video_repr(i) for i in indices]

        return dict(
            xyz=(
                clips,
                labels,
                dict(
                    masks=masks
                )
            ),
            indices=indices,
            dts_name=dts_name,
            names=names,
            df_labels=df_labels
        )


class FFPPDataModule(DeepFakeDataModule):
    def __init__(
        self,
        df_types: List[str] = [],
        compressions: List[str] = [],
        strategy: FFPPSampleStrategy = FFPPSampleStrategy.NORMAL,
        augmentations: List[FFPPAugmentation] = [FFPPAugmentation.NONE],
        force_random_speed: bool = None,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.df_types = sorted(
            set([i for i in df_types if i in FFPP.TYPE_DIRS]),
            reverse=True
        )
        self.compressions = sorted(
            set([i for i in compressions if i in FFPP.COMPRESSIONS]),
            reverse=True
        )
        self.strategy = strategy
        self.augmentations = FFPPAugmentation(sum(augmentations))
        self.force_random_speed = force_random_speed

    def affine_model(self, model):
        super().affine_model(model)
        self.n_px = model.n_px

    def prepare_data(self):
        FFPP.prepare_data(self.data_dir, self.compressions, self.vid_ext)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        data_cls = partial(
            FFPP,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            df_types=self.df_types,
            compressions=self.compressions,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            n_px=self.n_px,
            ratio=self.ratio,
            force_random_speed=self.force_random_speed
        )

        if stage == "fit":
            self._train_dataset = data_cls(
                split="train",
                pack=self.pack,
                strategy=self.strategy,
                augmentations=self.augmentations,
                max_clips=self.max_clips
            )
            self._val_dataset = data_cls(
                split="val",
                pack=self.pack,
                strategy=FFPPSampleStrategy.NORMAL,
                augmentations=FFPPAugmentation.NONE,
                max_clips=self.max_clips
            )

        elif stage == "test":
            self._test_dataset = data_cls(
                split="test",
                pack=self.pack,
                strategy=FFPPSampleStrategy.NORMAL,
                augmentations=FFPPAugmentation.NONE
            )


if __name__ == "__main__":
    from src.utility.visualize import dataset_entity_visualize
    # logging.basicConfig(level="DEBUG")

    class Dummy():
        pass

    dtm = FFPPDataModule(
        ["REAL", "DF", "FS", "F2F", "NT"],
        ["c23"],
        data_dir="datasets/ffpp/",
        vid_ext='.avi',
        batch_size=24,
        num_workers=8,
        num_frames=10,
        clip_duration=4,
        force_random_speed=False,
        strategy=FFPPSampleStrategy.CONTRAST_RAND,
        augmentations=[
            FFPPAugmentation.NORMAL,
            FFPPAugmentation.VIDEO,
            FFPPAugmentation.VIDEO_RRC,
            FFPPAugmentation.FRAME
        ],
        pack=False,
        ratio=0.5,
        max_clips=5
    )

    model = Dummy()
    model.n_px = 224
    model.transform = lambda x: x
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")
    dtm.setup("validate")
    dtm.setup("test")

    # # iterate the whole dataset for visualization and sanity check
    # for split, iterable in {
    #     'train': dtm._train_dataset, 'val': dtm._val_dataset, 'test': dtm._test_dataset
    # }.items():
    #     save_folder = f"./misc/extern/dump_dataset/ffpp/{split}/"
    #     # entity dump
    #     # for entity_idx in tqdm(range(len(iterable))):
    #     #     if (entity_idx > 100):
    #     #         break
    #     #     dataset_entity_visualize(iterable.get_entity(entity_idx, with_entity_info=True), base_dir=save_folder)

    #     # item dump
    #     for item_idx in tqdm(range(len(iterable))):
    #         if (item_idx > 100):
    #             break
    #         save_prefix = f"{item_idx}-"
    #         for entity_data in iterable.get_item(item_idx, with_entity_info=True):
    #             dataset_entity_visualize(entity_data, base_dir=save_folder, save_prefix=save_prefix)

    # iterate the all dataloaders for debugging.
    for fn in [dtm.train_dataloader, dtm.val_dataloader, dtm.test_dataloader]:
        iterable = fn()
        for batch in tqdm(iterable):
            pass
