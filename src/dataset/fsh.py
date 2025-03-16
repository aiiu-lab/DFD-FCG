from .base import *


class FSh(DeepFakeDataset):
    TYPE_DIRS = {
        'REAL': 'real/',
        'FSh': 'FSh/',
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
        compressions: List[str],
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)
        assert self.split == "test", "FSh only supports the test set."
        # configurations
        self.df_types = list(FSh.TYPE_DIRS.keys())
        self.compressions = compressions

        # record missing videos in the csv file for further usage.
        self.stray_videos = {}

        # stacking data clips
        self.stack_video_clips = []

        # build video metadata structure for fast retreival
        self._build_video_table()
        self._build_video_list()

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
        for df_type, _, idx, i in self.video_list:
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

    def get_entity(self, idx, replay=None, with_entity_info=False):
        video_idx, df_type, comp, video_name, num_clips = self.video_info(idx)
        video_meta = self.video_table[df_type][comp][video_name]
        logging.debug(f"Entity/Video Index:{idx}/{video_idx}")
        logging.debug(f"Entity DF/COMP:{df_type}/{comp}")

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

            # the amount of frames to skip
            video_sample_offset = int(clip_of_video * self.clip_duration)

            # the amount of frames for the duration of a clip
            video_clip_samples = int(video_sample_freq * self.clip_duration)

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
            "comp": comp,
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

    def video_meta(self, idx):
        df_type, comp, name = self.video_info(idx)[1:4]
        return self.video_table[df_type][comp][name]

    def video_repr(self, idx):
        return '/'.join([str(i) for i in self.video_info(idx)[1:-1]])


class FShDataModule(DeepFakeDataModule):
    def __init__(
        self,
        compressions: List[str] = [],
        *args, **kargs

    ):
        super().__init__(*args, **kargs)
        self.compressions = sorted(
            set([i for i in compressions if i in FSh.COMPRESSIONS]),
            reverse=True
        )

    def prepare_data(self):
        FSh.prepare_data(self.data_dir, self.compressions, self.vid_ext)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        data_cls = partial(
            FSh,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            compressions=self.compressions,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            ratio=self.ratio,
            max_clips=self.max_clips,
            split="test",
            pack=self.pack
        )

        if (stage == "fit" or stage == "valid"):
            self._val_dataset = data_cls(
                max_clips=self.max_clips
            )
        elif (stage == "test"):
            self._test_dataset = data_cls()


if __name__ == "__main__":
    from src.utility.visualize import dataset_entity_visualize
    # logging.basicConfig(level="DEBUG")

    class Dummy():
        pass

    dtm = FShDataModule(
        ["c23"],
        data_dir="datasets/ffpp/",
        vid_ext='.avi',
        batch_size=5,
        num_workers=8,
        num_frames=10,
        clip_duration=5,
        pack=False,
        ratio=1.0,
        max_clips=3
    )

    model = Dummy()
    model.transform = lambda x: x
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")
    dtm.setup("validate")
    dtm.setup("test")

    # iterate the whole dataset for visualization and sanity check
    iterable = dtm._test_dataset
    save_folder = f"./misc/extern/dump_dataset/fsh/test/"

    # entity dump
    for entity_idx in tqdm(range(len(iterable))):
        if (entity_idx > 300):
            break
        dataset_entity_visualize(iterable.get_entity(entity_idx, with_entity_info=True), base_dir=save_folder)

    # iterate the all dataloaders for debugging.
    # for fn in [dtm.val_dataloader, dtm.test_dataloader]:
    #     iterable = fn()
    #     for batch in tqdm(iterable):
    #         pass
