from .base import *


from .base import *


class DFDC(DeepFakeDataset):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._build_video_table()
        self._build_video_list()

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

            # the amount of frames to skip
            video_sample_offset = int(video_offset_duration)

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


class DFDCDataModule(DeepFakeDataModule):
    def __init__(
        self,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)

    def prepare_data(self):
        DFDC.prepare_data(self.data_dir, self.vid_ext)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        data_cls = partial(
            DFDC,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            ratio=self.ratio,
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

    class Dummy():
        pass

    dtm = DFDCDataModule(
        data_dir="datasets/dfdc/",
        vid_ext=".avi",
        batch_size=24,
        num_workers=16,
        num_frames=10,
        clip_duration=1,
        ratio=0.5,
        pack=True
    )

    model = Dummy()
    model.transform = lambda x: x
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")
    dtm.setup("validate")
    dtm.setup("test")

    # # iterate the whole dataset for visualization and sanity check
    # iterable = dtm._test_dataset
    # save_folder = f"./misc/extern/dump_dataset/dfdc/test/"
    # # entity dump
    # for entity_idx in tqdm(range(len(iterable))):
    #     if (entity_idx > 100):
    #         break
    #     dataset_entity_visualize(iterable.get_entity(entity_idx, with_entity_info=True), base_dir=save_folder)

    # iterate the all dataloaders for debugging.
    for fn in [dtm.val_dataloader, dtm.test_dataloader]:
        iterable = fn()
        for batch in tqdm(iterable):
            pass
