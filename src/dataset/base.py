import cv2
import json
import torch
import pickle
import random
import logging
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as alb
import lightning.pytorch as pl

from tqdm import tqdm
from functools import partial
from enum import IntEnum, auto, IntFlag, Enum
from os import path, scandir, makedirs
from torchvision.io import VideoReader
from torch.utils.data import Dataset, DataLoader
from typing import List, Set, Dict, Tuple, Optional, Callable, Union


# torchvision.set_video_backend("video_reader")


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, ratio_list, always_apply=False, p=0.5):
        super(RandomDownScale, self).__init__(always_apply, p)
        self.ratio_list = ratio_list

    def apply(self, image, scale=1.0, **params):
        return self.randomdownscale(image, scale)

    def randomdownscale(self, img, scale, **params):
        keep_input_shape = True
        H, W, C = img.shape
        img_ds = cv2.resize(
            img,
            (int(W / scale), int(H / scale)),
            interpolation=cv2.INTER_CUBIC
        )
        logging.debug(f"Downscale Ratio: {scale}")
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_ds

    def get_params(self):
        return {
            "scale": np.random.randint(self.ratio_list[0], self.ratio_list[1] + 1)
        }

    def get_transform_init_args_names(self):
        return ("ratio_list",)


class DeepFakeDataset(Dataset):
    @classmethod
    def get_cache_dir(cls, *args):
        return path.expanduser(
            f"./.cache/{'-'.join([cls.__name__] +[str(i) for i in args])}.pkl"
        )

    @classmethod
    def prepare_data(cls):
        raise NotImplementedError()

    @staticmethod
    def build_metadata(data_dir, video_dir, vid_ext):
        video_metas = {}
        # build metadata
        for f in scandir(video_dir):
            if vid_ext in f.name:
                try:
                    vid_reader = torchvision.io.VideoReader(
                        f.path,
                        "video"
                    )
                    fps = vid_reader.get_metadata()["video"]["fps"][0]
                    duration = vid_reader.get_metadata()["video"]["duration"][0]
                    video_metas[f.name[:-len(vid_ext)]] = {
                        "fps": fps,
                        "frames": round(duration * fps),
                        "duration": duration,
                        "path": f.path[len(data_dir):-len(vid_ext)]
                    }
                except:
                    logging.error(f"Error Occur During Video Table Creation: {f.path}")
        return video_metas

    @property
    def cls_name(self):
        return self.__class__.__name__

    def __init__(
        self,
        data_dir: str,
        vid_ext: str,
        num_frames: int,
        clip_duration: int,
        split: str,
        transform: Optional[Callable],
        pack: bool,
        ratio: float = 1.0,
        max_clips: int = 100
    ):
        self.data_dir = data_dir
        self.vid_ext = vid_ext
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.transform = transform
        self.split = split
        self.pack = pack
        self.ratio = ratio
        self.max_clips = max_clips

        # list of video infos
        self.video_list = []

        # comprehensive record of video metas
        self.video_table = {}

        # record missing videos in the csv file for further usage.
        self.stray_videos = {}

        # stacking video clips
        self.stack_video_clips = []

    def _build_video_table(cls):
        raise NotImplementedError()

    def _build_video_list(cls):
        raise NotImplementedError()

    def video_info(self, idx):
        raise NotImplementedError()

    def video_meta(self, idx):
        raise NotImplementedError()

    def video_repr(self, idx):
        return '/'.join([str(i) for i in self.video_info(idx)[1:]])

    def get_item(self, idx, with_entity_info=False):
        raise NotImplementedError()

    # The 'idx' here represents the entity index from the __getitem__.
    # Depending on self.pack, the entity index either indicates a clip or the video.
    def get_entity(self, idx, with_entity_info=False):
        raise NotImplementedError()

    def collate_fn(self, batch):
        item_videos, item_labels, item_masks, item_entity_indices = list(zip(*batch))

        batch_entity_clips = [i for l in item_videos for i in l]
        batch_entity_label = [i for l in item_labels for i in l]
        batch_entity_masks = [i for l in item_masks for i in l]
        batch_entity_indices = [i for l in item_entity_indices for i in l]

        clips = torch.cat(batch_entity_clips)
        masks = torch.cat(batch_entity_masks)

        # post-process the label & index to match the shape of corresponding clips
        num_clips_per_entity = torch.tensor([entity_clips.shape[0] for entity_clips in batch_entity_clips])
        labels = torch.tensor(batch_entity_label).repeat_interleave(num_clips_per_entity)
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
            names=names
        )


class DeepFakeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vid_ext: str,
        data_dir: str,
        num_frames: int = None,
        batch_size: int = None,
        num_workers: int = None,
        clip_duration: int = None,
        pack: bool = False,
        ratio: float = 1.0,
        max_clips: int = 100
    ):
        super().__init__()
        # generic parameters
        self.transform = lambda x: x
        self.batch_size = batch_size
        self.accum_batch = 1
        self.num_workers = num_workers

        # dataset metadata
        self.data_dir = data_dir
        self.vid_ext = vid_ext
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.pack = pack
        self.ratio = ratio
        self.max_clips = max_clips

        # dataset splits
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._predict_dataset = None

    def overwrite_params(self, force=False, **kargs):
        for k, v in kargs.items():
            cur_v = getattr(self, k)
            if not force and not type(cur_v) == type(None):
                logging.debug(f"Parameter '{k}' has specified value '{cur_v}', ignore overwrite '{v}'.")
            else:
                logging.debug(f"Overwrite parameter '{k}' with value '{v}'")
                setattr(self, k, v)

    def create_dataloader(self, dataset, train=False):
        if (type(dataset) == type(None)):
            return None
        else:
            params = dict(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                pin_memory=True
            )

            if train:
                assert (self.batch_size >= self.accum_batch)
                params["batch_size"] = int(self.batch_size / self.accum_batch)
                params["shuffle"] = True
                params["drop_last"] = True

            return DataLoader(**params)

    def affine_model(self, model):
        self.transform = model.transform

    def affine_trainer(self, trainer):
        self.accum_batch = trainer.accumulate_grad_batches

    def prepare_data(self):
        raise NotImplementedError()

    def setup(self, stage: str):
        raise NotImplementedError()

    def train_dataloader(self):
        return self.create_dataloader(self._train_dataset, train=True)

    def val_dataloader(self):
        return self.create_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self._test_dataset)

    def predict_dataloader(self):
        return self.create_dataloader(self._predict_dataset)


class ODDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datamodules: List[pl.LightningDataModule] = [],
        val_datamodules: List[pl.LightningDataModule] = [],
        test_datamodules: List[pl.LightningDataModule] = []
    ):
        super().__init__()
        self._train_datamodules = train_datamodules
        self._test_datamodules = test_datamodules
        self._val_datamodules = val_datamodules

    def affine_model(self, model):
        for dtm in [
            *self._train_datamodules,
            *self._test_datamodules,
            *self._val_datamodules
        ]:
            dtm.affine_model(model)

    def affine_trainer(self, trainer):
        for dtm in [
            *self._train_datamodules,
            *self._test_datamodules,
            *self._val_datamodules
        ]:
            dtm.affine_trainer(trainer)

    def prepare_data(self):
        for dtm in [
            *self._train_datamodules,
            *self._test_datamodules,
            *self._val_datamodules
        ]:
            dtm.prepare_data()

    def overwrite_params(self, force=False, **kargs):
        for dtm in [
            *self._train_datamodules,
            *self._test_datamodules,
            *self._val_datamodules
        ]:
            dtm.overwrite_params(force=force, **kargs)

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            for dtm in [
                *self._train_datamodules,
                *self._val_datamodules
            ]:
                dtm.setup('fit')

        if stage == "test":
            for dtm in [
                *self._test_datamodules,
                *self._val_datamodules
            ]:
                dtm.setup('test')

    def train_dataloader(self):
        dataloaders = {
            dtm._train_dataset.cls_name:
            dtm.train_dataloader()
            for dtm in self._train_datamodules
        }
        return dataloaders

    def val_dataloader(self):
        dataloaders = {
            dtm._val_dataset.cls_name:
            dtm.val_dataloader()
            for dtm in self._val_datamodules
        }
        return dataloaders

    def test_dataloader(self):
        dataloaders = {
            dtm._test_dataset.cls_name:
            dtm.test_dataloader()
            for dtm in self._test_datamodules
        }
        return dataloaders

    def predict_dataloader(self):
        return self.test_dataloader()


class ODDeepFakeDataModule(ODDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_frames: int,
        clip_duration: int,
        *args,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        global_defaults = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            num_frames=num_frames,
            clip_duration=clip_duration
        )
        self.overwrite_params(force=False, **global_defaults)
