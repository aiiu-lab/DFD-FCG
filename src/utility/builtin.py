import os
import torch
import lightning as pl

from typing import Optional
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, RichProgressBar


class ODLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stop")
        parser.set_defaults(
            {
                "early_stop.patience": 10,
            }
        )
        parser.add_lightning_class_args(ODModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                'checkpoint.save_last': True,
                'checkpoint.save_top_k': 1,
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults(
            {
                'lr_monitor.log_momentum': True,
                'lr_monitor.logging_interval': 'step'
            }
        )

        parser.add_lightning_class_args(RichProgressBar, "progress_bar")

        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR)

        parser.add_argument("--notes", default="")
        parser.add_argument("--ckpt_path", default=None)
        parser.add_argument("--ckpt_mode", default="cont")


class ODTrainer(Trainer):
    # rewrite the log_dir property to sync with logger configurations.
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. note:: You must call this on all processes. Failing to do so will cause your program to stall forever.

         .. code-block:: python

             def training_step(self, batch, batch_idx):
                 img = ...
                 save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
            name = self.loggers[0].name
            version = self.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            dirpath = os.path.join(dirpath, str(name), version)
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class ODModelCheckpoint(ModelCheckpoint):
    # force overwrite the name mangling for checkpoint directory resolution to sync with the trainer's log directory.
    def _ModelCheckpoint__resolve_ckpt_dir(self, trainer: "pl.Trainer") -> _PATH:
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath

        return os.path.join(trainer.log_dir, "checkpoints")
