
import lightning.pytorch as pl

from torch import optim, nn
from functools import partial
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader

from model.clip.lprobe import LinearProbe


# define the LightningModule
class EasyCLIPClassifier(pl.LightningModule):
    def __init__(self, output_dim=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearProbe(output_dim=output_dim)

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.model.visual.input_resolution

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            "train/loss",
            loss,
            batch_size=x.shape[0]
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # this is the test loop
        x, y = batch
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            f"test/loss",
            loss,
            batch_size=x.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # this is the validation loop
        x, y = batch
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            f"valid/loss",
            loss,
            batch_size=x.shape[0]
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./datasets/", batch_size: int = 64, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def affine_model(self, model):
        self.transform = model.transform

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
