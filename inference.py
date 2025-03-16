import os
import sys
import yaml
import json
import torch
import pickle
import shutil
import logging
import warnings
import argparse


from os import path
from datetime import datetime
from torchmetrics.classification import AUROC, Accuracy
from src.utility.builtin import ODTrainer, ODLightningCLI


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_cfg_path", type=str)
    parser.add_argument("data_cfg_path", type=str)
    parser.add_argument("model_ckpt_path", type=str)
    parser.add_argument("--precision", type=str, default="16")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--notes", type=str, default='')
    return parser.parse_args(args=args)


class StatsRecorder:
    def __init__(self, label):
        self.label = label
        self.prob = 0
        self.count = 0

    def update(self, prob, label):
        assert label == self.label
        self.prob += prob
        self.count += 1

    def compute(self):
        return {
            "label": self.label,
            "prob": self.prob / self.count
        }


def configure_logging():
    logging_fmt = "[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"
    logging.basicConfig(level="INFO", format=logging_fmt)
    warnings.filterwarnings(action="ignore")


@torch.inference_mode()
def inference_driver(cli, cfg_dir, ckpt_path, notes=None):

    timestamp = datetime.now().strftime("%m%dT%H%M%S")

    trainer = cli.trainer

    # setup model
    model = cli.model

    try:
        model = model.__class__.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Unable to load model from checkpoint in strict mode: {e}")
        print(f"Loading model from checkpoint in non-strict mode.")
        model = model.__class__.load_from_checkpoint(ckpt_path, strict=False)

    model.eval()

    # setup dataset
    datamodule = cli.datamodule
    datamodule.prepare_data()
    datamodule.affine_model(cli.model)
    datamodule.setup('test')

    stats = {}
    report = {}

    test_dataloaders = datamodule.test_dataloader()

    for dts_name, dataloader in test_dataloaders.items():
        # iterate all videos
        auc_calc = AUROC(task="BINARY", num_classes=2)
        acc_calc = Accuracy(task="BINARY", num_classes=2)
        dataset = dataloader.dataset
        dts_stats = {}

        # perform ddp prediction
        batch_results = trainer.predict(
            model=model,
            dataloaders=[dataloader]
        )

        gathered_results = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_results, batch_results)
        torch.distributed.barrier()

        if (trainer.is_global_zero):
            # fetch predict results and aggregate.
            for batch_results in gathered_results:
                for batch_result in batch_results:
                    probs = batch_result["probs"]
                    names = batch_result["names"]
                    y = batch_result["y"]
                    for prob, label, name in zip(probs, y, names):
                        if (not name in dts_stats):
                            dts_stats[name] = StatsRecorder(label)
                        dts_stats[name].update(prob, label)

            # compute the average probability.
            for k in dts_stats:
                dts_stats[k] = dts_stats[k].compute()

            # add straying videos into metric calculation
            for k, v in dataset.stray_videos.items():
                dts_stats[k] = dict(
                    label=v,
                    prob=0.5,
                    stray=1
                )

            # compute the metric scores
            dataset_labels = []
            dataset_probs = []
            for v in dts_stats.values():
                dataset_labels.append(v["label"])
                dataset_probs.append(v["prob"])
            dataset_labels = torch.tensor(dataset_labels)
            dataset_probs = torch.tensor(dataset_probs)
            accuracy = acc_calc(dataset_probs, dataset_labels).item()
            roc_auc = auc_calc(dataset_probs, dataset_labels).item()
            accuracy = round(accuracy, 3)
            roc_auc = round(roc_auc, 3)
            logging.info(f'[{dts_name}] accuracy: {accuracy}, roc_auc: {roc_auc}')
            stats[dts_name] = dts_stats
            report[dts_name] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc
            }

    if (trainer.is_global_zero):
        # save report and stats.
        with open(path.join(cfg_dir, f'report_{timestamp}.json'), "w") as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

        with open(path.join(cfg_dir, f'stats_{timestamp}.pickle'), "wb") as f:
            pickle.dump(stats, f)

    return report


if __name__ == "__main__":
    configure_logging()

    params = parse_args()

    cli = ODLightningCLI(
        run=False,
        trainer_class=ODTrainer,
        save_config_callback=None,
        parser_kwargs={
            "parser_mode": "omegaconf"
        },
        auto_configure_optimizers=False,
        seed_everything_default=1019,
        args=[
            '-c', params.model_cfg_path,
            '-c', params.data_cfg_path,
            '--trainer.logger=null',
            f'--trainer.devices={params.devices}',
            f'--trainer.precision={params.precision}',
        ],
    )

    cfg_dir = os.path.split(params.model_cfg_path)[0]
    ckpt_path = params.model_ckpt_path
    notes = params.notes

    inference_driver(
        cli=cli,
        cfg_dir=cfg_dir,
        ckpt_path=ckpt_path,
        notes=notes
    )
