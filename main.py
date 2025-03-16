import os
import gc
import json
import torch
import logging
import warnings
import lightning.pytorch as pl

from lightning.pytorch.utilities import rank_zero_only
from src.utility.builtin import ODTrainer, ODLightningCLI
from inference import inference_driver
torch.set_float32_matmul_precision('high')


def configure_logging():
    logging_fmt = "[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"
    logging.basicConfig(level="INFO", format=logging_fmt)
    warnings.filterwarnings(action="ignore")

    # disable warnings from the xformers efficient attention module due to torch.user_deterministic_algorithms(True,warn_only=True)
    warnings.filterwarnings(
        action="ignore",
        message=".*efficient_attention_forward_cutlass.*",
        category=UserWarning
    )

    # logging.basicConfig(level="DEBUG", format=logging_fmt)


def configure_cli():
    return ODLightningCLI(
        run=False,
        trainer_class=ODTrainer,
        save_config_kwargs={
            'config_filename': 'setting.yaml'
        },
        auto_configure_optimizers=True,
        seed_everything_default=1019
    )


def inference(cli):
    # inference the best model
    cfg_dir = cli.trainer.log_dir
    ckpt_path = cli.trainer.checkpoint_callback.best_model_path

    results = inference_driver(
        cli=cli,
        cfg_dir=cfg_dir,
        ckpt_path=ckpt_path,
    )

    # log inference results
    cli.trainer.logger.experiment.log(
        {
            "/".join(["infer", dts_name, metric]): value
            for dts_name, metrics in results.items()
            for metric, value in metrics.items()
        },
        commit=True
    )

    return results


def cli_main():
    # logging configuration
    configure_logging()

    # initialize cli
    cli = configure_cli()

    # update experiment notes
    cli.trainer.logger.experiment.notes = cli.config.notes
    cli.trainer.logger.experiment.save()

    # monitor model gradient and parameter histograms
    # (this severely slow down the training speed)
    # cli.trainer.logger.experiment.watch(cli.model, log='all', log_graph=False)

    # load & configure datasets
    cli.datamodule.affine_model(cli.model)
    cli.datamodule.affine_trainer(cli.trainer)

    # determine the purpose of the given checkpoint
    cont_ckpt_path = None
    if not cli.config.ckpt_path is None:
        if cli.config.ckpt_mode == "cont":
            cont_ckpt_path = cli.config.ckpt_path
        elif cli.config.ckpt_mode == "tune":
            cli.model.load_state_dict(torch.load(cli.config.ckpt_path)["state_dict"])
        else:
            raise NotImplementedError()

    # run
    cli.trainer.fit(
        cli.model,
        datamodule=cli.datamodule,
        ckpt_path=cont_ckpt_path
    )

    # after training:
    # 1. unwatch model
    # cli.trainer.logger.experiment.unwatch(cli.model)
    # 2. save the config
    cli.trainer.logger.experiment.save(
        glob_str=os.path.join(cli.trainer.log_dir, 'setting.yaml'),
        base_path=cli.trainer.log_dir,
        policy="now"
    )

    gc.collect()
    torch.cuda.empty_cache()

    # inference the best model.
    scores = inference(cli=cli)

    # finally
    cli.trainer.logger.experiment.finish()


if __name__ == "__main__":
    cli_main()
