python -m main \
--config configs/base.yaml \
--config configs/models/evl.yaml \
--data.init_args.train_datamodules.init_args.batch_size=40 \
--trainer.accumulate_grad_batches=5 \
--config configs/inference.yaml \
--notes="full evl_l14"
