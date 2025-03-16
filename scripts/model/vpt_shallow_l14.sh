python -m main \
--config configs/base.yaml \
--config configs/models/vpt.yaml \
--trainer.accumulate_grad_batches=10 \
--config configs/inference.yaml \
--notes="full vpt_l14"
