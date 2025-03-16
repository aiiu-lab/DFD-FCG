python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/fulltune.yaml \
--optimizer.lr=1e-5 \
--trainer.accumulate_grad_batches=15 \
--config configs/inference.yaml \
--notes="full fulltune_l14"
