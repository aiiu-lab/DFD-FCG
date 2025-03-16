python -m main \
--config configs/base.yaml \
--config configs/models/ffg.yaml \
--model.init_args.is_focal_loss=false \
--config configs/generic/inference.yaml \
--notes="no focal loss"
