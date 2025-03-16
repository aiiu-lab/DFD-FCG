python -m main \
--config configs/base.yaml \
--config configs/models/ffg.yaml \
--model.init_args.op_mode=["S"] \
--config configs/generic/inference.yaml \
--notes="no temporal branch"
