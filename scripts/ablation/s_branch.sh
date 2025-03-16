python -m main \
--config configs/base.yaml \
--config configs/models/svl.yaml \
--model.init_args.op_mode=["T"] \
--config configs/generic/inference.yaml \
--notes="no spatial branch"
