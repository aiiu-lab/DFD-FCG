python -m main \
--config configs/base.yaml \
--config configs/models/svl.yaml \
--model.init_args.num_synos=4 \
--config configs/generic/inference.yaml \
--notes="no ffg guidance"
