python -m main \
--config configs/base.yaml \
--config configs/models/ffg.yaml \
--model.init_args.face_parts=["skin","eyes","nose"] \
--config configs/generic/inference.yaml \
--notes="parts_no_lips"
