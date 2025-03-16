python -m main \
--config configs/base.yaml \
--config configs/models/ffg.yaml \
--model.init_args.face_parts=["lips","skin","nose"] \
--config configs/generic/inference.yaml \
--notes="parts_no_eyes"
