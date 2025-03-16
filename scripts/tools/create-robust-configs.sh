TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
        cat << EOT > "configs/robustness/$T($L).yaml" 
data:
  class_path: src.dataset.base.ODDeepFakeDataModule
  init_args:
    test_datamodules:
    - class_path: src.dataset.ffpp.FFPPDataModule
      init_args:
        df_types: ['REAL','DF','FS','F2F','NT']
        compressions: ['c23']
        strategy: NORMAL
        augmentations: 
        - NONE
        force_random_speed: null
        data_dir: 'datasets/robustness/$T/$L/'
        vid_ext: .avi
EOT
    done
done