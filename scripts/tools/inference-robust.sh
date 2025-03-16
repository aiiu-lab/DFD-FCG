TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
CKPT="logs/best/checkpoint.ckpt"
SETTING="logs/best/setting.yaml"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
            python -m inference \
            $SETTING \
            "./configs/robustness/$T($L).yaml" \
            $CKPT \
            --notes "$T($L)" \
            --devices -1 
    done
done