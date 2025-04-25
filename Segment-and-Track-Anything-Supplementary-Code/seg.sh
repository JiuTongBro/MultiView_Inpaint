SCENE=$1
OBJ=$2
CTRLID=${3:-'-1'}
GPU=${4:-'3'}

CUDA_VISIBLE_DEVICES=$GPU python seg_gs.py "${SCENE}_${OBJ}" "$OBJ" x1 $CTRLID && python seg_gs.py "${SCENE}_${OBJ}" "$OBJ" x2 $CTRLID
