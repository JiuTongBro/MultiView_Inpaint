SCENE=$1
OBJ=$2
GPU=$3

CUDA_VISIBLE_DEVICES=$GPU python sds_train.py --scene_id "${SCENE}_${OBJ}" -s dataset/${SCENE}
