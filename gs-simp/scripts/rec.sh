SCENE=$1
OBJ=$2
GPU=$3
CTRLID=${4:-'-1'}

CUDA_VISIBLE_DEVICES=$GPU python inpaint_rec.py --scene_id "${SCENE}_${OBJ}" -s dataset/${SCENE} --ctrl_id $CTRLID
