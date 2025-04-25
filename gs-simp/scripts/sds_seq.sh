SCENE=$1
OBJ=$2
GPU=$3

CUDA_VISIBLE_DEVICES=$GPU python gen_seq.py --scene_id "${SCENE}_${OBJ}" -m "output_sds/${SCENE}_${OBJ}" --sds
