SCENE=$1
OBJ=$2
GPU=$3

CUDA_VISIBLE_DEVICES=$GPU python vis_render.py -m "output/$SCENE" --scene_id "${SCENE}_${OBJ}"
