SCENE=$1
GPU=$2
CTRLID=${3:-'-1'}

CUDA_VISIBLE_DEVICES=$GPU python vis_render.py -m "output_rec/$SCENE" --inpainted --ctrl_id $CTRLID
