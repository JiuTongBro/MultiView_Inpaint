# Preparation

Download the processed dataset [here](https://huggingface.co/datasets/jtbzhl/mvinpainter_svd/blob/main/gsdataset.zip). Unzip it and rename it as the `dataset`, put it under the `gs-simp` folder.

Reconstruct the original background GS scenes using the [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) codes. For most of the cases, we set the `sh_degreee=1`. But if you would like to generate some reflective objects, it is suggested to set the `sh_degreee=3`. Put the reconstructed scenes in the `output` folder, and place it under the `gs-simp` folder.

Run the following code to extract the point cloud of each scene：
```
python gen_pc.py -m output/{scenename}
```

Modify the bounding box, a `.obj` file of a cude, using 3D softwares like blender, and put the bd boxes in the `bds/add` folder. In our exp, we also delete several object from certain scenes for a better results, like the vase in the `garden` scene. To do so, modify a bd box and put it under the `bds/del` folder, then run the following script. This step is required for all scenes, for data format convertion, but it will not do any deletion for a scene if there is no corresponding bd boxes under `bds/del`.
```
python del.py
```

We have put some examples under `bds` directory. For adding objects, the bd box shall be named as `{scenename}_{casename}.obj`, this will be used later.

# First-stage Running

Generate the inpainting inputs such as scene BG using:
```
bash scripts/gen_seq.sh {scenename} {casename} {gpuid}
```
The results will be saved in the `inpaint` folder.

Then train a coarse geometry using SDS:
```
bash scripts/sds.sh {scenename} {casename} {gpuid}
```
The results will be saved in the `output_sds` folder.

Afterwards, generate the corresponding views of the SDS-coarse model:
```
bash scripts/sds_seq.sh {scenename} {casename} {gpuid}
```
The results will be saved in the `inpaint_sds` folder.


Then run the following command to get the coarse depth input for all scenes.
```
python gen_depth.py
```
The results will be saved in the `inpaint/depth` folder.

Ultimately, generate the 2D inpainting results for the reference view:
```
python ctrl_inpaint.py
```
The results will be saved in the `inpaint/ctrl` folder.

We generate ten 2D inpainting samples for each case, each one named as `ctrl_{ctrlid}.png`. Copy the `inpaint/ctrl` to `inpaint/ctrl1`, and delete the unwanted inpaintint results.
To reproduce our exp, we provide our inpainted images for metric computation [here](https://huggingface.co/jtbzhl/mvinpainter_svd/blob/main/ctrl1.zip).

Now move to the `svd_inpaint1` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/svd_inpaint1) for the next step.


