# Generative Object Insertion in Gaussian Splatting with a Multi-View Diffusion Model

Code Releasement for 'Generative Object Insertion in Gaussian Splatting with a Multi-View  Diffusion Model'

Accepted by Visual Informatics. [[Paper Link]](https://www.sciencedirect.com/science/article/pii/S2468502X2500021X) [[Video Demo]](https://youtu.be/p_ZnFuhcECw?si=NtSuMB0AlPYFAh6x)

## Overview

Our code can be divided into four parts.

- `gs-simp` is modified from the [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). This is used to modify the 3D scene.
- `svd_inpaint1` is modified from the [SVD](https://github.com/Stability-AI/generative-models). This is used for multi-view inpainting.
- `Segment-and-Track-Anything-Supplementary-Code` contains a few supplementry codes for [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything). They are used to segment the target object from the inpainted frames, then re-integrate them into the original background.
- `metrics` are for metric evaluation.

We sincerely thank all the code authors.

In our running process, we need to first run in the `gs-simp` to generate inpainting inputs, then run `svd_inpaint1` for multi-view inpainting. Afterwards, we need to run `Segment-and-Track-Anything` to re-compose the inpainted object and the scene background. Finally, we run `gs-simp` again to reconstruct the object from multi-view inpainted frames. You can also refer to the `metrics` for evaluation.

## Set Up

The 'gs-simp', 'svd_inpaint1', and 'Segment-and-Track-Anything' require three individual environments. The 'metrics' only contains a few dependencies that can be easily installed. 

The environments for those modules are almost the same with the original repo, we just modified the codes based on their realization. A few packages may also needs to be installed manually, but they are all common pacages and can be easily installed with `pip`.

- Install the env for `gs-simp` following the guidance in [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). Note that we use the Gaussian Renderer with Depth output from [here](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth), so you need to install this renderer instead of the officaial one.
- Install the env for `svd_inpaint1` following the guidance in [SVD](https://github.com/Stability-AI/generative-models).
- Install the code and env for `Segment-and-Track-Anything-Supplementary-Code` following [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything). And put our supplementary files in the installed folder.

## Usage

Note that, you may need to configure the input and output paths in the running files by yourselves.

- Firstly, go to the `gs-simp` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/gs-simp) and follow the first-stage running.
- Secondly, go to the `svd_inpaint1` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/svd_inpaint1) to run the multi-view inpainting.
- Thirdly, go to the `Segment-and-Track-Anything` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/Segment-and-Track-Anything-Supplementary-Code) to run the re-integrate the inpainted object into the scene background.
- Finally, go to the `gs-simp` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/gs-simp) and follow the second-part running.
- (Optionnal) Run the `metrics/cmp.py` for metric evaluation.

## FAQ

If there is any problem, please open an issue. We will try to assist if we find time.


