Here we add two more codes to the orginal 'Segment-and-Track-Anything', which supports the segmentation on our results. You need to first install [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything), then put the two files in the official root directory to use them:
```
bash seg.sh {scenename} {casename} {ctrlid} {gpuid}
```
The results will be outputed to `gs-simp/inpaint/sam_mask`.

Now jump back to `gs-simp` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/gs-simp) and follow the second-stage running.
