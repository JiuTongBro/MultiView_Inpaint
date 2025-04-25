# Training Data and Pretrained Weights

The training data is available [here](https://huggingface.co/datasets/jtbzhl/mvinpainter_svd/blob/main/dst14_est_forward60_2k.zip).

The pretrained weights is avaliable [here](https://huggingface.co/jtbzhl/mvinpainter_svd/blob/main/simp1_release.zip).

# Preparation

Link the `gs-simp/inpaint` folder to `./gs` (`svd_inpaint1/gs`). You can either modify the path in config file.

# Testing

```
python -u test.py --base configs/test/svd_f_est_ctrl_simp1.yaml
```
Please be reminded to update the paths in the config files.

Afterwards, find the running results (`logs/xxxxxx`), config the `svd_out_root` in [Line 27](https://github.com/JiuTongBro/MultiView_Inpaint/blob/e4d0ddbdb96695224aeb29e8b0a7dbc8549c16ef/svd_inpaint1/divide_test.py#L27) of the `divide_test.py`, and run:
```
python divide_test.py
```
This step divides the concated inpainting frames into single images, and stored it in `./gs/inpainted` (since it is a softlink, also in the `gs-simp/inpaint/inpainted`).

You can then jump to the `Segment-and-Track-Anything` [directory](https://github.com/JiuTongBro/MultiView_Inpaint/tree/main/Segment-and-Track-Anything-Supplementary-Code).

# Training

```
python -u main.py --base configs/training/svd_f_est_ctrl_simp1.yaml
```
