import torch
import clip
from PIL import Image
import torch
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F
import pyiqa

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


#---- CLIP

def get_text_emb(text):

    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features # 1, c


def get_img_emb(img_path):

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features # 1, c


def text_img_sim(img_path, text):

    text_feature = get_text_emb(text)
    img_feature = get_img_emb(img_path)

    score = img_feature @ text_feature.t()

    return score.detach().cpu().numpy()[0][0]


def directional_sim(origin_path, edited_path, origin_text, edited_text):

    origin_text = get_text_emb(origin_text)
    edited_text = get_text_emb(edited_text)

    origin_img = get_img_emb(origin_path)
    edited_img = get_img_emb(edited_path)

    text_feature = edited_text - origin_text
    text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)

    img_feature = edited_img - origin_img
    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)

    score = img_feature @ text_feature.t()

    return score.detach().cpu().numpy()[0][0]


def temporal_sim(origin_paths, edited_paths):

    origin_path1, origin_path2 = origin_paths
    edited_path1, edited_path2 = edited_paths

    origin_feature1 = get_img_emb(origin_path1)
    origin_feature2 = get_img_emb(origin_path2)

    edited_feature1 = get_img_emb(edited_path1)
    edited_feature2 = get_img_emb(edited_path2)

    delta_origin = origin_feature2 - origin_feature1
    delta_edited = edited_feature2 - edited_feature1

    delta_origin = delta_origin / delta_origin.norm(dim=1, keepdim=True)
    delta_edited = delta_edited / delta_edited.norm(dim=1, keepdim=True)

    score = delta_origin @ delta_edited.t()

    return score.detach().cpu().numpy()[0][0]


#---- Reality

def laplacian(img_path, n_px=512):

    frame = cv2.imread(img_path)
    img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img2gray, (n_px, n_px))
    score = cv2.Laplacian(img_resize, cv2.CV_64F).var()

    return score

class MUSIQ:

    def __init__(self):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.iqa_metric = pyiqa.create_metric('musiq', device=device)
        print('Lower Better?', self.iqa_metric.lower_better)

    def __call__(self, img_path):
        score_fr = self.iqa_metric(img_path,)
        return score_fr.detach().cpu().numpy()[0][0]


class WADIQMA:

    def __init__(self):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.iqa_metric = pyiqa.create_metric('wadiqam_nr', device=device)
        print('Lower Better?', self.iqa_metric.lower_better)

    def __call__(self, img_path):
        score_fr = self.iqa_metric(img_path,)
        return score_fr.detach().cpu().numpy()[0][0]


img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))



#---- Rec

# structural similarity index
class SSIM(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

def psnr(pd_path, gt_path, mask_path=None):

    pd_img = cv2.imread(pd_path, -1)[..., :3] / 255.
    gt_img = cv2.imread(gt_path, -1)[..., :3] / 255.

    if mask_path is not None:
        mask = cv2.imread(mask_path, -1) / 255.
        if len(mask.shape)<3: mask = mask[..., None]
        pd_img = pd_img * (1. - mask)
        gt_img = gt_img * (1. - mask)

    pd_img = torch.from_numpy(pd_img)
    gt_img = torch.from_numpy(gt_img)

    score = mse2psnr(img2mse(pd_img, gt_img)).numpy()[0]

    return score


def ssim(pd_path, gt_path, mask_path=None):

    ssim = SSIM()

    pd_img = cv2.imread(pd_path, -1)[..., :3] / 255.
    gt_img = cv2.imread(gt_path, -1)[..., :3] / 255.

    if mask_path is not None:
        mask = cv2.imread(mask_path, -1) / 255.
        if len(mask.shape)<3: mask = mask[..., None]
        pd_img = pd_img * (1. - mask)
        gt_img = gt_img * (1. - mask)

    pd_img = torch.from_numpy(pd_img).to(torch.float32)
    gt_img = torch.from_numpy(gt_img).to(torch.float32)

    pd_img = torch.permute(pd_img, (2, 0, 1))[None, ...]
    gt_img = torch.permute(gt_img, (2, 0, 1))[None, ...]

    score = ssim(pd_img, gt_img)

    return score



