from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import network
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image, ImageFile
import math
import tifffile
from skimage import morphology
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
os.environ['KMP_DUPLICATE_LIB_OK']='True'
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def  cal_iou(pred,gt):
    i=np.sum(pred*gt)
    u=np.sum((pred+gt)>0)
    return i,u

def main(im_clip,start_h,start_w,mask,gt,p1,gt1,i_all,u_all,):
    out_mask = np.zeros((im_clip.shape[0], im_clip.shape[1]))
    im_raw = im_clip.copy()
    im = im_clip.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to('cuda')
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred_mask = model(im)
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = np.squeeze(pred_mask.cpu().detach().numpy())
    pred_mask = np.round(pred_mask)
    pred_mask = morphology.remove_small_objects(pred_mask.astype(bool), 100)
    '''if np.sum(pred_mask):
        cv2.imshow("im", im_raw)
        cv2.imshow('gt', gt*255)
        anns = mask_generator.generate(im_raw)
        for ann in anns:
            mask = ann['segmentation']
            sam_deep = np.sum(mask * pred_mask)
            sam_num = np.sum(mask)
            if (sam_deep / sam_num) > 0.6:
                # print(sam_deep)
                # print(sam_num)
                # cv2.imshow('sam_mask',(mask*1).astype('uint8')*255)
                # cv2.imshow('pred_mask',pred_mask.astype('uint8')*255)
                out_mask += mask
        out_mask[out_mask > 0] = 1
        cv2.imshow('pm', (out_mask * 255).astype('uint8'))
        cv2.waitKey()'''
    out_mask[out_mask > 0] = 1
    #cv2.imshow('pm', (out_mask * 255).astype('uint8'))
    #cv2.waitKey()
    #print(start_h,start_w)
    if np.sum(gt) > 0:
        p1 += np.sum(out_mask)
        gt1 += np.sum(gt)
        i, u = cal_iou(out_mask, gt)
        i_all[0] += i
        u_all[0] += u




 




if __name__ == "__main__":
    im = Image.open(r'D:\lhx\toUser\train\img2\img2.tif')
    im=np.array(im)
    gt = tifffile.imread(r'D:\lhx\toUser\train\img2\img2_mask.tif')
    sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    # model = SegNet(in_channels=3, num_classes=3)
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    model = model_map['deeplabv3plus_resnet101'](num_classes=1, output_stride=16)
    model.to('cuda')
    model.eval()
    # model.load_state_dict(torch.load(r'./checkpoint/segnet.pth'))
    model.load_state_dict(torch.load(r'runs/train/last.pth'))

    i_all = np.zeros(3)
    u_all = np.zeros(3)
    p1 = 0
    gt1 = 0
    mask = np.zeros((im.shape[0], im.shape[1]))
    h = im.shape[0]
    w = im.shape[1]
    r_h = math.ceil(h / 512)
    r_w = math.ceil(w / 512)
    h_jiange = h // r_h
    w_jiange = w // r_w
    for i in tqdm(range(r_h)):
        start_h = h_jiange * i
        if start_h + 512 > h:
            start_h = h - 512
        for j in tqdm(range(r_w)):
            start_w = w_jiange * j
            if start_w + 512 > w:
                start_w = w - 512
            im_clip = im[start_h:start_h + 512, start_w:start_w + 512, :]
            gt_clip = gt[start_h:start_h + 512, start_w:start_w + 512]
            main(im_clip=im_clip, start_h=start_h, start_w=start_w, mask=mask, gt=gt_clip, gt1=gt1, p1=p1, i_all=i_all,
                 u_all=u_all)
    #iou = i_all / u_all
    #print(iou)
    #print((2 * (i_all[0] / p1) * (i_all[0] / gt1)) / ((i_all[0] / p1) + (i_all[0] / gt1)))
    # cv2.imwrite(r"D:\lhx\YOLOX-main\detect_image\\test" + '.png', im)











