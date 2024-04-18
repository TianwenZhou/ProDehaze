# validated metrics
import os
import math
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import lpips
import argparse
import torch
from scipy.linalg import sqrtm
import torch
from util_image import batch_PSNR, batch_SSIM, imread, img2tensor
import pyiqa
import torch
import torchvision


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
lpips = pyiqa.create_metric('lpips', device=device)
psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
musiq = pyiqa.create_metric('musiq', device=device)
clipiqa = pyiqa.create_metric('clipiqa', device=device)

def resize_image(img, size):
    transform = torchvision.transforms.Resize(size)
    return transform(img)

if __name__ == "__main__":
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description='Calculate PSNR SSIM')
    parser.add_argument('--lr_folder', type=str, help='LR folder', default="/root/autodl-tmp/StableSR/Test_without_CFW/")
    parser.add_argument('--hr_folder', type=str, help='HR folder', default="/root/autodl-tmp/StableSR/Data_for_quantity/gts/")
    args = parser.parse_args()
    # loss_fn = lpips.LPIPS(net='alex')
    lr_folder = args.lr_folder
    hr_folder = args.hr_folder
    

    lr_images = os.listdir(lr_folder)
    hr_images = os.listdir(hr_folder)
    hr_images.sort()
    lr_images.sort()
    i = 0
    j = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_musiq = 0
    total_clipiqa = 0
    


    
    for hr_img, lr_img in zip(hr_images, lr_images):
        img1 = os.path.join(hr_folder, hr_img)
        img2 = os.path.join(lr_folder, lr_img)
        img1 = imread(img1)
        img2 = imread(img2)
        h1 = img1.shape[0]
        w1 = img1.shape[1]
        h2 = img2.shape[0]
        w2 = img2.shape[1]
        print(img1.shape, img2.shape)
        if h1 != h2 or w1 != w2:
            j = j + 1
            continue
        
        i = i + 1

        img1 = img2tensor(img1)
        img2 = img2tensor(img2)
        

        print("loaded{}and{}".format(hr_img,lr_img))
        lpips_loss_val = lpips(img1, img2).item()
        psnr_val = psnr(img1, img2).item()
        ssim_val = ssim(img1, img2).item()
        musiq_val = musiq(img2).item()
        clipiqa_val = clipiqa(img2).item()
        # img1 is the hr img
        print("Calculating...")

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_lpips += lpips_loss_val
        total_musiq += musiq_val
        total_clipiqa += clipiqa_val
        
        print("PSNR: {:.2f}".format(psnr_val))
        print("SSIM: {:.2f}".format(ssim_val))
        print("LPIPS: {:.2f}".format(lpips_loss_val))
        print("MUSIQ: {:.2f}".format(musiq_val))
        print("CLIPIQA: {:.2f}".format(clipiqa_val))
        

    avg_psnr = total_psnr / (len(lr_images) - j)
    avg_ssim = total_ssim / (len(lr_images) - j)
    avg_lpips = total_lpips / (len(lr_images) - j)
    avg_musiq = total_musiq / (len(lr_images) - j)
    avg_clipiqa = total_clipiqa / (len(lr_images) - j)


    
    print("AVG_PSNR: {:.2f}".format(avg_psnr))
    print("AVG_SSIM: {:.2f}".format(avg_ssim))
    print("AVG_LPIPS: {:.2f}".format(avg_lpips))
    print("AVG_MUSIQ: {:.2f}".format(avg_musiq))
    print("AVG_CLIPIQA: {:.2f}".format(avg_clipiqa))


