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
# from torchmetrics.functional import fid



####################
# miscellaneous
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################
def Tensor2np(tensor_list, rgb_range):

    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()


####################
# metric
####################
def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))
        

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    lpips_loss = lpips_fn(lpips.im2tensor(img1).to(device),lpips.im2tensor(img2).to(device)).item()
    # fid = calc_fid(img1, img2)

    return psnr, ssim, lpips_loss



def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calc_fid(img1, img2):
    
    img1_tensor = torch.tensor(img1.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
    img2_tensor = torch.tensor(img2.transpose((2, 0, 1)), dtype=torch.float32) / 255.0

    # Expand dimensions to batch size of 1
    img1_tensor = img1_tensor.unsqueeze(0)
    img2_tensor = img2_tensor.unsqueeze(0)

    fid_value = fid(img1_tensor, img2_tensor, device, dims=2048)
    return fid_value

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
if __name__ == "__main__":
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description='Calculate PSNR SSIM')
    parser.add_argument('--lr_folder', type=str, help='LR folder', default="/root/autodl-tmp/StableSR/Test_without_CFW/")
    parser.add_argument('--hr_folder', type=str, help='HR folder', default="/root/autodl-tmp/StableSR/Data_for_quantity/gts/")
    args = parser.parse_args()

    lr_folder = args.lr_folder
    hr_folder = args.hr_folder
    lpips_fn = lpips.LPIPS(net="vgg").cuda()
    

    lr_images = os.listdir(lr_folder)
    hr_images = os.listdir(hr_folder)
    hr_images.sort()
    lr_images.sort()
    i = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    


    
    for hr_img, lr_img in zip(hr_images, lr_images):
        img1 = os.path.join(hr_folder, hr_img)
        img2 = os.path.join(lr_folder, lr_img)
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        i = i + 1
        print("loaded{}and{}".format(hr_img,lr_img))
        psnr_val, ssim_val, lpips_loss_val = calc_metrics(img1, img2, 4, test_Y=True)
        print("Calculating...")

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_lpips += lpips_loss_val
        # total_fid += fid_val
        
        print("PSNR: {:.2f}".format(total_psnr/i))
        print("SSIM: {:.2f}".format(total_ssim/i))
        print("LPIPS: {:.2f}".format(total_lpips/i))
        # print("FID: {:.2f}".format(fid_val))
        

    avg_psnr = total_psnr / len(hr_images)
    avg_ssim = total_ssim / len(hr_images)
    avg_lpips = total_lpips / len(hr_images)
    # avg_fid = total_fid / len(hr_images)


    
    print("PSNR: {:.2f}".format(avg_psnr))
    print("SSIM: {:.2f}".format(avg_ssim))
    print("LPIPS: {:.2f}".format(avg_lpips))
    # print("FID: {:.2f}".format(avg_fid))


