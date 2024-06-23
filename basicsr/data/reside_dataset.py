import cv2
import os
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.utils as vutils
import random
from PIL import Image
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.utils.img_util import tensor2img
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from pathlib import Path
from basicsr.data.FDA import trans_image_by_ref
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)

@DATASET_REGISTRY.register()
class RESIDEDataset(data.Dataset):
    """Hazy image dataset generation
    It reads groundtruth dataset from RESIDE, and then generate hazy images on-the-fly.
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            gt_path (str): Data root path for gt.
            dataroot_depth(str): Data root path for depth
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RESIDEDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.crop_size = opt['crop_size']
        self.io_backend_opt = opt['io_backend']
        self.dataroot_depth = opt['dataroot_depth']
        self.ref_path = opt['ref_path']
        
        if 'image_type' not in opt:
            opt['image_type'] = 'png'
        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths = []
        self.depths = []
        self.ref_imgs = []
        self.ref_imgs = paths_from_folder(self.ref_path)

        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
        if 'dataroot_depth' in opt:
            if isinstance(opt['dataroot_depth'], str):
                self.depths.extend(sorted([str(x) for x in Path(opt['dataroot_depth']).glob('*.'+opt['image_type'])]))
            else:
                self.depths.extend(sorted([str(x) for x in Path(opt['dataroot_depth'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['dataroot_depth']) > 1:
                    for i in range(len(opt['dataroot_depth'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['dataroot_depth'][i+1]).glob('*.'+opt['image_type'])]))
        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt and depth images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        depth_path = self.depths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                depth_bytes = self.file_client.get(depth_path, 'depth')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                depth_path = self.depths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        
        color_strategy = np.random.rand()
        if color_strategy <= 0.5: # 50%
            strategy = 'add_haze'
    #     elif 0.3 < color_strategy <= 0.6:
    #         strategy = 'luminance'
        else:
            strategy = 'colour_cast'

        img_gt = imfrombytes(img_bytes, float32=True)
        depth = imfrombytes(depth_bytes, float32=True)

        # add_haze
        A = np.random.rand() * 1.3 + 0.5 #Generate A
        beta = 2 * np.random.rand() + 0.8 # Generate Beta
        # depth is a grayscale image


        img_f = img_gt

        td_bk = np.exp(- np.array(1 - depth) * beta)
        # td_bk = np.expand_dims(td_bk, axis=-1).repeat(3, axis=-1)
        img_bk = np.array(img_f) * td_bk + A * (1 - td_bk) # I_aug

        img_bk = img_bk / np.max(img_bk) * 255
        
        
        img_bk = img_bk[:, :, ::-1]

        if strategy == 'colour_cast':
                  # .covert('RGB')
            img_bk = Image.fromarray(np.uint8(img_bk))
            ref_num = random.randint(0, len(self.ref_imgs)-1)
            
            img_bk = trans_image_by_ref(
                    in_path=img_bk,
                    ref_path=self.ref_imgs[ref_num],
                    value=np.random.rand() * 0.002 + 0.0001
                )

        else:
            img_bk = img_bk / 255 # .covert('RGB')
  

        h, w, c = img_gt.shape[0:3]
        
        crop_pad_size = self.crop_size
        # pad
        if h != crop_pad_size or w != crop_pad_size:
            img_gt = cv2.resize(img_gt, (crop_pad_size,crop_pad_size))
            img_bk = cv2.resize(img_bk, (crop_pad_size,crop_pad_size))
           
        img_gt, img_bk = augment([img_gt,img_bk], self.opt['use_hflip'], self.opt['use_rot'])
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_bk = img2tensor([img_bk], bgr2rgb=False, float32=True)[0]
        depth = img2tensor([depth], bgr2rgb=True, float32=True)[0]


        return_d = {'gt': img_gt, 'hazy': img_bk}
        return return_d

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":
    opt = {}
    opt["scale"] = 1
    opt["phase"] = "train"
    opt["queue_size"] = 180
    opt["gt_path"] = '/home/intern/ztw/ztw/ztw/Data/RESIDE_new/HR'
    opt["dataroot_depth"] = '/home/intern/ztw/ztw/ztw/Data/RESIDE_new/HR_depth'
    opt["ref_path"] = "/home/intern/ztw/ztw/ztw/Data/RESIDE_ours/HR_hazy_src"
    opt["crop_size"] = 512
    opt["io_backend"] =  {"type": "disk"}
    opt['use_hflip'] = True
    opt['use_rot'] = True

    print(opt)
    reside_set = RESIDEDataset(opt)
    print(reside_set)

    for image in reside_set:
        
        vutils.save_image(image['gt'],'/home/intern/ztw/ztw/ztw/Methods/LatentDehazing/test_data/gt/test_1_gt.png')
        vutils.save_image(image['hazy'],'/home/intern/ztw/ztw/ztw/Methods/LatentDehazing/test_data/hazy/test_1_hazy.png')


