import cv2
from pyciede2000 import ciede2000
import numpy as np


def read_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (512, 512))
    image = np.float32(image)
    image *= 1. / 255
    Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(Lab)

    L = np.reshape(L, (-1))
    a = np.reshape(a, (-1))
    b = np.reshape(b, (-1))

    return L, a, b


def calc_ciede2000(path1, path2):
    Lc, ac, bc = read_img(path1)
    Lh, ah, bh = read_img(path2)

    cider_all = 0
    for idx in range(len(Lc)):
        tmp = ciede2000((Lc[idx], ac[idx], bc[idx]), (Lh[idx], ah[idx], bh[idx]))['delta_E_00']
        cider_all += tmp
    return cider_all / idx


def comp(x):
    #x = int(x.replace(".JPG", "").replace("_GT", "").replace('_outdoor', '').replace('_indoor', '').replace(".png",
                                                                                                            #"").replace(
        #".jpg", ""))
    return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-sr', type=str, default='/data/ImageDehazing/O-HAZE-PROCESS')
    parser.add_argument('-hr', type=str, default='/data/ImageDehazing/O-HAZE-PROCESS/HR')
    parser.add_argument('-d', type=str, default='O-haze')
    parser.add_argument('-is_diff', type=str, default='False')
    args = parser.parse_args()
    import os

    hazy_path = args.sr
    GT_path = args.hr
    datasets = args.d
    print(datasets)
    print(hazy_path)
    # print(GT_path)

    if args.is_diff == 'true':
        import os

        path = hazy_path

        all_ = 0
        idx = 0
        items = [i for i in os.listdir(path) if "_hr.png" in i]
        for item in items:
            src = os.path.join(path, item)
            dst = os.path.join(path, item.replace("_hr", "_sr"))
            ciede_tmp = calc_ciede2000(src, dst)
            all_ += ciede_tmp
            idx += 1
            print("{}:{}".format(src, ciede_tmp))
        print(all_ / idx)

    else:
        import os

        hazy_path = args.sr
        GT_path = args.hr
        datasets = args.d
        print(datasets)

        all_ = 0
        idx = 0
        hazy_items = [i for i in os.listdir(hazy_path)]
        GT_items = [i for i in os.listdir(GT_path)]
        hazy_items.sort(key=comp)
        GT_items.sort(key=comp)
        for hazy, gt in zip(hazy_items, GT_items):
            src = os.path.join(hazy_path, hazy)
            dst = os.path.join(GT_path, gt)
            ciede_tmp = calc_ciede2000(src, dst)
            all_ += ciede_tmp
            idx += 1
            print("{}/{} {}-{}:{}".format(idx, len(hazy_items), hazy, gt, ciede_tmp))
        print(all_ / idx)
