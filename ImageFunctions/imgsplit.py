import numpy as np
from ImageFunctions import imgbase as imgb
from tqdm import tqdm
import os
import sys
import argparse


path = sys.path[0]
sys.path.append(path[0 : path.rfind(os.path.sep)])
import filelist


default_size = (2592, 3888)


class ImageSplitting:
    def __init__(
        self, root_dir, raw_f, r_save_f, clean_f=None, c_save_f=None, save=True
    ):
        self.root_dir = root_dir
        self.raw_dir = os.path.join(root_dir, raw_f)
        self.split_clean = False
        if c_save_f != None:
            self.split_clean = True
            self.clean_dir = os.path.join(root_dir, clean_f)
        self.save = save
        if self.save:
            self.r_save_dir = os.path.join(root_dir, r_save_f)
            if self.split_clean:
                self.c_save_dir = os.path.join(root_dir, c_save_f)
        return

    def process(self, n, overlap, method, useall):
        if method == "size":
            self.func = split_img_size
        elif method == "n":
            self.func = split_img_n
        rfnames = filelist.fnames(self.raw_dir)
        print("Splitting Raw Images")
        with tqdm(total=len(rfnames)) as pbar1:
            for k in rfnames:
                img = imgb.read_image(os.path.join(self.raw_dir, k + ".jpg"), np.uint8)
                imgs = self.func(img, n, overlap, useall)
                ImageSplitting.save(imgs, os.path.join(self.r_save_dir, k))
                pbar1.update(1)
        if self.split_clean:
            cfnames = filelist.fnames(self.clean_dir)
            print("Splitting Clean Images")
            with tqdm(total=len(cfnames)) as pbar2:
                for k in cfnames:
                    img = imgb.read_image(
                        os.path.join(self.clean_dir, k + ".png"), np.uint8
                    )
                    imgs = self.func(img, n, overlap, useall)
                    ImageSplitting.save(imgs, os.path.join(self.c_save_dir, k))
                    pbar2.update(1)
        return

    @staticmethod
    def save(imgs, save_pth):
        for k in range(imgs.shape[0]):
            imgb.save_image(imgs[k], save_pth + "_" + str(k) + ".png")
        return


class ImageCombining:
    def __init__(
        self, root_dir, split_dir, save_dir, use_default_img_size=True, save_imgs=True
    ):
        self.split_dir = os.path.join(root_dir, split_dir)
        self.save_dir = os.path.join(root_dir, save_dir)
        self.use_default_img_size = use_default_img_size
        self.save_imgs = save_imgs
        return

    def process(self, n, overlap):
        if type(overlap) is not tuple:
            overlap = (overlap, overlap)
        fnames = filelist.fnames(self.split_dir)
        img_names = list(set([fname[: fname.rfind("_")] for fname in fnames]))
        full_sizes = {}
        if self.use_default_img_size:
            for img_name in img_names:
                full_sizes[img_name] = default_size
        dic = {}
        for img_name in img_names:
            dic[img_name] = 0
        for fname in fnames:
            n_found = True
            i = 0
            while n_found and i < len(img_names):
                if fname[: fname.rfind("_")] == img_names[i]:
                    dic[img_names[i]] += 1
                    n_found = False
                i += 1
        print("Combining Images")
        with tqdm(total=len(img_names)) as pbar:
            for img_name in img_names:
                partials = np.empty(
                    (dic[img_name], n[0] + 2 * overlap[0], n[1] + 2 * overlap[1]),
                    dtype=np.uint8,
                )
                for k in range(dic[img_name]):
                    partials[k] = imgb.read_image(
                        os.path.join(self.split_dir, img_name + "_" + str(k) + ".png"),
                        np.uint8,
                    )
                ImageCombining.save(
                    combine_imgs_size(n, overlap, partials, full_sizes[img_name]),
                    os.path.join(self.save_dir, img_name + ".png"),
                )
                pbar.update(1)
        return

    @staticmethod
    def save(img, save_path):
        imgb.save_image(img, save_path)
        return


def combine_imgs_size(n, overlap, partials, full_size):
    n_imgs, posx, posy = size_maker(n, full_size, overlap)
    full_image = np.empty(full_size, dtype=np.uint8)
    s = 0
    for i in range(n_imgs[0]):
        for j in range(n_imgs[1]):
            full_image[posy[i] : posy[i] + n[0], posx[j] : posx[j] + n[1]] = partials[
                s, overlap[0] : -overlap[0], overlap[1] : -overlap[1]
            ]
            s += 1
    return full_image


def size_maker(shape, size, overlap):
    n = np.ceil(np.array(size) / np.array(shape)).astype(np.int32)
    extra = n * np.array(shape) - np.array(size)
    olp = np.floor(extra / (n - 1))
    extraolp = (extra - olp * (n - 1)).astype(np.int32)
    posy = np.zeros((n[0]), dtype=np.int32)
    posx = np.zeros((n[1]), dtype=np.int32)
    ychng = (
        np.array(
            [1 for _ in range(extraolp[0])] + [0 for _ in range(n[0] - extraolp[0] - 1)]
        )
        + olp[0]
    )
    xchng = (
        np.array(
            [1 for _ in range(extraolp[1])] + [0 for _ in range(n[1] - extraolp[1] - 1)]
        )
        + olp[1]
    )
    for k in range(1, n[0]):
        posy[k] = posy[k - 1] + shape[0] - ychng[k - 1]
    for k in range(1, n[1]):
        posx[k] = posx[k - 1] + shape[1] - xchng[k - 1]
    return n, posx, posy


"""
Output Size Based:
"""


def split_img_size(img, shape, overlap, useall=True):
    size = img.shape[:2]
    if type(shape) is not tuple:
        shape = (shape, shape)
    if type(overlap) is not tuple:
        overlap = (overlap, overlap)
    n, posx, posy = size_maker(shape, size, overlap)
    if len(img.shape) == 3:
        imgo = np.zeros(
            (
                n[0] * n[1],
                shape[0] + 2 * overlap[0],
                shape[1] + 2 * overlap[1],
                3,
            ),
            dtype=np.uint8,
        )
        imgback = np.zeros(
            (size[0] + 2 * overlap[0], size[1] + 2 * overlap[1], 3),
            dtype=np.uint8,
        )
    else:
        imgo = np.zeros(
            (
                n[0] * n[1],
                shape[0] + 2 * overlap[0],
                shape[1] + 2 * overlap[1],
            ),
            dtype=np.uint8,
        )
        imgback = np.zeros(
            (size[0] + 2 * overlap[0], size[1] + 2 * overlap[1]),
            dtype=np.uint8,
        )
    imgback[overlap[0] : -overlap[0], overlap[1] : -overlap[1]] = img
    posy += overlap[0]
    posx += overlap[1]
    s = 0
    for i in range(n[0]):
        for j in range(n[1]):
            imgo[s] = imgback[
                posy[i] - overlap[0] : posy[i] + shape[0] + overlap[0],
                posx[j] - overlap[1] : posx[j] + shape[1] + overlap[1],
            ]
            s += 1
    return imgo


"""
Output Number of Images Based
"""


def split_img_n(img, n, overlap, useall=False):
    shape = img.shape
    if len(shape) == 2:
        z = np.zeros((shape[0] + 2 * overlap, shape[1] + 2 * overlap), dtype=np.uint8)
        z[overlap : shape[0] + overlap, overlap : shape[1] + overlap] = img
    else:
        z = np.zeros(
            (shape[0] + 2 * overlap, shape[1] + 2 * overlap, 3), dtype=np.uint8
        )
        z[overlap : shape[0] + overlap, overlap : shape[1] + overlap, :] = img
    img = z
    h = shape[0] // n
    w = shape[1] // n
    rn = np.arange(n, dtype=int)
    rh = rn * h + overlap
    rw = rn * w + overlap
    if not useall:
        rw = rw[1:-1]
    s = 0
    if len(shape) == 2:
        bigman = np.zeros(
            (len(rh) * len(rw), h + 2 * overlap, w + 2 * overlap),
            dtype=np.uint8,
        )
        with tqdm(total=n**2) as pbar:
            for k in rh:
                for i in rw:
                    bigman[s] = img[
                        k - overlap : (k + h) + overlap,
                        i - overlap : (i + w) + overlap,
                    ]
                    s += 1
                    pbar.update(s)
    else:
        bigman = np.zeros(
            (len(rh) * len(rw), h + 2 * overlap, w + 2 * overlap, 3),
            dtype=np.uint8,
        )
        with tqdm(total=n**2) as pbar:
            for k in rh:
                for i in rw:
                    bigman[s] = img[
                        k - overlap : (k + h) + overlap,
                        i - overlap : (i + w) + overlap,
                        :,
                    ]
                    s += 1
                    pbar.update(s)
    return bigman


"""
Recombine Images
"""


def combine_img_n(n, overlap, dir, save_dir):
    fnames = filelist.fnames2(dir)
    imgp = imgb.read_image(os.path.join(dir, fnames[0] + ".png"))
    ogshape = (imgp.shape[0] - 2 * overlap, imgp.shape[1] - 2 * overlap)
    shape = (ogshape[0] * 6, ogshape[1] * 6)
    out = np.zeros(shape, dtype=np.uint8)
    for k in range(n):
        for i in range(n):
            out[
                ogshape[0] * k : ogshape[0] * (k + 1),
                ogshape[1] * i : ogshape[1] * (i + 1),
            ] = imgb.read_image(
                os.path.join(dir, fnames[6 * k + i] + ".png"), np.uint8
            )[
                overlap : ogshape[0] + overlap,
                overlap : ogshape[1] + overlap,
            ]
    imgb.save_image(out, os.path.join(save_dir, fnames[0] + "full.png"))
    return
