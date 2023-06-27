import numpy as np
import os
import torch
import torchvision
from PIL import Image
import sys
import os

path = sys.path[0]
bpath = path[: path.rfind(os.path.sep)]
sys.path.append(bpath)
import filelist


def get_interferom_path(type):
    return "Data/interferometry/img/" + type + "/"


def def_img_size():
    return (2592, 3888)


def read_image(img_path, type=np.float32):
    img = Image.open(img_path)
    imga = np.array(img).astype(type)
    img.close()
    return imga


def read_image_model(img_path):
    Pad = torchvision.transforms.Pad(100)
    img = read_image(img_path, np.uint8)
    img = Pad(torch.from_numpy(img).unsqueeze(0)).float() / 255
    return img.to("cpu")


def save_image(arr, save_path):
    img = Image.fromarray(arr)
    img.save(save_path)
    return


def move_image(img_path, save_path, type=np.uint8):
    arr = read_image(img_path, type)
    save_image(arr, save_path)
    return


def save_arr_split(arr, save_dir, fname, s, type):
    save_image(arr, os.path.join(save_dir, fname + "_" + str(s) + "." + type))
    return


def fringeseg_invdir(dir):
    fnames = filelist.fnames(dir)
    for fname in fnames:
        path = os.path.join(dir, fname + ".png")
        arr = read_image(path, np.uint8)
        arr = fringeseg_invarr(arr)
        save_image(arr, path)
    return


def fringeseg_invarr(arr):
    if len(arr.shape) == 2:
        return ((arr == 0) * 255).astype(np.uint8)
    for k in range(arr.shape[0]):
        arr[k] = ((arr[k] == 0) * 255).astype(np.uint8)
    return arr
