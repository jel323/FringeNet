import numpy as np
from tqdm import tqdm
import torch
import torchvision
import sys
import os

path = sys.path[0]
sys.path.append(path[0 : path.rfind(os.path.sep)])
import filelist
import time

path = sys.path[0]
bpath = path[0 : path.rfind(os.path.sep)]
sys.path.append(bpath)
from ImageFunctions import imgbase as imgb
from ImageFunctions import imgsplit as imgs

from TrainingFunctions import model
from TrainingFunctions import lossfuns


def get_checkpoint(mpath, device):
    device = torch.device(device)
    checkpoint = torch.load(mpath, map_location=device)
    return checkpoint, device


def openmodel(mtype, mpath, device):
    checkpoint, device = get_checkpoint(mpath, device)
    if mtype == "UNet":
        net = model.UNet(3, 1).to(device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    return net


def openmodel_params(mtype, mpath, device, mparams=None):
    checkpoint, device = get_checkpoint(mpath, device)
    if mtype == "NUNet":
        if mparams == None:
            mparams = model.NUNet.default_params()
        net = model.NUNet(*mparams).to(device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    return net


def openmodel_disc(mtype, mpath, device):
    checkpoint, device = get_checkpoint(mpath, device)
    if mtype == "UNet":
        net = model.UNet(3, 1)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    return


def imgpath2tensor(img_path, pad=torchvision.transforms.Pad(0)):
    return img2tensor(imgb.read_image(img_path, np.uint8), pad)


def img2tensor(img: np.ndarray, pad=torchvision.transforms.Pad(0)):
    return pad(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)).float() / 255


def trace_img(net, img, pad=torchvision.transforms.Pad(0), bcrop=True):
    img_shape = img.shape[0:2]
    img = pad(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)).float() / 255
    crop = torchvision.transforms.CenterCrop(size=img_shape)
    out = net(img)
    if bcrop:
        out = crop(out)
    out = out.detach().numpy()
    out = out > 0.4
    out = (out[0, 0, :, :] * 255).astype(np.uint8)
    return out


def trace_imgpath(net, img_path, pad=torchvision.transforms.Pad(0), bcrop=True):
    img = imgb.read_image(img_path, np.uint8)
    return trace_img(net, img, pad, bcrop)


def trace_img_save(
    net, img_path, save_path, pad=torchvision.transforms.Pad(0), bcrop=True
):
    out = trace_imgpath(net, img_path, pad, bcrop)
    imgb.save_image(out, save_path)
    return


def dir_save(net, img_dir, save_dir, pad=torchvision.transforms.Pad(0)):
    filelst = filelist.fnames(img_dir)
    for k in filelst:
        trace_img_save(
            net,
            os.path.join(img_dir, k + ".png"),
            os.path.join(save_dir, k + ".png"),
            pad,
        )
    return


def dir_save_img(net, img_path, save_path, pad, n=6, overlap=5):
    t = time.time()
    raws = imgs.split_img_overlap_nosave(n, overlap, img_path)
    print(f"Time to split image up: {time.time() - t}")
    t = time.time()
    shape = raws.shape
    out = np.zeros(shape[0:3], dtype=np.uint8)
    for k in range(shape[0]):
        out[k] = trace_img(net, raws[k], pad)
    print(f"Time to apply model to all subimages: {time.time() - t}")
    t = time.time()
    full = imgs.combine_img_overlap_ez2(n, overlap, out)
    print(f"Time to recombine image: {time.time() - t}")
    imgb.save_image(full, save_path)
    return


def img_func(func, img_path, device="cpu"):
    img = imgb.read_image(img_path, np.float32)
    out = func(img)
    out = out.astype(np.float32)
    out = torch.from_numpy(out)
    return out.to(device)


def compare_models(mtype1, mpath1, mtype2, mpath2, img_path, truth, loss_fn):
    net1 = openmodel(mtype1, mpath1, "cpu")
    net2 = openmodel(mtype2, mpath2, "cpu")
    out1 = trace_imgpath(net1, img_path, "cpu")
    out2 = trace_imgpath(net2, img_path, "cpu")
    loss1 = loss_fn(truth, out1)
    loss2 = loss_fn(truth, out2)
    return loss1, loss2


def compare_model_to_func(mtype, mpath, func, img_path, truth, loss_fn):
    net = openmodel(mtype, mpath, "cpu")
    out_model = trace_imgpath(net, img_path, "cpu")
    out_func = img_func(func, img_path, "cpu")
    loss_model = loss_fn(truth, out_model)
    loss_func = loss_fn(truth, out_func)
    return loss_model, loss_func


def loss_print_models1(net1name, loss1, net2name, loss2):
    print("Loss of " + net1name + " (model 1) = ", loss1)
    print("Loss of " + net2name + " (model 2) = ", loss2)
    return


def loss_print_model_func(netname, loss_model, loss_func):
    print("Loss of " + netname + " (model) = ", loss_model)
    print("Loss of function = ", loss_func)
    return
