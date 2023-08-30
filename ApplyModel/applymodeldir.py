import argparse
import os
import torchvision
import open_model as openm
import sys
from tqdm import tqdm

path = sys.path[0]
bpath = path[0 : path.rfind(os.path.sep)]
sys.path.append(bpath)
import filelist
from ImageFunctions import imgsplit as imgs


# This file applies the specified model to a directory of raw images, and stores the traces in a new directory


def main(split, combine, img_dir, run_path, model_filename, model_type, device, pad):
    img_path = os.path.join(bpath, "Data")
    m_path = os.path.join(bpath, "Models")
    if split:
        img_dir_s = os.path.join(img_path, "extras", img_dir + "_split")
        save_dir_s = os.path.join(img_path, "extras", img_dir + "_trace_split")
        os.makedirs(img_dir_s, exist_ok=True)
        os.makedirs(save_dir_s, exist_ok=True)
        splitter = imgs.ImageSplitting(img_path, img_dir, img_dir_s)
        splitter.process((400, 600), 100, "size", True)
    else:
        img_dir_s = os.path.join(img_path, img_dir + "_split")
        save_dir_s = os.path.join(img_path, img_dir + "_trace_split")
    net = openm.openmodel_params(
        model_type, os.path.join(m_path, run_path, model_filename + ".tar"), device
    )
    if pad:
        pad = torchvision.transforms.Pad(100)
    else:
        pad = torchvision.transforms.Pad(0)
    print("Applying Model")
    fnames = filelist.fnames(img_dir_s)
    with tqdm(total=len(fnames)) as pbar:
        for k in fnames:
            openm.trace_img_save(
                net,
                os.path.join(img_dir_s, k + ".png"),
                os.path.join(save_dir_s, k + ".png"),
                pad,
            )
            pbar.update(1)

    f_save_dir = os.path.join(img_path, "modeloutputs", img_dir + "_trace")
    os.makedirs(f_save_dir)
    if combine:
        combiner = imgs.ImageCombining(
            img_path,
            save_dir_s,
            f_save_dir,
        )
        combiner.process((400, 600), 100)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def str2bool(s):
        return s.lower().startswith("t")

    # Should/Must use
    parser.add_argument(
        "-run_path",
        default=None,
        type=str,
        help="Path of file for models from a specific run",
    )
    parser.add_argument(
        "-model_filename",
        default="recent_pth",
        type=str,
        help="File name of model in specified run file",
    )
    parser.add_argument(
        "-model_type", default="NUNet", type=str, help="Type of model to use"
    )
    parser.add_argument(
        "-img_dir", default=None, type=str, help="Directory of raw images"
    )

    # Shouldnt use
    parser.add_argument(
        "-device", default="cpu", type=str, help="Device to run model on"
    )
    parser.add_argument(
        "-pad",
        default=False,
        type=str2bool,
        help="Wether or not to pad image input",
    )
    parser.add_argument(
        "-split",
        default=True,
        type=str2bool,
        help="Wether the images in the directory need to be split or not",
    )
    parser.add_argument(
        "-combine",
        default=True,
        type=str2bool,
        help="Wether to combine images back together into the full image when saved",
    )
    args = parser.parse_args()
    main(
        args.split,
        args.combine,
        args.img_dir,
        args.run_path,
        args.model_filename,
        args.model_type,
        args.device,
        args.pad,
    )
