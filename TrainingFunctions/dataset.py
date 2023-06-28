import os
import sys
import random
import torch
import numpy as np
import torch.utils.data
import torchvision
import filelist

path = sys.path[0]
bpath = path[0 : path.rfind(os.path.sep)]
sys.path.append(os.path.join(bpath))
from ImageFunctions import imgbase as imgb


def npydataloader(dset_path, testing_samples, test_set=False):
    if not test_set:
        data = np.load(dset_path, allow_pickle=True)[testing_samples:]
    else:
        data = np.load(dset_path, allow_pickle=True)[:testing_samples]
    for k in range(data.shape[0]):
        data[k] = data[k].astype(np.float32) / 255
    return data


def str2tuple(s):
    s = s[1:-1].split(",")
    l = [int(k) for k in s]
    return tuple(l)


def read_txt(txt_pth):
    f = open(txt_pth, "r")
    lines = f.readlines()
    f.close()
    l = [k[:-1].split("-") for k in lines]
    d = {
        l[0][0][:-1]: l[0][1][1:],
        l[1][0][:-1]: l[1][1][1:],
        l[2][0][:-1]: str2tuple(l[2][1][1:]),
        l[3][0][:-1]: int(l[3][1][1:]),
    }
    return d


def fringe_img2seg_name(img_fname: str) -> str:
    ind1 = img_fname.rfind("_")
    fname = img_fname[:ind1]
    return fname + "_fringe_cleaned" + img_fname[ind1:] + ".png"


class ImgSegmentationDataset3d(torch.utils.data.Dataset):
    """
    General Dataset for fringe image segmentation from 3d images.
    """

    def __init__(
        self,
        dir: os.PathLike,
        fname: str,
    ) -> None:
        self.dir = dir
        self.fname = fname
        self.txt_pth = os.path.join(dir, "txtfiles", fname + ".txt")
        self.infodisc = read_txt(self.txt_pth)
        return

    def __len__(self) -> int:
        Exception(
            "Length method of Image Segmentation Dataset subclass not implemented"
        )
        return

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        Exception(
            "Get Item method of Image Segmentation Dataset subclass not implemented"
        )
        return

    def transformboth(
        self, img: torch.Tensor, seg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = img.size()
        tboth = torch.empty((4, size[1], size[2]))
        tboth[0:3, :, :] = img
        tboth[3, :, :] = seg

        trans = torch.nn.Sequential()
        if np.random.rand() < 0.08:
            trans.append(
                torchvision.transforms.ElasticTransform(
                    np.abs(np.random.normal(200, 150)), 10.0
                )
            )
        trans.append(torchvision.transforms.RandomHorizontalFlip(0.15))
        trans.append(torchvision.transforms.RandomVerticalFlip(0.15))
        scripted_trans = torch.jit.script(trans)
        tout = scripted_trans(tboth)
        img = tout[0:3, :, :]
        seg = tout[3]
        return img, seg

    def _get_shape(self) -> tuple[int, int]:
        return self.infodisc["n used to cut images"]

    def _get_out_shape(self) -> tuple[int, int]:
        return tuple(
            np.array(self._get_shape()) + 2 * self.infodisc["Overlap used in cutting"]
        )

    def _get_img_info(
        self,
    ) -> tuple[tuple[int], torch.FloatType, torch.FloatType, torch.FloatType]:
        img = self.__getitem__(0)[0]
        return (img.size(), img.dtype, img.max(), img.min())

    def _get_seg_info(
        self,
    ) -> tuple[tuple[int], torch.FloatType, torch.FloatType, torch.FloatType]:
        seg = self.__getitem__(0)[1]
        return (seg.size(), seg.dtype, seg.max(), seg.min())

    def _check_set_len(self) -> None:
        Exception(
            "Check Set Length method if Image Segmentation Dataset subclass not implemented"
        )
        return

    def test(self) -> None:
        assert (
            self._check_set_len()
        ), "Number of Images and Segmentation Maps \
            in the training set are not equal"
        print("Passed Tests")

        img_info = self._get_img_info()
        seg_info = self._get_seg_info()

        print(f"Fed Image Shape - {img_info[0]}")
        print(f"Image Type - {img_info[1]}")
        print(f"Image Max - {img_info[2]}")
        print(f"Image Min - {img_info[3]}")

        print(f"Fed Seg Shape - {seg_info[0]}")
        print(f"Seg Type - {seg_info[1]}")
        print(f"Seg Max - {seg_info[2]}")
        print(f"Seg Min - {seg_info[3]}")
        print("\n")
        return


class ImgSegGetOne3d(ImgSegmentationDataset3d):
    """
    Implementation of fringe image segmentation dataset. This implementation
    assumes that the dataset is stored as png images in two directories,
    one for raw images, one for fringe traces.
    """

    def __init__(
        self,
        dir: os.PathLike,
        fname: str,
        testing_samples: int = 8,
        pad: bool = False,
        test_set: bool = False,
    ) -> None:
        super().__init__(dir, fname)
        self.pad = pad
        if self.pad:
            self.Pad = torchvision.transforms.Pad(100)
        else:
            self.Pad = torchvision.transforms.Pad(0)
        self.img_dir = os.path.join(self.dir, "raw" + self.fname)
        self.seg_dir = os.path.join(self.dir, "clean" + self.fname)
        fnames = filelist.fnames(self.img_dir)
        image_fnames = np.array([k + ".png" for k in fnames])
        segment_fnames = [fringe_img2seg_name(fname) for fname in image_fnames]
        segment_fnames = np.array(segment_fnames)

        if not test_set:
            self.img_fnames = image_fnames[testing_samples:]
            self.seg_fnames = segment_fnames[testing_samples:]
        else:
            self.img_fnames = image_fnames[:testing_samples]
            self.seg_fnames = segment_fnames[:testing_samples]

        assert self.img_fnames.shape == self.seg_fnames.shape
        return

    def __len__(self) -> int:
        return self.img_fnames.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(
            imgb.read_image(os.path.join(self.img_dir, self.img_fnames[idx]), np.uint8)
        ).permute(2, 0, 1)
        seg = torch.from_numpy(
            imgb.read_image(os.path.join(self.seg_dir, self.seg_fnames[idx]), np.uint8)
        )
        img, seg = self.transformboth(img, seg)
        return (
            self.Pad(img).float() / 255,
            seg.unsqueeze(0).float() / 255,
        )

    def _check_set_len(self) -> bool:
        return len(self.img_fnames) == len(self.seg_fnames)

    def _check_copies(self) -> bool:
        return len(sorted(self.img_fnames)) == len(sorted(self.seg_fnames))

    def _check_set_values(self) -> bool:
        def get_key(fname):
            if fname[13] == "p":
                type = "p"
            else:
                type = "s"
            return fname[0:5] + fname[7] + type

        dici = {}
        dics = {}
        for k in self.img_fnames:
            key = get_key(k)
            dici[key] = 0
            dics[key] = 0
        for k in self.img_fnames:
            dici[get_key(k)] = dici[get_key(k)] + 1
        for k in self.seg_fnames:
            dics[get_key(k)] = dics[get_key(k)] + 1
        ans = True
        for k in dici.keys():
            if dici[k] != dics[k]:
                ans = False
        return ans


class ImgDatasetStore3d(ImgSegmentationDataset3d):
    """
    Implementation of fringe image segmentation dataset. This implementation
    assumes that the dataset is stored as npy files, one npy file for raw
    images, one for fringe traces.
    """

    def __init__(
        self,
        dir: os.PathLike,
        fname: str,
        testing_samples: int = 8,
        use_trans: bool = True,
        pad: bool = False,
        test_set: bool = False,
    ) -> None:
        super().__init__(dir, fname)
        self.use_trans = use_trans
        self.pad = pad
        self.test_set = test_set
        if self.pad:
            self.Pad = torchvision.transforms.Pad(100)
        else:
            self.Pad = torchvision.transforms.Pad(0)
        self.img_pth = os.path.join(dir, "npy", fname, "img.npy")
        self.seg_pth = os.path.join(dir, "npy", fname, "seg.npy")
        if not self.test_set:
            self.img = np.load(self.img_pth)[testing_samples:]
            self.seg = np.load(self.seg_pth)[testing_samples:]
        else:
            self.img = np.load(self.img_pth)[:testing_samples]
            self.seg = np.load(self.seg_pth)[:testing_samples]
        return

    def __len__(self) -> int:
        return self.img.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.img[idx]).permute(2, 0, 1)
        seg = torch.from_numpy(self.seg[idx])
        if not self.test_set and self.use_trans:
            img, seg = self.transformboth(img, seg)
        return (
            self.Pad(img).float() / 255,
            seg.unsqueeze(0).float() / 255,
        )

    def _check_set_len(self) -> bool:
        return self.img.shape[0] == self.seg.shape[0]


class ImgDatasetGetOne(torch.utils.data.Dataset):
    def __init__(self, img_dir, seg_dir, testing_samples=8, test_set=False):
        self.Pad = torchvision.transforms.Pad(100)
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        fnames = filelist.fnames(img_dir)
        image_fnames = np.array([k + ".png" for k in fnames])
        segment_fnames = []
        for k in fnames:
            ind = k.rfind("_")
            snum = k[ind + 1 :]
            num = int(snum)
            fname = k[:ind]
            segment_fnames.append(fname + "_fringe_cleaned" + "_" + str(num) + ".png")
        segment_fnames = np.array(segment_fnames)
        # segment_fnames = np.array([k + '_fringe_cleaned.png' for k in fnames])
        if not test_set:
            self.img_fnames = image_fnames[: -1 * testing_samples]
            self.seg_fnames = segment_fnames[: -1 * testing_samples]
        else:
            self.img_fnames = image_fnames[:testing_samples]
            self.seg_fnames = segment_fnames[:testing_samples]
        assert self.img_fnames.shape == self.seg_fnames.shape

    def __len__(self):
        return self.img_fnames.shape[0]

    def __getitem__(self, idx: int):
        img = imgb.read_image(
            os.path.join(self.img_dir, self.img_fnames[idx]), np.uint8
        )
        seg = imgb.read_image(
            os.path.join(self.seg_dir, self.seg_fnames[idx]), np.uint8
        )
        return (
            self.Pad(torch.from_numpy(img).unsqueeze(0)).float() / 255,
            torch.from_numpy(seg).unsqueeze(0).float() / 255,
        )

    def _get_shape(self):
        return imgb.read_image(
            os.path.join(self.img_dir, self.img_fnames[0]), np.uint8
        ).shape

    def _get_img_info(self):
        img = imgb.read_image(os.path.join(self.img_dir, self.img_fnames[0]), np.uint8)
        shape = img.shape
        img = self.Pad(torch.from_numpy(img).unsqueeze(0)).float() / 255
        return (shape, img.size(), img.dtype, img.max(), img.min())

    def _get_seg_info(self):
        seg = imgb.read_image(os.path.join(self.seg_dir, self.seg_fnames[0]), np.uint8)
        shape = seg.shape
        seg = torch.from_numpy(seg).unsqueeze(0).float() / 255
        return (shape, seg.size(), seg.dtype, seg.max(), seg.min())

    def _check_set_len(self):
        return len(self.img_fnames) == len(self.seg_fnames)

    def _check_set_values(self):
        return sorted(self.img_fnames) == sorted(self.seg_fnames)

    def _check_copies(self):
        return len(sorted(self.img_fnames)) == len(sorted(self.seg_fnames))
