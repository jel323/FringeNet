import argparse
import numpy as np
import sys
import os
import random
import torch as torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import torchvision
import util
from tqdm import tqdm
import filelist
from PIL import Image
import lossfuns
import dataset as dset
import model

path = sys.path[0]
bpath = path[: path.rfind(os.path.sep)]


class TrainFringe:
    @torch.no_grad()
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Set up main device and scale batch size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and args.gpu_ids
            else torch.device("cpu")
        )

        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # Making filenames and directories
        self.fname1 = args.filename
        self.check_filename()
        print(f"Images Filename - {self.fname1}")
        self.data_path = os.path.join(bpath, "Data", "interferometry")
        self.fname2 = self.fname1 + args.snameadd
        print(f"Save Filename - {self.fname2}")
        if not self.args.resume:
            self.maketxt()
        return

    @torch.no_grad()
    def check_filename(self) -> None:
        """
        Checks that a filename was given and throws exception if not.
        """
        if self.fname1 == None:
            raise Exception("No Filename Given, use -fname 'filename'")
        return

    @torch.no_grad()
    def make_datasets(self) -> None:
        """
        Creates the training/testing datasets and dataloaders using the
        pytorch dataset and dataloader abstract classes. Also runs
        some tests on the datasets, and prints image shapes.
        """
        self.train_dataset = dset.ImgDatasetStore3d(
            os.path.join(self.data_path, "img"),
            self.fname1,
            testing_samples=self.args.testing_samples,
            pad=False,
        )
        self.test_dataset = dset.ImgDatasetStore3d(
            os.path.join(self.data_path, "img"),
            self.fname1,
            testing_samples=self.args.testing_samples,
            pad=False,
            test_set=True,
        )
        self.train_loader = data.DataLoader(
            self.train_datasetset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        self.test_loader = data.DataLoader(
            self.test_datasettestset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        self.train_dataset.test()
        self.test_dataset.test()
        self.img_shape_before = self.train_dataset._get_shape()
        self.img_shape_after = self.test_dataset._get_out_shape()
        print(f"Image Shape - {self.img_shape_before}")
        print(f"Image Shape With Padding - {self.img_shape_after}")
        return

    @torch.no_grad()
    def make_model(self) -> None:
        """
        Makes the UNet model and Adam to be trained via the
        hyper-parameters given through the file arguments
        (like -filename). Also prints the model parameter summary.
        """
        self.net = model.UNet(
            3,
            1,
            n_layers=self.args.n_layers,
            init_kernel_size=self.args.kernel_size,
            nconvs=self.args.nconvs,
            nrepititions=self.args.nrepititions,
            bn=self.args.bn,
            recurrent=self.args.recurrent,
            recurrent_mid=self.args.recurrent_mid,
            dropout=self.args.dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        print("Model Type - " + self.net.name())
        model.count_params(self.net)
        return

    @torch.no_grad()
    def initialize_model(self) -> None:
        """
        Initializes the model parameters, either from scratch if training is
        not being resumed, or from a previous save if training is being
        resumed.
        """
        self.start_epoch = 0
        self.best_loss = float("inf")
        if self.args.resume:
            if self.args.resume_best:
                resume_str = os.path.join(
                    self.data_path, "saves", self.fname2, "best_pth.tar"
                )
            if self.args.resume_epoch:
                resume_str = os.path.join(
                    self.data_path,
                    "saves",
                    self.fname2,
                    "all",
                    str(self.args.resume_epoch_n) + "_pth.tar",
                )
            else:
                resume_str = os.path.join(
                    self.data_path, "saves", self.fname2, "recent_pth.tar"
                )
            print("Resuming from checkpoint at " + resume_str)
            checkpoint = torch.load(resume_str)
            self.net.load_state_dict(checkpoint["net"])
            self.optimizer.load_state_dict(checkpoint["net_opt"])
            self.best_loss = checkpoint["test_loss"]
            self.start_epoch = checkpoint["epoch"] + 1
        return

    @torch.no_grad()
    def set_loss_fn(self) -> None:
        """
        Creates the loss function. (Dice Loss + Binary Cross-Entropy Loss)
        """
        self.loss_fn = lossfuns.DiceBCELoss().to(self.device)
        return

    @staticmethod
    @torch.no_grad()
    def _getcrops(img_shape):
        """
        Creates the image crops to apply to segmentations to remove
        unwanted boundaries.
        """
        if img_shape[0] > img_shape[1]:
            croph = torchvision.transforms.CenterCrop(size=img_shape)
            cropw = torchvision.transforms.CenterCrop(size=(img_shape[1], img_shape[0]))
        else:
            cropw = torchvision.transforms.CenterCrop(size=img_shape)
            croph = torchvision.transforms.CenterCrop(size=(img_shape[1], img_shape[0]))
        return croph, cropw

    @torch.no_grad()
    def order_crops(self) -> None:
        """
        Adds the correct crop to the instance.
        """
        croph, cropw = TrainFringe._getcrops(self.img_shape_before)
        self.crop = cropw
        return

    @torch.no_grad()
    def get_img_picks(self, nbatches: int) -> list[int]:
        """
        Gets the random picks for saving image, model segmentation,
        true segmentation triples, these random picks are used to
        select random triples in the dataset.
        """
        picks = [0, nbatches - 1] + list(
            np.random.randint(1, nbatches - 2, size=(self.args.num_img_plots - 2))
        )
        sa = set(picks)
        l = len(picks)
        while len(sa) != l:
            picks.append(random.randint(0, nbatches - 2))
            sa = set(picks)
        return picks

    @torch.enable_grad()
    def train_batch(
        self, imgs: torch.Tensor, segs: torch.Tensor
    ) -> tuple[torch.Tensor, float, int]:
        """
        Trains the model on a single batch of images.
        """
        self.net.train()
        self.net.zero_grad()
        self.optimizer.zero_grad()
        b_size = imgs.size(0)
        m_segs, segs = self.crop(self.net(imgs)), self.crop(segs)
        loss = self.loss_fn(m_segs, segs)
        loss_out = loss.item()
        loss.backward()
        self.optimizer.step()
        del loss
        torch.cuda.empty_cache()
        return m_segs, loss_out, b_size

    @torch.enable_grad()
    def train_epoch(self, epoch: int) -> float:
        """
        Trains the model on an epoch, or the entire
        training dataset
        """
        print(f"\nEpoch: {epoch}")
        loss_meter = util.AverageMeter()
        self.train_triples = []
        picks = self.get_img_picks(len(self.train_loader))
        s = 0
        with tqdm(total=len(self.train_dataset)) as progress_bar:
            for imgs, segs in self.train_loader:
                m_segs, loss, b_size = self.train_batch(imgs, segs)
                segs = self.crop(segs)
                loss_meter.update(loss, b_size)

                progress_bar.set_postfix(
                    {
                        "Model Loss": loss_meter.avg,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                progress_bar.update(b_size)

                if s in picks:
                    self.train_triples.append(
                        [
                            imgs[0, :, :, :].cpu(),
                            m_segs[0, :, :, :].cpu(),
                            segs[0, :, :, :].cpu(),
                        ]
                    )
                del imgs
                del segs
                del m_segs
                torch.cuda.empty_cache()
                s += 1
        return loss_meter.avg

    @torch.no_grad()
    def test_batch(
        self, imgs: torch.Tensor, segs: torch.Tensor
    ) -> tuple[torch.Tensor, float, int]:
        """
        Tests the model on a batch of images.
        """
        self.net.eval()
        b_size = imgs.size(0)
        m_segs, segs = self.crop(self.net(imgs)), self.crop(segs)
        loss = self.loss_fn(m_segs, segs)
        loss_out = loss.item()
        del loss
        torch.cuda.empty_cache()
        return m_segs, loss_out, b_size

    @torch.no_grad()
    def test_epoch(self) -> float:
        """
        Tests the model on an epoch, or the entire testing dataset.
        """
        loss_meter = util.AverageMeter()
        self.test_triples = []
        picks = self.get_img_picks(len(self.test_loader))
        s = 0
        with tqdm(total=len(self.test_dataset)) as progress_bar:
            for imgs, segs in self.test_loader:
                m_segs, loss, b_size = self.test_batch(imgs, segs)
                segs = self.crop(segs)
                loss_meter.update(loss, b_size)

                progress_bar.set_postfix(
                    {
                        "Model Loss": loss_meter.avg,
                    }
                )
                progress_bar.update(b_size)
                if s in picks:
                    self.test_triples.append(
                        (
                            imgs[0, :, :, :].cpu(),
                            m_segs[0, :, :, :].cpu(),
                            segs[0, :, :, :].cpu(),
                        )
                    )
                del imgs
                del m_segs
                del segs
                torch.cuda.empty_cache()
                s += 1
        return loss_meter.avg

    @torch.no_grad()
    def save_models(
        self, epoch: int, losses: tuple[list[float], list[float]], msave: bool
    ) -> None:
        """
        Saves the current model. Current model always saved as recent model,
        and depending on saving parameters, by epoch #.
        """
        state = {
            "net": self.net.state_dict(),
            "net_opt": self.optimizer.state_dict(),
            "test_loss": self.best_loss,
            "prev_losses": losses,
            "epoch": epoch,
        }
        save_pth = os.path.join(self.data_path, "saves", self.fname2)
        os.makedirs(save_pth, exist_ok=True)
        print("Saving")
        if losses[1][-1] < self.best_loss:
            self.best_loss = losses[1][-1]
            print("Saving Best ...")
            state["test_loss"] = self.best_loss
            torch.save(state, os.path.join(save_pth, "best_pth.tar"))
        torch.save(state, os.path.join(save_pth, "recent_pth.tar"))
        if msave:
            os.makedirs(os.path.join(save_pth, "all"), exist_ok=True)
            torch.save(state, os.path.join(save_pth, "all", str(epoch) + "_pth.tar"))
        return

    @staticmethod
    @torch.no_grad()
    def _transimg(img: np.ndarray) -> np.ndarray:
        """
        Helper function for converting from normalized np.float32 to np.uint8.
        """
        img = img * 255
        return img.astype(np.uint8)

    @staticmethod
    @torch.no_grad()
    def _saveImages(
        epoch: int,
        n: int,
        imgs: torch.Tensor,
        set_name: str,
        dir_path: os.PathLike,
        fname: str,
    ) -> None:
        """
        Helper function to handle image saves of a image, model segmentation,
        true segmentation triple.
        """
        (img, out, seg) = imgs
        img = img[50:-50, 50:-50, :]
        out = out[0, :, :]
        out = torch.tensor(out).numpy()
        out_th = out > 0.5
        seg = seg[0, :, :]
        img = Image.fromarray(TrainFringe._transimg(img))
        out = Image.fromarray(TrainFringe._transimg(out))
        out_th = Image.fromarray(TrainFringe._transimg(out_th))
        seg = Image.fromarray(TrainFringe._transimg(seg))
        dir_path = os.path.join(dir_path, fname, str(epoch))
        os.makedirs(dir_path, exist_ok=True)
        save_fname = "_" + set_name + "_" + str(n) + ".png"
        img.save(os.path.join(dir_path, "img" + save_fname))
        out.save(os.path.join(dir_path, "out" + save_fname))
        out_th.save(os.path.join(dir_path, "out_th" + save_fname))
        seg.save(os.path.join(dir_path, "seg" + save_fname))
        return

    @staticmethod
    @torch.no_grad()
    def _open_triples(
        triples: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function for converting image, model segmentation,
        true segmentation triples from torch.Tensors to np.ndarrays
        """
        imgs, outs, segs = triples
        imgs, outs, segs = (
            imgs.permute(1, 2, 0).numpy(),
            outs.numpy(),
            segs.numpy(),
        )
        return (imgs, outs, segs)

    @torch.no_grad()
    def save_images(self, epoch: int) -> None:
        """
        Saves the partially randomly chosen image, model segmentation,
        true segmentation triples for an epoch.
        """
        dir_path = os.path.join(self.data_path, "saves", "imagesaves")
        n = len(self.train_triples)
        for k in range(n):
            TrainFringe._saveImages(
                epoch,
                k,
                TrainFringe._open_triples(self.train_triples[k]),
                "train",
                dir_path,
                self.fname2,
            )
            TrainFringe._saveImages(
                epoch,
                k,
                TrainFringe._open_pairs(self.test_triples[k]),
                "test",
                dir_path,
                self.fname2,
            )
        return

    @staticmethod
    @torch.no_grad()
    def _saveLoss(loss_lst: list[float], dir_path: os.PathLike, set_name: str) -> None:
        """
        Saves a given loss list in the directory given. Helper function
        for save_losses.
        """
        save_fname = set_name + "_loss"
        loss = np.array(loss_lst)
        np.save(os.path.join(dir_path, save_fname), loss)
        return

    @torch.no_grad()
    def save_losses(self, losses: tuple[list[float], list[float]]) -> None:
        """
        Saves the list of training and testing losses for each epoch
        (0 to current).
        """
        dir_path = os.path.join(self.data_path, "saves", self.fname2)
        os.makedirs(dir_path, exist_ok=True)
        TrainFringe._saveLoss(losses[0], dir_path, "train_m")
        TrainFringe._saveLoss(losses[1], dir_path, "test_m")
        return

    @torch.no_grad()
    def maketxt(self):
        """
        Makes the txt file that describes the current run, this is only
        done if creating a new run, not resuming previous.
        """
        disc = input("Input Discription of This Run:\n>")
        dic = {
            "Model Type": "1D GAN",
            "Model Name": self.fname2,
            "Dataset": self.fname1,
            "Number of Layers": str(self.args.n_layers),
            "Learning Rate": str(self.args.lr),
            "Batch Size": str(self.args.batch_size),
            "Testing Samples": str(self.args.testing_samples),
        }
        f = open(os.path.join("ONEDGan", "saves", self.args.save_dir, "info.txt"), "w")
        for k in dic.keys():
            f.write(k + " - " + dic[k] + "\n")

        def parse(disc):
            words = disc.split(" ")
            lines = []
            lines.append([])
            s = 0
            l = 0
            maxl = 50
            for k in words:
                l += len(k) + 1
                if l < maxl:
                    lines[s].append(k)
                else:
                    lines.append([])
                    s += 1
                    l = len(k) + 1
                    lines[s].append(k)
            strings = []
            for k in range(len(lines)):
                string = ""
                i = -1
                for i in range(len(lines[k]) - 1):
                    string += lines[k][i] + " "
                string += lines[k][i + 1]
                strings.append(string)
            return strings

        lines = parse(disc)
        f.write("\n")
        f.write("Description:\n")
        for k in lines:
            f.write(k + "\n")
        fnames = filelist.fnames("Data/interferometry/img/raw" + self.fname1)
        f.write("\n")
        f.write("Images Used:\n")
        for k in fnames:
            f.write(k + "\n")
        f.close()
        return
