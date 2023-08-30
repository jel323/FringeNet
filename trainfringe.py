import argparse
import sys
import os
import torch
import numpy as np
from TrainingFunctions import model

path = sys.path[0]
print(f"Current Path - {path}")
bpath = path[0 : path.rfind(os.path.sep)]
from TrainingFunctions import trainer as tr


def main(args: argparse.Namespace):
    if args.filename == "bigcutbright":
        img_size = (400, 600)
    elif args.filename == "bigcutbrights":
        img_size = (200, 250)
    args.n_layers = np.array(args.n_layers, dtype=np.int32)
    args.channel_mult = np.array(args.channel_mult, dtype=np.int32)
    args.kernel_size = np.array(args.kernel_size, dtype=np.int32)
    args.nconvs = np.array(args.nconvs, dtype=np.int32)
    args.nrepititions = np.array(args.nrepititions, dtype=np.int32)
    trainer = tr.TrainFringe(args)

    trainer.make_datasets()
    trainer.set_model(
        model.NUNet(
            n_unets=args.n_unets,
            n_channels=3,
            n_classes=1,
            n_layers=args.n_layers,
            mult=args.channel_mult,
            init_kernel_size=args.kernel_size,
            nconvs=args.nconvs,
            nrepititions=args.nrepititions,
            bn=args.bn,
            recurrent=args.recurrent,
            recurrent_mid=args.recurrent_mid,
            dropout=args.dropout,
            img_size=img_size,
        ).to(trainer.get_device())
    )
    train_loss_lst, test_loss_lst = trainer.initialize_model()

    trainer.set_loss_fn()
    trainer.order_crops()

    # Train and test loop
    for epoch in range(trainer.start_epoch, trainer.start_epoch + args.num_epochs):
        train_loss = trainer.train_epoch(epoch)
        test_loss = trainer.test_epoch()
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)

        msave = not bool(epoch % args.save_freq)
        trainer.save_models(epoch, (train_loss_lst, test_loss_lst), msave)
        trainer.save_images(epoch)
        trainer.save_losses((train_loss_lst, test_loss_lst))
    return


@torch.no_grad()
def getcudainfo(device: torch.device) -> None:
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print(f"Allocated - {round(torch.cuda.memory_allocated(0)/1024**3,1)}GB")
        print(f"Cached - {round(torch.cuda.memory_reserved(0)/1024**3,1)}GB")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def str2bool(s: str) -> bool:
        return s.lower().startswith("t")

    # ------------------------Must-Use----------------------------------------
    parser.add_argument(
        "-filename",
        default=None,
        type=str,
        help="Name of file to get imgs from",
    )
    parser.add_argument(
        "-snameadd", type=str, default="", help="Added to the end of the save file"
    )
    parser.add_argument("-batch_size", default=1, type=int, help="Batch size per GPU")
    parser.add_argument("-lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "-transformations",
        default=False,
        type=str2bool,
        help="Wether or not to apply image transformations during training epochs",
    )
    parser.add_argument(
        "-n_unets",
        default=2,
        type=int,
        help="Number of sequential unets",
    )

    # ------------------------Should/Can-Use----------------------------------
    parser.add_argument(
        "-seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-use_npy",
        default=True,
        type=str2bool,
        help="Use the npy dataset (used when images are saved as a "
        "npy file) instead of get one dataset",
    )
    parser.add_argument(
        "-num_img_plots",
        default=10,
        type=int,
        help="Number of Images plots made and stored in google drive",
    )
    parser.add_argument(
        "-save_freq", default=10, type=int, help="Frequency of epochs to save"
    )
    parser.add_argument(
        "-bn", default=True, type=str2bool, help="Use batch normalization"
    )
    parser.add_argument(
        "-channel_mult",
        default=[64],
        type=int,
        nargs="+",
        help="Base multiplier for channels",
    )
    parser.add_argument(
        "-num_epochs", default=50, type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "-testing_samples",
        default=100,
        type=int,
        help="Number of samples in test_set",
    )
    parser.add_argument(
        "-resume", type=str2bool, default=False, help="Resume from checkpoint"
    )
    parser.add_argument(
        "-resume_epoch",
        type=str2bool,
        default=False,
        help="Resume from a specific epoch",
    )
    parser.add_argument(
        "-resume_epoch_n",
        type=int,
        default=0,
        help="Epoch number to resume on if resuming from epoch",
    )

    # ------------------------Should-Not-Use----------------------------------
    parser.add_argument(
        "-save_ext",
        type=str,
        default=".pt",
    )
    parser.add_argument(
        "-true_img_name",
        type=str2bool,
        default=True,
        help="True if the images in the dataset have their"
        "true interferometry number as name, usually"
        "false if you have randomized the dataset order",
    )
    parser.add_argument(
        "-n_layers",
        default=[4],
        type=int,
        nargs="+",
        help="Number of conv layers in model",
    )
    parser.add_argument(
        "-kernel_size",
        default=[3],
        type=int,
        nargs="+",
        help="Kernel size in initial convolution operator",
    )
    parser.add_argument(
        "-nconvs",
        default=[2],
        type=int,
        nargs="+",
        help="Number of Convolutions in Convolution Block",
    )
    parser.add_argument(
        "-nrepititions",
        default=[2],
        type=int,
        nargs="+",
        help="Number of repititions in recurrent block",
    )
    parser.add_argument(
        "-recurrent",
        default=False,
        type=str2bool,
        help="Wether or not to use recurrent convolutions",
    )
    parser.add_argument(
        "-recurrent_mid",
        default=False,
        type=str2bool,
        help="Wether to use recurrent block in middle layer",
    )
    parser.add_argument(
        "-attention",
        default=False,
        type=str2bool,
        help="Wether or not to use attention block",
    )
    parser.add_argument(
        "-dropout", default=False, type=str2bool, help="Use dropout operation"
    )
    parser.add_argument("-gpu_ids", default=[0], type=eval, help="IDs of GPUs to use")
    parser.add_argument(
        "-num_workers",
        default=4,
        type=int,
        help="Number of data loader threads",
    )
    parser.add_argument(
        "-resume_best",
        type=str2bool,
        default=False,
        help="Resume from best checkpoint",
    )
    parser.add_argument(
        "-annealingRate",
        default=1e-5,
        type=float,
        help="Annealing rate per epoch",
    )
    parser.add_argument(
        "-weight_decay", default=0.0, type=float, help="L2 Weight Decay"
    )

    main(parser.parse_args())
