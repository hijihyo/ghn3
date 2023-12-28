# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains a Graph HyperNetwork (GHN-3) on DeepNets-1M and ImageNet. DistributedDataParallel (DDP) training is
used if `torchrun` is used as shown below.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

Example:

    # To train GHN-3-T/m8 on ImageNet (make sure to put the DeepNets-1M dataset in $SLURM_TMPDIR or in ./data) on
    # single GPU, automatic mixed precision:
    python train_ghn_ddp.py -d imagenet -D $SLURM_TMPDIR -n -v 50 --ln \
    -e 75 --opt adamw --lr 4e-4 --wd 1e-2 -b 128 --amp -m 8 --name ghn3tm8 --hid 64 --scheduler cosine-warmup

    # 4 GPUs (DDP), automatic mixed precision (as in the paper):
    export OMP_NUM_THREADS=8
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ghn_ddp.py -d imagenet -D $SLURM_TMPDIR -n -v 50 --ln \
    -e 75 --opt adamw --lr 4e-4 --wd 1e-2 -b 128 --amp -m 8 --name ghn3tm8 --hid 64 --scheduler cosine-warmup

    # Sometimes, there can be mysterious errors due to DDP (depending on the pytorch/cuda version).
    # So it can be a good idea to wrap this command in a for loop to continue training in case of failure.

    # To train GHN-3-T/m8 on CIFAR-10:
    python train_ghn_ddp.py -n -v 50 --ln -m 8 --name ghn3tm8-c10 --hid 64 --layers 3 --opt adamw --lr 4e-4 --wd 1e-2 \
     --scheduler cosine-warmup --amp

    # Use eval_ghn.py to evaluate the trained GHN-3 model on ImageNet/CIFAR-10.

"""


import argparse
import os
import time
from copy import deepcopy
from functools import partial

import torch
from ppuda.config import init_config
from ppuda.vision.imagenet import ImageNetDataset
from ppuda.vision.transforms import transforms_cifar as get_cifar_transforms
from ppuda.vision.transforms import transforms_imagenet as get_imagenet_transforms
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import wandb
from ghn3 import GHN3, DeepNets1MDDP, Trainer, clean_ddp, log, setup_ddp

log = partial(log, flush=True)


def get_train_image_dataloader(
    dataset,
    data_dir,
    batch_size,
    train_eval_transforms=None,
    add_eval_noise=False,
    image_size=32,
    use_cutout=False,
    cutout_length=16,
    num_workers=0,
    generator=None,
    verbose=True,
):
    lower_dataset = dataset.lower()
    assert lower_dataset in ["imagenet", "cifar10"]
    if lower_dataset == "imagenet":
        if train_eval_transforms is None:
            train_eval_transforms = get_imagenet_transforms(noise=add_eval_noise, im_size=image_size)
        train_transform, _ = train_eval_transforms

        dataset_dir = os.path.join(data_dir, "imagenet")
        train_dataset = ImageNetDataset(dataset_dir, split="train", transform=train_transform, has_validation=True)
        num_classes = len(train_dataset.classes)
    else:  # lower_dataset == "cifar10"
        if train_eval_transforms is None:
            train_eval_transforms = get_cifar_transforms(
                noise=add_eval_noise, sz=image_size, cutout=use_cutout, cutout_length=cutout_length
            )
        train_transform, _ = train_eval_transforms

        train_dataset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        num_all = len(train_dataset.targets)
        num_valid = num_all // 10
        train_indices, _ = torch.split(torch.arange(num_all), (num_all - num_valid, num_valid))
        train_dataset.data = train_dataset.data[train_indices]
        train_dataset.targets = [train_dataset.targets[i] for i in train_indices]

        train_dataset.checksum = train_dataset.data.mean()
        train_dataset.num_examples = len(train_dataset.targets)
        num_classes = len(torch.unique(torch.tensor(train_dataset.targets)))

    if verbose:
        print(
            f"Loaded {dataset} dataset (num_classes={num_classes}, num_examples={train_dataset.num_examples}, "
            f"checksum={train_dataset.checksum})"
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, generator=generator, pin_memory=True, num_workers=num_workers
    )
    return train_dataloader, num_classes


def is_synchronized(tensor, world_size, rank):
    tensor = tensor.to(rank)
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    _is_synchronized = all(torch.allclose(tensors[i], tensors[i + 1]) for i in range(world_size - 1))
    del tensors
    return _is_synchronized


def main():
    parser = argparse.ArgumentParser(description="GHN-3 training")
    parser.add_argument("--heads", type=int, default=8, help="number of self-attention heads in GHN-3")
    parser.add_argument("--compile", type=str, default=None, help="use pytorch2.0 compilation for potential speedup")
    parser.add_argument(
        "--ghn2",
        action="store_true",
        help="train GHN-2, also can use code from" " https://github.com/facebookresearch/ppuda to train GHN-2",
    )
    parser.add_argument("--interm_epoch", type=int, default=5, help="intermediate epochs to keep checkpoints for")

    parser.add_argument("--wandb", action="store_true", default=False, help="Enable logging at Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="default", help="The name of a project in Weights & Biases")
    parser.add_argument(
        "--wandb_entity", type=str, default="mlai-nngen", help="The name of an entity in Weights & Biases"
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None, help="The name of an experiment in Weights & Biases. Defaults to None"
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="Short description about the experiment. Recorded in Weights & Biases",
    )

    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(
        mode="train_ghn",
        parser=parser,
        verbose=ddp.rank == 0,
        debug=0,  # to avoid extra sanity checks and make training faster
        layers=3,  # default number of layers in GHN-3
        shape_multiplier=2 if ghn2 else 1,
    )  # max_shape default setting (can be overriden by --max_shape)

    if args.wandb_name is None:
        args.wandb_name = args.name

    if hasattr(args, "multigpu") and args.multigpu:
        raise NotImplementedError(
            "the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. "
            "nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel "
            "(https://github.com/pytorch/pytorch/issues/659360)."
            "Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. "
            "nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top)."
        )

    use_wandb = args.wandb and ddp.rank == 0
    if use_wandb:
        vars_args = vars(deepcopy(args))
        del vars_args["wandb"]
        del vars_args["wandb_project"]
        del vars_args["wandb_entity"]
        del vars_args["wandb_name"]
        del vars_args["wandb_notes"]
        wandb.init(
            config=vars_args,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            notes=args.wandb_notes,
        )

    is_imagenet = args.dataset.startswith("imagenet")
    log("loading the %s dataset..." % args.dataset.upper())
    generator = torch.Generator().manual_seed(42)
    train_queue, num_classes = get_train_image_dataloader(
        args.dataset,
        args.data_dir,
        args.batch_size,
        image_size=args.imsize,
        num_workers=args.num_workers,
        generator=generator,
        verbose=(ddp.rank == 0),
    )

    hid = args.hid
    s = 16 if is_imagenet else 11
    default_max_shape = (hid * 2, hid * 2, s, s) if ghn2 else (hid, hid, s, s)
    log(
        "current max_shape: {} {} default max_shape: {}".format(
            args.max_shape, "=" if args.max_shape == default_max_shape else "!=", default_max_shape
        )
    )

    config = {
        "max_shape": args.max_shape,
        "num_classes": num_classes,
        "hypernet": args.hypernet,
        "decoder": args.decoder,
        "weight_norm": args.weight_norm,
        "ve": args.virtual_edges > 1,
        "layernorm": args.ln,
        "hid": hid,
        "layers": args.layers,
        "heads": args.heads,
        "is_ghn2": ghn2,
    }

    ghn = GHN3(**config, debug_level=args.debug)
    graphs_queue, sampler = DeepNets1MDDP.loader(
        args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
        dense=ghn.is_dense(),
        wider_nets=is_imagenet,
        split=args.split,
        nets_dir=args.data_dir,
        virtual_edges=args.virtual_edges,
        num_nets=args.num_nets,
        large_images=is_imagenet,
        verbose=ddp.rank == 0,
        debug=args.debug > 0,
    )

    trainer = Trainer(
        ghn,
        opt=args.opt,
        opt_args={"lr": args.lr, "weight_decay": args.wd, "momentum": args.momentum},
        scheduler="mstep" if args.scheduler is None else args.scheduler,
        scheduler_args={"milestones": args.lr_steps, "gamma": args.gamma},
        n_batches=len(train_queue),
        grad_clip=args.grad_clip,
        device=args.device,
        log_interval=args.log_interval,
        amp=args.amp,
        amp_min_scale=1024,  # this helped stabilize AMP training
        amp_growth_interval=100,  # this helped stabilize AMP training
        predparam_wd=0 if ghn2 else 3e-5,
        label_smoothing=0.1 if is_imagenet else 0.0,
        save_dir=args.save,
        ckpt=args.ckpt,
        epochs=args.epochs,
        verbose=ddp.rank == 0,
        compile_mode=args.compile,
    )

    log("\nStarting training GHN with {} parameters!".format(sum([p.numel() for p in ghn.parameters()])))
    if ddp.ddp:
        # make sure sample order is different for each seed
        sampler.sampler.seed = args.seed
        log(f"shuffle DeepNets1MDDP train loader: set seed to {args.seed}")
        # for each DeepNets1MDDP epoch, the graph loader will be shuffled inside the ghn3/deepnets1m.py

    graphs_queue = iter(graphs_queue)

    for epoch in range(trainer.start_epoch, args.epochs):
        log("\nepoch={:03d}/{:03d}, lr={:e}".format(epoch + 1, args.epochs, trainer.get_lr()))

        trainer.reset_metrics(epoch)

        for step, (images, targets) in enumerate(train_queue, start=trainer.start_step):
            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break

            if ddp.ddp and step % 100 == 0:
                if is_synchronized(images, ddp.world_size, ddp.rank) and is_synchronized(
                    targets, ddp.world_size, ddp.rank
                ):
                    log(f"DDP: Training batches are being synchronized (step={step}, world_size={ddp.world_size})")
                else:
                    raise RuntimeError(
                        f"DDP: Training batches are not synchronized (step={step}, world_size={ddp.world_size})"
                    )

            trainer.update(images, targets, graphs=next(graphs_queue))
            metrics = trainer.log(step)
            if metrics is not None and use_wandb:
                _step = epoch * len(train_queue) + step + 1
                wandb.log(metrics, _step)

            if args.save:
                # save GHN checkpoint
                trainer.save(epoch, step, {"args": args, "config": config}, interm_epoch=args.interm_epoch)

        trainer.scheduler_step()  # lr scheduler step

    log("done at {}!".format(time.strftime("%Y%m%d-%H%M%S")))

    if use_wandb:
        wandb.finish()

    if ddp.ddp:
        clean_ddp()


if __name__ == "__main__":
    main()
