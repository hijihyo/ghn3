# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluates a trained PyTorch model on ImageNet.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

Example:

    python eval.py -d imagenet -D $SLURM_TMPDIR --arch resnet50 --ckpt ./checkpoints/resnet50/checkpoint.pt

"""


import torch
import torchvision
import time
from ppuda.config import init_config
from ppuda.utils import infer, adjust_net
from ppuda.vision.loader import image_loader


args = init_config(mode='eval')
is_imagenet = args.dataset.startswith('imagenet')
print('loading the %s dataset...' % args.dataset)
val_loader, num_classes = image_loader(args.dataset,
                                       args.data_dir,
                                       test=True,
                                       test_batch_size=args.test_batch_size,
                                       num_workers=args.num_workers,
                                       noise=args.noise,
                                       im_size=299 if args.arch == 'inception_v3' else (224 if is_imagenet else 32),
                                       seed=args.seed)[1:]

model = eval(f'torchvision.models.{args.arch}(num_classes=num_classes)').to(args.device)
model = adjust_net(model, large_input=is_imagenet)
state_dict = torch.load(args.ckpt, map_location=args.device)
if 'state_dict' in state_dict:
    epoch, step, state_dict = state_dict['epoch'], state_dict['step'], state_dict['state_dict']
else:
    epoch = step = -1

model.load_state_dict(state_dict)
n_params = sum([p.numel() for p in model.parameters()]) / 10 ** 6
print('Model {} with {} parameters loaded from epoch {}, step {}.'.format(args.arch.upper(),
                                                                          n_params,
                                                                          epoch,
                                                                          step))

print('Running evaluation for {} with {:.2f}M parameters...'.format(args.arch.upper(), n_params))
if is_imagenet:
    val_loader.sampler.generator.manual_seed(args.seed)  # set the generator seed to reproduce results
start = time.time()
top1, top5 = infer(model.to(args.device), val_loader, verbose=False)
print('\ntesting: top1={:.3f}, top5={:.3f} ({} eval samples, time={:.2f} seconds)'.format(
    top1, top5, val_loader.dataset.num_examples, time.time() - start), flush=True)
