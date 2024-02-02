# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import wandb
import datetime
from pathlib import Path
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

from engine_pretrain import train_one_epoch
import models.fcmae as fcmae

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler, dedense_checkpoint_keys
from utils import str2bool

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--update_freq', default=8, type=int,
                        help='gradient accumulation step')
    parser.add_argument('--patch_size', default=32, type=int,
                        help='Patch size')
    parser.add_argument('--pretraining', default=None, type=str,
                    help='Per GPU batch size')
    parser.add_argument('--sigmoid', default=False, type=bool,
                    help='use sigmoid')
    parser.add_argument('--use_fpn', default=False, type=bool,
                    help='use fpn')
    # Model parameters
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.025,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def main(args):
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3,ratio=(1,1)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale()
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    torch.cuda.set_device(device)
    model = fcmae.__dict__[args.model](
        mask_ratio=args.mask_ratio,
        img_size=args.input_size,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        norm_pix_loss=args.norm_pix_loss,
        patch_size=args.patch_size,
        sigmoid = args.sigmoid,
        use_fpn = args.use_fpn
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)


    args.output_dir = f'{args.output_dir}/img_size_{args.input_size}_lr_{args.blr}_mask_ammount_{args.mask_ratio}_sigmoid_{args.sigmoid}_pretraining_{args.pretraining}'
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)

    run = wandb.init(
        project="sem-segmentation",
        mode="online",
        group="DDP75",
        config={
            "input_size": args.input_size,
            "model": args.model,
            "batch_size": args.batch_size,
            "blr": args.blr,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "data_path": args.data_path,     
            "output_dir": args.output_dir,
            "mask_ratio": args.mask_ratio,
            "patch_size": args.patch_size,
            "sigmoid": args.sigmoid,
            "pretraining": args.pretraining
        }
    )
    run.name = f'img_size_{args.input_size}_lr_{args.blr}_mask_ammount_{args.mask_ratio}_sigmoid_{args.sigmoid}_pretraining_{args.pretraining}'
    run.save()
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)
    
    
    if args.pretraining == 'ssl':
        preloaded_weights = torch.load('/home/hk-project-test-dl4pm/hgf_xda8301/ConvNeXt-V2/ssl_pretrain/pretrained_ssl.pt')
        #preloaded_weights = torch.load('/home/ws/kg2371/projects/ConvNeXt-V2/pretrained_online/pretrained_ssl.pt')
        preloaded_weights = dedense_checkpoint_keys(preloaded_weights['model'])
        preloaded_weights['encoder.downsample_layers.0.0.weight'] = torch.mean(preloaded_weights['encoder.downsample_layers.0.0.weight'],dim=1).unsqueeze(1)
        model.load_state_dict(preloaded_weights,strict=False)
    if args.pretraining == 'ft':
        preloaded_weights = torch.load('/home/ws/kg2371/projects/ConvNeXt-V2/pretrained_online/pretrained_ft.pt')
        preloaded_weights = dedense_checkpoint_keys(preloaded_weights['model'])
        preloaded_weights['encoder.downsample_layers.0.0.weight'] = torch.mean(preloaded_weights['encoder.downsample_layers.0.0.weight'],dim=1).unsqueeze(1)
        model.load_state_dict(preloaded_weights,strict=False)     

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            if (epoch + 1) % 250 == 0:
                # save every 250 epochs
                utils.save_model_intermediate(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        wandb.log({"epoch": epoch, "n_parameters": n_parameters})
        wandb.log({f'train_{k}': v for k, v in train_stats.items()})
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()


if __name__ == '__main__':
    
    # distributeeeed
    distributed=False
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        distributed = world_size > 1
        ngpus_per_node = torch.cuda.device_count()
    if distributed:
        rank = int(os.environ['SLURM_PROCID'])
        print('device count')
        print(torch.cuda.device_count())
        gpu = rank % torch.cuda.device_count()
        dist_backend = 'nccl'
        dist_url = 'env://'
        dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.distributed = distributed
    args.gpu = gpu
    args.device = gpu
    main(args)