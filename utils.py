# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf

from tensorboardX import SummaryWriter
from collections import OrderedDict

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        try:
            x_reduce = torch.tensor(x).cuda()
            dist.all_reduce(x_reduce)
            x_reduce /= world_size
            return x_reduce.item()
        except:
            print('REDUCE DID NOT WORK HUH')
            return 1000
    else:
        return x

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def save_model_intermediate(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(f'{args.output_dir}/intermediate')
    Path(f'{args.output_dir}/intermediate').mkdir(parents=True, exist_ok=True)
    epoch_name = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    if model_ema is not None:
        to_save['model_ema'] = get_state_dict(model_ema)
    save_on_master(to_save, checkpoint_path)

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)
    
    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



depthwise_convs = [
    "stages.0.0.dwconv.weight",
    "stages.0.1.dwconv.weight",
    "stages.0.2.dwconv.weight",
    "stages.1.0.dwconv.weight",
    "stages.1.1.dwconv.weight",
    "stages.1.2.dwconv.weight",
    "stages.2.0.dwconv.weight",
    "stages.2.1.dwconv.weight",
    "stages.2.2.dwconv.weight",
    "stages.2.3.dwconv.weight",
    "stages.2.4.dwconv.weight",
    "stages.2.5.dwconv.weight",
    "stages.2.6.dwconv.weight",
    "stages.2.7.dwconv.weight",
    "stages.2.8.dwconv.weight",
    "stages.2.9.dwconv.weight",
    "stages.2.10.dwconv.weight",
    "stages.2.11.dwconv.weight",
    "stages.2.12.dwconv.weight",
    "stages.2.13.dwconv.weight",
    "stages.2.14.dwconv.weight",
    "stages.2.15.dwconv.weight",
    "stages.2.16.dwconv.weight",
    "stages.2.17.dwconv.weight",
    "stages.2.18.dwconv.weight",
    "stages.2.19.dwconv.weight",
    "stages.2.20.dwconv.weight",
    "stages.2.21.dwconv.weight",
    "stages.2.22.dwconv.weight",
    "stages.2.23.dwconv.weight",
    "stages.2.24.dwconv.weight",
    "stages.2.25.dwconv.weight",
    "stages.2.26.dwconv.weight",
    "stages.3.0.dwconv.weight",
    "stages.3.1.dwconv.weight",
    "stages.3.2.dwconv.weight"
]

standard_convs = [
    "downsample_layers.1.1.weight",
    "downsample_layers.2.1.weight",
    "downsample_layers.3.1.weight",

]

mappings = {
    "downsample_layers.0.0.weight": "encoder.downsample_layers.0.0.weight",
    "downsample_layers.0.0.bias": "encoder.downsample_layers.0.0.bias",
    "downsample_layers.0.1.weight": "encoder.downsample_layers.0.1.weight",
    "downsample_layers.0.1.bias": "encoder.downsample_layers.0.1.bias",
    "downsample_layers.1.0.weight": "encoder.downsample_layers.1.0.ln.weight",
    "downsample_layers.1.0.bias": "encoder.downsample_layers.1.0.ln.bias",
    "downsample_layers.1.1.weight": "encoder.downsample_layers.1.1.kernel",
    "downsample_layers.1.1.bias": "encoder.downsample_layers.1.1.bias",
    "downsample_layers.2.0.weight": "encoder.downsample_layers.2.0.ln.weight",
    "downsample_layers.2.0.bias": "encoder.downsample_layers.2.0.ln.bias",
    "downsample_layers.2.1.weight": "encoder.downsample_layers.2.1.kernel",
    "downsample_layers.2.1.bias": "encoder.downsample_layers.2.1.bias",
    "downsample_layers.3.0.weight": "encoder.downsample_layers.3.0.ln.weight",
    "downsample_layers.3.0.bias": "encoder.downsample_layers.3.0.ln.bias",
    "downsample_layers.3.1.weight": "encoder.downsample_layers.3.1.kernel",
    "downsample_layers.3.1.bias": "encoder.downsample_layers.3.1.bias",
    "stages.0.0.dwconv.weight": "encoder.stages.0.0.dwconv.kernel",
    "stages.0.0.dwconv.bias": "encoder.stages.0.0.dwconv.bias",
    "stages.0.0.norm.weight": "encoder.stages.0.0.norm.ln.weight",
    "stages.0.0.norm.bias": "encoder.stages.0.0.norm.ln.bias",
    "stages.0.0.pwconv1.weight": "encoder.stages.0.0.pwconv1.linear.weight",
    "stages.0.0.pwconv1.bias": "encoder.stages.0.0.pwconv1.linear.bias",
    "stages.0.0.pwconv2.weight": "encoder.stages.0.0.pwconv2.linear.weight",
    "stages.0.0.pwconv2.bias": "encoder.stages.0.0.pwconv2.linear.bias",
    "stages.0.0.grn.gamma": "encoder.stages.0.0.grn.gamma",
    "stages.0.0.grn.beta": "encoder.stages.0.0.grn.beta",
    "stages.0.1.dwconv.weight": "encoder.stages.0.1.dwconv.kernel",
    "stages.0.1.dwconv.bias": "encoder.stages.0.1.dwconv.bias",
    "stages.0.1.norm.weight": "encoder.stages.0.1.norm.ln.weight",
    "stages.0.1.norm.bias": "encoder.stages.0.1.norm.ln.bias",
    "stages.0.1.pwconv1.weight": "encoder.stages.0.1.pwconv1.linear.weight",
    "stages.0.1.pwconv1.bias": "encoder.stages.0.1.pwconv1.linear.bias",
    "stages.0.1.pwconv2.weight": "encoder.stages.0.1.pwconv2.linear.weight",
    "stages.0.1.pwconv2.bias": "encoder.stages.0.1.pwconv2.linear.bias",
    "stages.0.1.grn.gamma": "encoder.stages.0.1.grn.gamma",
    "stages.0.1.grn.beta": "encoder.stages.0.1.grn.beta",
    "stages.0.2.dwconv.weight": "encoder.stages.0.2.dwconv.kernel",
    "stages.0.2.dwconv.bias": "encoder.stages.0.2.dwconv.bias",
    "stages.0.2.norm.weight": "encoder.stages.0.2.norm.ln.weight",
    "stages.0.2.norm.bias": "encoder.stages.0.2.norm.ln.bias",
    "stages.0.2.pwconv1.weight": "encoder.stages.0.2.pwconv1.linear.weight",
    "stages.0.2.pwconv1.bias": "encoder.stages.0.2.pwconv1.linear.bias",
    "stages.0.2.pwconv2.weight": "encoder.stages.0.2.pwconv2.linear.weight",
    "stages.0.2.pwconv2.bias": "encoder.stages.0.2.pwconv2.linear.bias",
    "stages.0.2.grn.gamma": "encoder.stages.0.2.grn.gamma",
    "stages.0.2.grn.beta": "encoder.stages.0.2.grn.beta",
    "stages.1.0.dwconv.weight": "encoder.stages.1.0.dwconv.kernel",
    "stages.1.0.dwconv.bias": "encoder.stages.1.0.dwconv.bias",
    "stages.1.0.norm.weight": "encoder.stages.1.0.norm.ln.weight",
    "stages.1.0.norm.bias": "encoder.stages.1.0.norm.ln.bias",
    "stages.1.0.pwconv1.weight": "encoder.stages.1.0.pwconv1.linear.weight",
    "stages.1.0.pwconv1.bias": "encoder.stages.1.0.pwconv1.linear.bias",
    "stages.1.0.pwconv2.weight": "encoder.stages.1.0.pwconv2.linear.weight",
    "stages.1.0.pwconv2.bias": "encoder.stages.1.0.pwconv2.linear.bias",
    "stages.1.0.grn.gamma": "encoder.stages.1.0.grn.gamma",
    "stages.1.0.grn.beta": "encoder.stages.1.0.grn.beta",
    "stages.1.1.dwconv.weight": "encoder.stages.1.1.dwconv.kernel",
    "stages.1.1.dwconv.bias": "encoder.stages.1.1.dwconv.bias",
    "stages.1.1.norm.weight": "encoder.stages.1.1.norm.ln.weight",
    "stages.1.1.norm.bias": "encoder.stages.1.1.norm.ln.bias",
    "stages.1.1.pwconv1.weight": "encoder.stages.1.1.pwconv1.linear.weight",
    "stages.1.1.pwconv1.bias": "encoder.stages.1.1.pwconv1.linear.bias",
    "stages.1.1.pwconv2.weight": "encoder.stages.1.1.pwconv2.linear.weight",
    "stages.1.1.pwconv2.bias": "encoder.stages.1.1.pwconv2.linear.bias",
    "stages.1.1.grn.gamma": "encoder.stages.1.1.grn.gamma",
    "stages.1.1.grn.beta": "encoder.stages.1.1.grn.beta",
    "stages.1.2.dwconv.weight": "encoder.stages.1.2.dwconv.kernel",
    "stages.1.2.dwconv.bias": "encoder.stages.1.2.dwconv.bias",
    "stages.1.2.norm.weight": "encoder.stages.1.2.norm.ln.weight",
    "stages.1.2.norm.bias": "encoder.stages.1.2.norm.ln.bias",
    "stages.1.2.pwconv1.weight": "encoder.stages.1.2.pwconv1.linear.weight",
    "stages.1.2.pwconv1.bias": "encoder.stages.1.2.pwconv1.linear.bias",
    "stages.1.2.pwconv2.weight": "encoder.stages.1.2.pwconv2.linear.weight",
    "stages.1.2.pwconv2.bias": "encoder.stages.1.2.pwconv2.linear.bias",
    "stages.1.2.grn.gamma": "encoder.stages.1.2.grn.gamma",
    "stages.1.2.grn.beta": "encoder.stages.1.2.grn.beta",
    "stages.2.0.dwconv.weight": "encoder.stages.2.0.dwconv.kernel",
    "stages.2.0.dwconv.bias": "encoder.stages.2.0.dwconv.bias",
    "stages.2.0.norm.weight": "encoder.stages.2.0.norm.ln.weight",
    "stages.2.0.norm.bias": "encoder.stages.2.0.norm.ln.bias",
    "stages.2.0.pwconv1.weight": "encoder.stages.2.0.pwconv1.linear.weight",
    "stages.2.0.pwconv1.bias": "encoder.stages.2.0.pwconv1.linear.bias",
    "stages.2.0.pwconv2.weight": "encoder.stages.2.0.pwconv2.linear.weight",
    "stages.2.0.pwconv2.bias": "encoder.stages.2.0.pwconv2.linear.bias",
    "stages.2.0.grn.gamma": "encoder.stages.2.0.grn.gamma",
    "stages.2.0.grn.beta": "encoder.stages.2.0.grn.beta",
    "stages.2.1.dwconv.weight": "encoder.stages.2.1.dwconv.kernel",
    "stages.2.1.dwconv.bias": "encoder.stages.2.1.dwconv.bias",
    "stages.2.1.norm.weight": "encoder.stages.2.1.norm.ln.weight",
    "stages.2.1.norm.bias": "encoder.stages.2.1.norm.ln.bias",
    "stages.2.1.pwconv1.weight": "encoder.stages.2.1.pwconv1.linear.weight",
    "stages.2.1.pwconv1.bias": "encoder.stages.2.1.pwconv1.linear.bias",
    "stages.2.1.pwconv2.weight": "encoder.stages.2.1.pwconv2.linear.weight",
    "stages.2.1.pwconv2.bias": "encoder.stages.2.1.pwconv2.linear.bias",
    "stages.2.1.grn.gamma": "encoder.stages.2.1.grn.gamma",
    "stages.2.1.grn.beta": "encoder.stages.2.1.grn.beta",
    "stages.2.2.dwconv.weight": "encoder.stages.2.2.dwconv.kernel",
    "stages.2.2.dwconv.bias": "encoder.stages.2.2.dwconv.bias",
    "stages.2.2.norm.weight": "encoder.stages.2.2.norm.ln.weight",
    "stages.2.2.norm.bias": "encoder.stages.2.2.norm.ln.bias",
    "stages.2.2.pwconv1.weight": "encoder.stages.2.2.pwconv1.linear.weight",
    "stages.2.2.pwconv1.bias": "encoder.stages.2.2.pwconv1.linear.bias",
    "stages.2.2.pwconv2.weight": "encoder.stages.2.2.pwconv2.linear.weight",
    "stages.2.2.pwconv2.bias": "encoder.stages.2.2.pwconv2.linear.bias",
    "stages.2.2.grn.gamma": "encoder.stages.2.2.grn.gamma",
    "stages.2.2.grn.beta": "encoder.stages.2.2.grn.beta",
    "stages.2.3.dwconv.weight": "encoder.stages.2.3.dwconv.kernel",
    "stages.2.3.dwconv.bias": "encoder.stages.2.3.dwconv.bias",
    "stages.2.3.norm.weight": "encoder.stages.2.3.norm.ln.weight",
    "stages.2.3.norm.bias": "encoder.stages.2.3.norm.ln.bias",
    "stages.2.3.pwconv1.weight": "encoder.stages.2.3.pwconv1.linear.weight",
    "stages.2.3.pwconv1.bias": "encoder.stages.2.3.pwconv1.linear.bias",
    "stages.2.3.pwconv2.weight": "encoder.stages.2.3.pwconv2.linear.weight",
    "stages.2.3.pwconv2.bias": "encoder.stages.2.3.pwconv2.linear.bias",
    "stages.2.3.grn.gamma": "encoder.stages.2.3.grn.gamma",
    "stages.2.3.grn.beta": "encoder.stages.2.3.grn.beta",
    "stages.2.4.dwconv.weight": "encoder.stages.2.4.dwconv.kernel",
    "stages.2.4.dwconv.bias": "encoder.stages.2.4.dwconv.bias",
    "stages.2.4.norm.weight": "encoder.stages.2.4.norm.ln.weight",
    "stages.2.4.norm.bias": "encoder.stages.2.4.norm.ln.bias",
    "stages.2.4.pwconv1.weight": "encoder.stages.2.4.pwconv1.linear.weight",
    "stages.2.4.pwconv1.bias": "encoder.stages.2.4.pwconv1.linear.bias",
    "stages.2.4.pwconv2.weight": "encoder.stages.2.4.pwconv2.linear.weight",
    "stages.2.4.pwconv2.bias": "encoder.stages.2.4.pwconv2.linear.bias",
    "stages.2.4.grn.gamma": "encoder.stages.2.4.grn.gamma",
    "stages.2.4.grn.beta": "encoder.stages.2.4.grn.beta",
    "stages.2.5.dwconv.weight": "encoder.stages.2.5.dwconv.kernel",
    "stages.2.5.dwconv.bias": "encoder.stages.2.5.dwconv.bias",
    "stages.2.5.norm.weight": "encoder.stages.2.5.norm.ln.weight",
    "stages.2.5.norm.bias": "encoder.stages.2.5.norm.ln.bias",
    "stages.2.5.pwconv1.weight": "encoder.stages.2.5.pwconv1.linear.weight",
    "stages.2.5.pwconv1.bias": "encoder.stages.2.5.pwconv1.linear.bias",
    "stages.2.5.pwconv2.weight": "encoder.stages.2.5.pwconv2.linear.weight",
    "stages.2.5.pwconv2.bias": "encoder.stages.2.5.pwconv2.linear.bias",
    "stages.2.5.grn.gamma": "encoder.stages.2.5.grn.gamma",
    "stages.2.5.grn.beta": "encoder.stages.2.5.grn.beta",
    "stages.2.6.dwconv.weight": "encoder.stages.2.6.dwconv.kernel",
    "stages.2.6.dwconv.bias": "encoder.stages.2.6.dwconv.bias",
    "stages.2.6.norm.weight": "encoder.stages.2.6.norm.ln.weight",
    "stages.2.6.norm.bias": "encoder.stages.2.6.norm.ln.bias",
    "stages.2.6.pwconv1.weight": "encoder.stages.2.6.pwconv1.linear.weight",
    "stages.2.6.pwconv1.bias": "encoder.stages.2.6.pwconv1.linear.bias",
    "stages.2.6.pwconv2.weight": "encoder.stages.2.6.pwconv2.linear.weight",
    "stages.2.6.pwconv2.bias": "encoder.stages.2.6.pwconv2.linear.bias",
    "stages.2.6.grn.gamma": "encoder.stages.2.6.grn.gamma",
    "stages.2.6.grn.beta": "encoder.stages.2.6.grn.beta",
    "stages.2.7.dwconv.weight": "encoder.stages.2.7.dwconv.kernel",
    "stages.2.7.dwconv.bias": "encoder.stages.2.7.dwconv.bias",
    "stages.2.7.norm.weight": "encoder.stages.2.7.norm.ln.weight",
    "stages.2.7.norm.bias": "encoder.stages.2.7.norm.ln.bias",
    "stages.2.7.pwconv1.weight": "encoder.stages.2.7.pwconv1.linear.weight",
    "stages.2.7.pwconv1.bias": "encoder.stages.2.7.pwconv1.linear.bias",
    "stages.2.7.pwconv2.weight": "encoder.stages.2.7.pwconv2.linear.weight",
    "stages.2.7.pwconv2.bias": "encoder.stages.2.7.pwconv2.linear.bias",
    "stages.2.7.grn.gamma": "encoder.stages.2.7.grn.gamma",
    "stages.2.7.grn.beta": "encoder.stages.2.7.grn.beta",
    "stages.2.8.dwconv.weight": "encoder.stages.2.8.dwconv.kernel",
    "stages.2.8.dwconv.bias": "encoder.stages.2.8.dwconv.bias",
    "stages.2.8.norm.weight": "encoder.stages.2.8.norm.ln.weight",
    "stages.2.8.norm.bias": "encoder.stages.2.8.norm.ln.bias",
    "stages.2.8.pwconv1.weight": "encoder.stages.2.8.pwconv1.linear.weight",
    "stages.2.8.pwconv1.bias": "encoder.stages.2.8.pwconv1.linear.bias",
    "stages.2.8.pwconv2.weight": "encoder.stages.2.8.pwconv2.linear.weight",
    "stages.2.8.pwconv2.bias": "encoder.stages.2.8.pwconv2.linear.bias",
    "stages.2.8.grn.gamma": "encoder.stages.2.8.grn.gamma",
    "stages.2.8.grn.beta": "encoder.stages.2.8.grn.beta",
    "stages.2.9.dwconv.weight": "encoder.stages.2.9.dwconv.kernel",
    "stages.2.9.dwconv.bias": "encoder.stages.2.9.dwconv.bias",
    "stages.2.9.norm.weight": "encoder.stages.2.9.norm.ln.weight",
    "stages.2.9.norm.bias": "encoder.stages.2.9.norm.ln.bias",
    "stages.2.9.pwconv1.weight": "encoder.stages.2.9.pwconv1.linear.weight",
    "stages.2.9.pwconv1.bias": "encoder.stages.2.9.pwconv1.linear.bias",
    "stages.2.9.pwconv2.weight": "encoder.stages.2.9.pwconv2.linear.weight",
    "stages.2.9.pwconv2.bias": "encoder.stages.2.9.pwconv2.linear.bias",
    "stages.2.9.grn.gamma": "encoder.stages.2.9.grn.gamma",
    "stages.2.9.grn.beta": "encoder.stages.2.9.grn.beta",
    "stages.2.10.dwconv.weight": "encoder.stages.2.10.dwconv.kernel",
    "stages.2.10.dwconv.bias": "encoder.stages.2.10.dwconv.bias",
    "stages.2.10.norm.weight": "encoder.stages.2.10.norm.ln.weight",
    "stages.2.10.norm.bias": "encoder.stages.2.10.norm.ln.bias",
    "stages.2.10.pwconv1.weight": "encoder.stages.2.10.pwconv1.linear.weight",
    "stages.2.10.pwconv1.bias": "encoder.stages.2.10.pwconv1.linear.bias",
    "stages.2.10.pwconv2.weight": "encoder.stages.2.10.pwconv2.linear.weight",
    "stages.2.10.pwconv2.bias": "encoder.stages.2.10.pwconv2.linear.bias",
    "stages.2.10.grn.gamma": "encoder.stages.2.10.grn.gamma",
    "stages.2.10.grn.beta": "encoder.stages.2.10.grn.beta",
    "stages.2.11.dwconv.weight": "encoder.stages.2.11.dwconv.kernel",
    "stages.2.11.dwconv.bias": "encoder.stages.2.11.dwconv.bias",
    "stages.2.11.norm.weight": "encoder.stages.2.11.norm.ln.weight",
    "stages.2.11.norm.bias": "encoder.stages.2.11.norm.ln.bias",
    "stages.2.11.pwconv1.weight": "encoder.stages.2.11.pwconv1.linear.weight",
    "stages.2.11.pwconv1.bias": "encoder.stages.2.11.pwconv1.linear.bias",
    "stages.2.11.pwconv2.weight": "encoder.stages.2.11.pwconv2.linear.weight",
    "stages.2.11.pwconv2.bias": "encoder.stages.2.11.pwconv2.linear.bias",
    "stages.2.11.grn.gamma": "encoder.stages.2.11.grn.gamma",
    "stages.2.11.grn.beta": "encoder.stages.2.11.grn.beta",
    "stages.2.12.dwconv.weight": "encoder.stages.2.12.dwconv.kernel",
    "stages.2.12.dwconv.bias": "encoder.stages.2.12.dwconv.bias",
    "stages.2.12.norm.weight": "encoder.stages.2.12.norm.ln.weight",
    "stages.2.12.norm.bias": "encoder.stages.2.12.norm.ln.bias",
    "stages.2.12.pwconv1.weight": "encoder.stages.2.12.pwconv1.linear.weight",
    "stages.2.12.pwconv1.bias": "encoder.stages.2.12.pwconv1.linear.bias",
    "stages.2.12.pwconv2.weight": "encoder.stages.2.12.pwconv2.linear.weight",
    "stages.2.12.pwconv2.bias": "encoder.stages.2.12.pwconv2.linear.bias",
    "stages.2.12.grn.gamma": "encoder.stages.2.12.grn.gamma",
    "stages.2.12.grn.beta": "encoder.stages.2.12.grn.beta",
    "stages.2.13.dwconv.weight": "encoder.stages.2.13.dwconv.kernel",
    "stages.2.13.dwconv.bias": "encoder.stages.2.13.dwconv.bias",
    "stages.2.13.norm.weight": "encoder.stages.2.13.norm.ln.weight",
    "stages.2.13.norm.bias": "encoder.stages.2.13.norm.ln.bias",
    "stages.2.13.pwconv1.weight": "encoder.stages.2.13.pwconv1.linear.weight",
    "stages.2.13.pwconv1.bias": "encoder.stages.2.13.pwconv1.linear.bias",
    "stages.2.13.pwconv2.weight": "encoder.stages.2.13.pwconv2.linear.weight",
    "stages.2.13.pwconv2.bias": "encoder.stages.2.13.pwconv2.linear.bias",
    "stages.2.13.grn.gamma": "encoder.stages.2.13.grn.gamma",
    "stages.2.13.grn.beta": "encoder.stages.2.13.grn.beta",
    "stages.2.14.dwconv.weight": "encoder.stages.2.14.dwconv.kernel",
    "stages.2.14.dwconv.bias": "encoder.stages.2.14.dwconv.bias",
    "stages.2.14.norm.weight": "encoder.stages.2.14.norm.ln.weight",
    "stages.2.14.norm.bias": "encoder.stages.2.14.norm.ln.bias",
    "stages.2.14.pwconv1.weight": "encoder.stages.2.14.pwconv1.linear.weight",
    "stages.2.14.pwconv1.bias": "encoder.stages.2.14.pwconv1.linear.bias",
    "stages.2.14.pwconv2.weight": "encoder.stages.2.14.pwconv2.linear.weight",
    "stages.2.14.pwconv2.bias": "encoder.stages.2.14.pwconv2.linear.bias",
    "stages.2.14.grn.gamma": "encoder.stages.2.14.grn.gamma",
    "stages.2.14.grn.beta": "encoder.stages.2.14.grn.beta",
    "stages.2.15.dwconv.weight": "encoder.stages.2.15.dwconv.kernel",
    "stages.2.15.dwconv.bias": "encoder.stages.2.15.dwconv.bias",
    "stages.2.15.norm.weight": "encoder.stages.2.15.norm.ln.weight",
    "stages.2.15.norm.bias": "encoder.stages.2.15.norm.ln.bias",
    "stages.2.15.pwconv1.weight": "encoder.stages.2.15.pwconv1.linear.weight",
    "stages.2.15.pwconv1.bias": "encoder.stages.2.15.pwconv1.linear.bias",
    "stages.2.15.pwconv2.weight": "encoder.stages.2.15.pwconv2.linear.weight",
    "stages.2.15.pwconv2.bias": "encoder.stages.2.15.pwconv2.linear.bias",
    "stages.2.15.grn.gamma": "encoder.stages.2.15.grn.gamma",
    "stages.2.15.grn.beta": "encoder.stages.2.15.grn.beta",
    "stages.2.16.dwconv.weight": "encoder.stages.2.16.dwconv.kernel",
    "stages.2.16.dwconv.bias": "encoder.stages.2.16.dwconv.bias",
    "stages.2.16.norm.weight": "encoder.stages.2.16.norm.ln.weight",
    "stages.2.16.norm.bias": "encoder.stages.2.16.norm.ln.bias",
    "stages.2.16.pwconv1.weight": "encoder.stages.2.16.pwconv1.linear.weight",
    "stages.2.16.pwconv1.bias": "encoder.stages.2.16.pwconv1.linear.bias",
    "stages.2.16.pwconv2.weight": "encoder.stages.2.16.pwconv2.linear.weight",
    "stages.2.16.pwconv2.bias": "encoder.stages.2.16.pwconv2.linear.bias",
    "stages.2.16.grn.gamma": "encoder.stages.2.16.grn.gamma",
    "stages.2.16.grn.beta": "encoder.stages.2.16.grn.beta",
    "stages.2.17.dwconv.weight": "encoder.stages.2.17.dwconv.kernel",
    "stages.2.17.dwconv.bias": "encoder.stages.2.17.dwconv.bias",
    "stages.2.17.norm.weight": "encoder.stages.2.17.norm.ln.weight",
    "stages.2.17.norm.bias": "encoder.stages.2.17.norm.ln.bias",
    "stages.2.17.pwconv1.weight": "encoder.stages.2.17.pwconv1.linear.weight",
    "stages.2.17.pwconv1.bias": "encoder.stages.2.17.pwconv1.linear.bias",
    "stages.2.17.pwconv2.weight": "encoder.stages.2.17.pwconv2.linear.weight",
    "stages.2.17.pwconv2.bias": "encoder.stages.2.17.pwconv2.linear.bias",
    "stages.2.17.grn.gamma": "encoder.stages.2.17.grn.gamma",
    "stages.2.17.grn.beta": "encoder.stages.2.17.grn.beta",
    "stages.2.18.dwconv.weight": "encoder.stages.2.18.dwconv.kernel",
    "stages.2.18.dwconv.bias": "encoder.stages.2.18.dwconv.bias",
    "stages.2.18.norm.weight": "encoder.stages.2.18.norm.ln.weight",
    "stages.2.18.norm.bias": "encoder.stages.2.18.norm.ln.bias",
    "stages.2.18.pwconv1.weight": "encoder.stages.2.18.pwconv1.linear.weight",
    "stages.2.18.pwconv1.bias": "encoder.stages.2.18.pwconv1.linear.bias",
    "stages.2.18.pwconv2.weight": "encoder.stages.2.18.pwconv2.linear.weight",
    "stages.2.18.pwconv2.bias": "encoder.stages.2.18.pwconv2.linear.bias",
    "stages.2.18.grn.gamma": "encoder.stages.2.18.grn.gamma",
    "stages.2.18.grn.beta": "encoder.stages.2.18.grn.beta",
    "stages.2.19.dwconv.weight": "encoder.stages.2.19.dwconv.kernel",
    "stages.2.19.dwconv.bias": "encoder.stages.2.19.dwconv.bias",
    "stages.2.19.norm.weight": "encoder.stages.2.19.norm.ln.weight",
    "stages.2.19.norm.bias": "encoder.stages.2.19.norm.ln.bias",
    "stages.2.19.pwconv1.weight": "encoder.stages.2.19.pwconv1.linear.weight",
    "stages.2.19.pwconv1.bias": "encoder.stages.2.19.pwconv1.linear.bias",
    "stages.2.19.pwconv2.weight": "encoder.stages.2.19.pwconv2.linear.weight",
    "stages.2.19.pwconv2.bias": "encoder.stages.2.19.pwconv2.linear.bias",
    "stages.2.19.grn.gamma": "encoder.stages.2.19.grn.gamma",
    "stages.2.19.grn.beta": "encoder.stages.2.19.grn.beta",
    "stages.2.20.dwconv.weight": "encoder.stages.2.20.dwconv.kernel",
    "stages.2.20.dwconv.bias": "encoder.stages.2.20.dwconv.bias",
    "stages.2.20.norm.weight": "encoder.stages.2.20.norm.ln.weight",
    "stages.2.20.norm.bias": "encoder.stages.2.20.norm.ln.bias",
    "stages.2.20.pwconv1.weight": "encoder.stages.2.20.pwconv1.linear.weight",
    "stages.2.20.pwconv1.bias": "encoder.stages.2.20.pwconv1.linear.bias",
    "stages.2.20.pwconv2.weight": "encoder.stages.2.20.pwconv2.linear.weight",
    "stages.2.20.pwconv2.bias": "encoder.stages.2.20.pwconv2.linear.bias",
    "stages.2.20.grn.gamma": "encoder.stages.2.20.grn.gamma",
    "stages.2.20.grn.beta": "encoder.stages.2.20.grn.beta",
    "stages.2.21.dwconv.weight": "encoder.stages.2.21.dwconv.kernel",
    "stages.2.21.dwconv.bias": "encoder.stages.2.21.dwconv.bias",
    "stages.2.21.norm.weight": "encoder.stages.2.21.norm.ln.weight",
    "stages.2.21.norm.bias": "encoder.stages.2.21.norm.ln.bias",
    "stages.2.21.pwconv1.weight": "encoder.stages.2.21.pwconv1.linear.weight",
    "stages.2.21.pwconv1.bias": "encoder.stages.2.21.pwconv1.linear.bias",
    "stages.2.21.pwconv2.weight": "encoder.stages.2.21.pwconv2.linear.weight",
    "stages.2.21.pwconv2.bias": "encoder.stages.2.21.pwconv2.linear.bias",
    "stages.2.21.grn.gamma": "encoder.stages.2.21.grn.gamma",
    "stages.2.21.grn.beta": "encoder.stages.2.21.grn.beta",
    "stages.2.22.dwconv.weight": "encoder.stages.2.22.dwconv.kernel",
    "stages.2.22.dwconv.bias": "encoder.stages.2.22.dwconv.bias",
    "stages.2.22.norm.weight": "encoder.stages.2.22.norm.ln.weight",
    "stages.2.22.norm.bias": "encoder.stages.2.22.norm.ln.bias",
    "stages.2.22.pwconv1.weight": "encoder.stages.2.22.pwconv1.linear.weight",
    "stages.2.22.pwconv1.bias": "encoder.stages.2.22.pwconv1.linear.bias",
    "stages.2.22.pwconv2.weight": "encoder.stages.2.22.pwconv2.linear.weight",
    "stages.2.22.pwconv2.bias": "encoder.stages.2.22.pwconv2.linear.bias",
    "stages.2.22.grn.gamma": "encoder.stages.2.22.grn.gamma",
    "stages.2.22.grn.beta": "encoder.stages.2.22.grn.beta",
    "stages.2.23.dwconv.weight": "encoder.stages.2.23.dwconv.kernel",
    "stages.2.23.dwconv.bias": "encoder.stages.2.23.dwconv.bias",
    "stages.2.23.norm.weight": "encoder.stages.2.23.norm.ln.weight",
    "stages.2.23.norm.bias": "encoder.stages.2.23.norm.ln.bias",
    "stages.2.23.pwconv1.weight": "encoder.stages.2.23.pwconv1.linear.weight",
    "stages.2.23.pwconv1.bias": "encoder.stages.2.23.pwconv1.linear.bias",
    "stages.2.23.pwconv2.weight": "encoder.stages.2.23.pwconv2.linear.weight",
    "stages.2.23.pwconv2.bias": "encoder.stages.2.23.pwconv2.linear.bias",
    "stages.2.23.grn.gamma": "encoder.stages.2.23.grn.gamma",
    "stages.2.23.grn.beta": "encoder.stages.2.23.grn.beta",
    "stages.2.24.dwconv.weight": "encoder.stages.2.24.dwconv.kernel",
    "stages.2.24.dwconv.bias": "encoder.stages.2.24.dwconv.bias",
    "stages.2.24.norm.weight": "encoder.stages.2.24.norm.ln.weight",
    "stages.2.24.norm.bias": "encoder.stages.2.24.norm.ln.bias",
    "stages.2.24.pwconv1.weight": "encoder.stages.2.24.pwconv1.linear.weight",
    "stages.2.24.pwconv1.bias": "encoder.stages.2.24.pwconv1.linear.bias",
    "stages.2.24.pwconv2.weight": "encoder.stages.2.24.pwconv2.linear.weight",
    "stages.2.24.pwconv2.bias": "encoder.stages.2.24.pwconv2.linear.bias",
    "stages.2.24.grn.gamma": "encoder.stages.2.24.grn.gamma",
    "stages.2.24.grn.beta": "encoder.stages.2.24.grn.beta",
    "stages.2.25.dwconv.weight": "encoder.stages.2.25.dwconv.kernel",
    "stages.2.25.dwconv.bias": "encoder.stages.2.25.dwconv.bias",
    "stages.2.25.norm.weight": "encoder.stages.2.25.norm.ln.weight",
    "stages.2.25.norm.bias": "encoder.stages.2.25.norm.ln.bias",
    "stages.2.25.pwconv1.weight": "encoder.stages.2.25.pwconv1.linear.weight",
    "stages.2.25.pwconv1.bias": "encoder.stages.2.25.pwconv1.linear.bias",
    "stages.2.25.pwconv2.weight": "encoder.stages.2.25.pwconv2.linear.weight",
    "stages.2.25.pwconv2.bias": "encoder.stages.2.25.pwconv2.linear.bias",
    "stages.2.25.grn.gamma": "encoder.stages.2.25.grn.gamma",
    "stages.2.25.grn.beta": "encoder.stages.2.25.grn.beta",
    "stages.2.26.dwconv.weight": "encoder.stages.2.26.dwconv.kernel",
    "stages.2.26.dwconv.bias": "encoder.stages.2.26.dwconv.bias",
    "stages.2.26.norm.weight": "encoder.stages.2.26.norm.ln.weight",
    "stages.2.26.norm.bias": "encoder.stages.2.26.norm.ln.bias",
    "stages.2.26.pwconv1.weight": "encoder.stages.2.26.pwconv1.linear.weight",
    "stages.2.26.pwconv1.bias": "encoder.stages.2.26.pwconv1.linear.bias",
    "stages.2.26.pwconv2.weight": "encoder.stages.2.26.pwconv2.linear.weight",
    "stages.2.26.pwconv2.bias": "encoder.stages.2.26.pwconv2.linear.bias",
    "stages.2.26.grn.gamma": "encoder.stages.2.26.grn.gamma",
    "stages.2.26.grn.beta": "encoder.stages.2.26.grn.beta",
    "stages.3.0.dwconv.weight": "encoder.stages.3.0.dwconv.kernel",
    "stages.3.0.dwconv.bias": "encoder.stages.3.0.dwconv.bias",
    "stages.3.0.norm.weight": "encoder.stages.3.0.norm.ln.weight",
    "stages.3.0.norm.bias": "encoder.stages.3.0.norm.ln.bias",
    "stages.3.0.pwconv1.weight": "encoder.stages.3.0.pwconv1.linear.weight",
    "stages.3.0.pwconv1.bias": "encoder.stages.3.0.pwconv1.linear.bias",
    "stages.3.0.pwconv2.weight": "encoder.stages.3.0.pwconv2.linear.weight",
    "stages.3.0.pwconv2.bias": "encoder.stages.3.0.pwconv2.linear.bias",
    "stages.3.0.grn.gamma": "encoder.stages.3.0.grn.gamma",
    "stages.3.0.grn.beta": "encoder.stages.3.0.grn.beta",
    "stages.3.1.dwconv.weight": "encoder.stages.3.1.dwconv.kernel",
    "stages.3.1.dwconv.bias": "encoder.stages.3.1.dwconv.bias",
    "stages.3.1.norm.weight": "encoder.stages.3.1.norm.ln.weight",
    "stages.3.1.norm.bias": "encoder.stages.3.1.norm.ln.bias",
    "stages.3.1.pwconv1.weight": "encoder.stages.3.1.pwconv1.linear.weight",
    "stages.3.1.pwconv1.bias": "encoder.stages.3.1.pwconv1.linear.bias",
    "stages.3.1.pwconv2.weight": "encoder.stages.3.1.pwconv2.linear.weight",
    "stages.3.1.pwconv2.bias": "encoder.stages.3.1.pwconv2.linear.bias",
    "stages.3.1.grn.gamma": "encoder.stages.3.1.grn.gamma",
    "stages.3.1.grn.beta": "encoder.stages.3.1.grn.beta",
    "stages.3.2.dwconv.weight": "encoder.stages.3.2.dwconv.kernel",
    "stages.3.2.dwconv.bias": "encoder.stages.3.2.dwconv.bias",
    "stages.3.2.norm.weight": "encoder.stages.3.2.norm.ln.weight",
    "stages.3.2.norm.bias": "encoder.stages.3.2.norm.ln.bias",
    "stages.3.2.pwconv1.weight": "encoder.stages.3.2.pwconv1.linear.weight",
    "stages.3.2.pwconv1.bias": "encoder.stages.3.2.pwconv1.linear.bias",
    "stages.3.2.pwconv2.weight": "encoder.stages.3.2.pwconv2.linear.weight",
    "stages.3.2.pwconv2.bias": "encoder.stages.3.2.pwconv2.linear.bias",
    "stages.3.2.grn.gamma": "encoder.stages.3.2.grn.gamma",
    "stages.3.2.grn.beta": "encoder.stages.3.2.grn.beta",
}

unsqueezer = [
    "encoder.downsample_layers.1.1.bias",
"encoder.downsample_layers.2.1.bias",
"encoder.downsample_layers.3.1.bias",
"encoder.stages.0.0.dwconv.bias",
"encoder.stages.0.0.grn.gamma",
"encoder.stages.0.0.grn.beta",
"encoder.stages.0.1.dwconv.bias",
"encoder.stages.0.1.grn.gamma",
"encoder.stages.0.1.grn.beta",
"encoder.stages.0.2.dwconv.bias",
"encoder.stages.0.2.grn.gamma",
"encoder.stages.0.2.grn.beta",
"encoder.stages.1.0.dwconv.bias",
"encoder.stages.1.0.grn.gamma",
"encoder.stages.1.0.grn.beta",
"encoder.stages.1.1.dwconv.bias",
"encoder.stages.1.1.grn.gamma",
"encoder.stages.1.1.grn.beta",
"encoder.stages.1.2.dwconv.bias",
"encoder.stages.1.2.grn.gamma",
"encoder.stages.1.2.grn.beta",
"encoder.stages.2.0.dwconv.bias",
"encoder.stages.2.0.grn.gamma",
"encoder.stages.2.0.grn.beta",
"encoder.stages.2.1.dwconv.bias",
"encoder.stages.2.1.grn.gamma",
"encoder.stages.2.1.grn.beta",
"encoder.stages.2.2.dwconv.bias",
"encoder.stages.2.2.grn.gamma",
"encoder.stages.2.2.grn.beta",
"encoder.stages.2.3.dwconv.bias",
"encoder.stages.2.3.grn.gamma",
"encoder.stages.2.3.grn.beta",
"encoder.stages.2.4.dwconv.bias",
"encoder.stages.2.4.grn.gamma",
"encoder.stages.2.4.grn.beta",
"encoder.stages.2.5.dwconv.bias",
"encoder.stages.2.5.grn.gamma",
"encoder.stages.2.5.grn.beta",
"encoder.stages.2.6.dwconv.bias",
"encoder.stages.2.6.grn.gamma",
"encoder.stages.2.6.grn.beta",
"encoder.stages.2.7.dwconv.bias",
"encoder.stages.2.7.grn.gamma",
"encoder.stages.2.7.grn.beta",
"encoder.stages.2.8.dwconv.bias",
"encoder.stages.2.8.grn.gamma",
"encoder.stages.2.8.grn.beta",
"encoder.stages.2.9.dwconv.bias",
"encoder.stages.2.9.grn.gamma",
"encoder.stages.2.9.grn.beta",
"encoder.stages.2.10.dwconv.bias",
"encoder.stages.2.10.grn.gamma",
"encoder.stages.2.10.grn.beta",
"encoder.stages.2.11.dwconv.bias",
"encoder.stages.2.11.grn.gamma",
"encoder.stages.2.11.grn.beta",
"encoder.stages.2.12.dwconv.bias",
"encoder.stages.2.12.grn.gamma",
"encoder.stages.2.12.grn.beta",
"encoder.stages.2.13.dwconv.bias",
"encoder.stages.2.13.grn.gamma",
"encoder.stages.2.13.grn.beta",
"encoder.stages.2.14.dwconv.bias",
"encoder.stages.2.14.grn.gamma",
"encoder.stages.2.14.grn.beta",
"encoder.stages.2.15.dwconv.bias",
"encoder.stages.2.15.grn.gamma",
"encoder.stages.2.15.grn.beta",
"encoder.stages.2.16.dwconv.bias",
"encoder.stages.2.16.grn.gamma",
"encoder.stages.2.16.grn.beta",
"encoder.stages.2.17.dwconv.bias",
"encoder.stages.2.17.grn.gamma",
"encoder.stages.2.17.grn.beta",
"encoder.stages.2.18.dwconv.bias",
"encoder.stages.2.18.grn.gamma",
"encoder.stages.2.18.grn.beta",
"encoder.stages.2.19.dwconv.bias",
"encoder.stages.2.19.grn.gamma",
"encoder.stages.2.19.grn.beta",
"encoder.stages.2.20.dwconv.bias",
"encoder.stages.2.20.grn.gamma",
"encoder.stages.2.20.grn.beta",
"encoder.stages.2.21.dwconv.bias",
"encoder.stages.2.21.grn.gamma",
"encoder.stages.2.21.grn.beta",
"encoder.stages.2.22.dwconv.bias",
"encoder.stages.2.22.grn.gamma",
"encoder.stages.2.22.grn.beta",
"encoder.stages.2.23.dwconv.bias",
"encoder.stages.2.23.grn.gamma",
"encoder.stages.2.23.grn.beta",
"encoder.stages.2.24.dwconv.bias",
"encoder.stages.2.24.grn.gamma",
"encoder.stages.2.24.grn.beta",
"encoder.stages.2.25.dwconv.bias",
"encoder.stages.2.25.grn.gamma",
"encoder.stages.2.25.grn.beta",
"encoder.stages.2.26.dwconv.bias",
"encoder.stages.2.26.grn.gamma",
"encoder.stages.2.26.grn.beta",
"encoder.stages.3.0.dwconv.bias",
"encoder.stages.3.0.grn.gamma",
"encoder.stages.3.0.grn.beta",
"encoder.stages.3.1.dwconv.bias",
"encoder.stages.3.1.grn.gamma",
"encoder.stages.3.1.grn.beta",
"encoder.stages.3.2.dwconv.bias",
"encoder.stages.3.2.grn.gamma",
"encoder.stages.3.2.grn.beta"
]

three_unsqueezers = [
"encoder.stages.0.0.grn.gamma",
"encoder.stages.0.0.grn.beta",
"encoder.stages.0.1.grn.gamma",
"encoder.stages.0.1.grn.beta",
"encoder.stages.0.2.grn.gamma",
"encoder.stages.0.2.grn.beta",
"encoder.stages.1.0.grn.gamma",
"encoder.stages.1.0.grn.beta",
"encoder.stages.1.1.grn.gamma",
"encoder.stages.1.1.grn.beta",
"encoder.stages.1.2.grn.gamma",
"encoder.stages.1.2.grn.beta",
"encoder.stages.2.0.grn.gamma",
"encoder.stages.2.0.grn.beta",
"encoder.stages.2.1.grn.gamma",
"encoder.stages.2.1.grn.beta",
"encoder.stages.2.2.grn.gamma",
"encoder.stages.2.2.grn.beta",
"encoder.stages.2.3.grn.gamma",
"encoder.stages.2.3.grn.beta",
"encoder.stages.2.4.grn.gamma",
"encoder.stages.2.4.grn.beta",
"encoder.stages.2.5.grn.gamma",
"encoder.stages.2.5.grn.beta",
"encoder.stages.2.6.grn.gamma",
"encoder.stages.2.6.grn.beta",
"encoder.stages.2.7.grn.gamma",
"encoder.stages.2.7.grn.beta",
"encoder.stages.2.8.grn.gamma",
"encoder.stages.2.8.grn.beta",
"encoder.stages.2.9.grn.gamma",
"encoder.stages.2.9.grn.beta",
"encoder.stages.2.10.grn.gamma",
"encoder.stages.2.10.grn.beta",
"encoder.stages.2.11.grn.gamma",
"encoder.stages.2.11.grn.beta",
"encoder.stages.2.12.grn.gamma",
"encoder.stages.2.12.grn.beta",
"encoder.stages.2.13.grn.gamma",
"encoder.stages.2.13.grn.beta",
"encoder.stages.2.14.grn.gamma",
"encoder.stages.2.14.grn.beta",
"encoder.stages.2.15.grn.gamma",
"encoder.stages.2.15.grn.beta",
"encoder.stages.2.16.grn.gamma",
"encoder.stages.2.16.grn.beta",
"encoder.stages.2.17.grn.gamma",
"encoder.stages.2.17.grn.beta",
"encoder.stages.2.18.grn.gamma",
"encoder.stages.2.18.grn.beta",
"encoder.stages.2.19.grn.gamma",
"encoder.stages.2.19.grn.beta",
"encoder.stages.2.20.grn.gamma",
"encoder.stages.2.20.grn.beta",
"encoder.stages.2.21.grn.gamma",
"encoder.stages.2.21.grn.beta",
"encoder.stages.2.22.grn.gamma",
"encoder.stages.2.22.grn.beta",
"encoder.stages.2.23.grn.gamma",
"encoder.stages.2.23.grn.beta",
"encoder.stages.2.24.grn.gamma",
"encoder.stages.2.24.grn.beta",
"encoder.stages.2.25.grn.gamma",
"encoder.stages.2.25.grn.beta",
"encoder.stages.2.26.grn.gamma",
"encoder.stages.2.26.grn.beta",
"encoder.stages.3.0.grn.gamma",
"encoder.stages.3.0.grn.beta",
"encoder.stages.3.1.grn.gamma",
"encoder.stages.3.1.grn.beta",
"encoder.stages.3.2.grn.gamma",
"encoder.stages.3.2.grn.beta",
]


def dedense_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k not in mappings:
            continue
        new_k = mappings[k]
        if k in standard_convs:
            out_dim,in_dim,ks,ks = v.shape
            new_ckpt[new_k] = v.transpose(3, 2).reshape(out_dim, in_dim, int(ks * ks)).permute(2, 1, 0)
        elif k in depthwise_convs:
            dim,_,ks,ks = v.shape
            new_ckpt[new_k] = v.transpose(3, 2).reshape(dim, ks * ks).permute(1, 0)
        elif new_k in unsqueezer:
            v = v.squeeze().unsqueeze(0)
            new_ckpt[new_k] = v
        else:
            new_ckpt[new_k] = v
    return new_ckpt

def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = '.'.join(k.split('.')[1:]) # remove encoder in the name
        if k.endswith('kernel'):
            k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3: # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).\
                    reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2: # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).\
                    reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2) # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt