# Training

We provide FCMAE ImageNet-1K pre-training and fine-tuning scripts here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Multi-node Training
We use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit) for reproducing the results in the paper. Please install:
```
pip install submitit
```
We provide example commands for both multi-node and single-machine training below.


## ImageNet-1K FCMAE Pre-Training 
ConvNeXt V2-Base pre-training on ImageNet-1K with 8 8-GPU nodes:
```
python submitit_pretrain.py --nodes 8 --ngpus 8 \
--model convnextv2_base \
--batch_size 64 \
--blr 1.5e-4 \
--epochs 1600 \
--warmup_epochs 40 \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

The following commands run the pre-training on a single machine:

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--model convnextv2_base \
--batch_size 64 --update_freq 8 \
--blr 1.5e-4 \
--epochs 1600 \
--warmup_epochs 40 \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```


## ImageNet-1K Fine-Tuning

ConvNeXt V2-Base fine-tuning on ImageNet-1K with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_base \
--batch_size 32 \
--blr 6.25e-4 \
--epochs 100 \
--warmup_epochs 20 \
--layer_decay_type 'group' \
--layer_decay 0.6 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

The following commands run the fine-tuning on a single machine:

```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_base \
--batch_size 32 --update_freq 4 \
--blr 6.25e-4 \
--epochs 100 \
--warmup_epochs 20 \
--layer_decay_type 'group' \
--layer_decay 0.6 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

<details>
<summary>
ConvNeXt-A
</summary>
  
ConvNeXt V2-Atto training on ImageNet-1K with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_atto \
--batch_size 32 \
--blr 2e-4 \
--epochs 600 \
--warmup_epochs 0 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.3 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0. \
--cutmix 0. \
--smoothing 0.2 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

The following commands run the fine-tuning on a single machine:
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_atto \
--batch_size 32 --update_freq 4 \
--blr 2e-4 \
--epochs 600 \
--warmup_epochs 0 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.3 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0. \
--cutmix 0. \
--smoothing 0.2 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>

<details>
<summary>
ConvNeXt-T
</summary>
  
ConvNeXt V2-Tiny training on ImageNet-1K with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_tiny \
--batch_size 32 \
--blr 8e-4 \
--epochs 300 \
--warmup_epochs 40 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.05 \
--drop_path 0.2 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

The following commands run the fine-tuning on a single machine:
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_ \
--batch_size 32 --update_freq 4 \
--blr 8e-4 \
--epochs 300 \
--warmup_epochs 40 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.05 \
--drop_path 0.2 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>

## Implementing FCMAE with Masked Convolution in JAX

In our paper, we trained our main results using the JAX framework on TPU VM Pods. However, we do not have an efficient sparse convolution kernel implementation in this environment. Therefore, we have included our JAX model definition that uses a masked (dense) convolution for FCMAE pre-training.

```python

from flax import linen as nn
import jax.numpy as jnp

class GRN(nn.Module):
  dim: int
  eps: float = 1e-6
  
  def init_fn(self, key, shape, fill_value):
    return jnp.full(shape, fill_value)
  
  @nn.compact
  def __call__(self, inputs, mask=None):
    gamma = self.param("gamma", self.init_fn, (self.dim,), 0.)
    beta = self.param("beta", self.init_fn, (self.dim,), 0.)
    
    x = inputs
    if mask is not None:
      x = x * (1. - mask)
    GX = jnp.power((jnp.sum(jnp.power(x, 2), axis=(1,2), keepdims=True) + self.eps), 0.5)
    Nx = Gx / (jnp.mean(Gx, axis=-1, keepdims=True) + self.eps)
    return gamma * (Nx * inputs) + beta + inputs
  
class Block(nn.Module):
  dim: int
  drop_path: float
  
  @nn.compact
  def __call__(self, inputs, mask=None):
    if mask is not None:
      x = inputs * (1. - mask)
    x = DepthwiseConv2D((7, 7), name='dwconv')(x)
    if mask is not None: # The binary masking is numerically identical to sparse conv.
      x = x * (1.- mask)
    x = nn.LayerNorm(name='norm')(x)
    x = nn.Dense(4 * self.dim, name='pwconv1')(x)
    x = nn.gelu(x)
    x = GRN(4 * self.dim, name='grn')(x, mask)
    x = nn.Dense(self.dim, name='pwconv2')(x)
    x = nn.Dropout(rate=self.drop_path, broadcast_dims=(1,2,3), name='droppath')(x, deterministic=not self.training)
    return x + inputs
```

