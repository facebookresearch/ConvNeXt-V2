import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

assert timm.__version__ == "0.3.2"  # version check
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import timm.optim.optim_factory as optim_factory
from timm.models.vision_transformer import PatchEmbed

import models.fcmae as fcmae
import utils
from clip_wrapper import CLIPWrapper
from datasets import build_convnextclip_pretraining_dataset
from engine_pretrain import train_one_epoch, train_one_epoch_clip
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool

args = argparse.Namespace()
args.data_path = '/imagenet/'
args.input_size = 224
teacher = CLIPWrapper(
        clip_model='ViT-L/14', 
        download_root='weights/clip/large'
    )

print("teacher = %s" % str(teacher))
args.teacher_out_feat_dim = teacher.net.visual.output_dim
print('teacher_out_feat_dim', args.teacher_out_feat_dim) 

dataset_train = build_convnextclip_pretraining_dataset(args, is_train=True)
print(dataset_train)

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()
seed = 0+ utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
sampler_train = torch.utils.data.DistributedSampler(
    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=0,
)
print("Sampler_train = %s" % str(sampler_train))

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=32,
    num_workers=1,    
    drop_last=True,
)

# define the model
model = fcmae.__dict__['convnextv2_atto'](
    mask_ratio=0.6,
    decoder_depth=1,
    decoder_embed_dim=512,
    norm_pix_loss=1
)

for batch, y in data_loader_train:
    samples, images = batch 
    break

device = 'cuda'
model.to(device)
samples = samples.to(device, non_blocking=True)
images = images.to(device, non_blocking=True)   

x,mask = model.forward_encoder(images,0.6)
pred = model.forward_decoder(x, mask)

def patchify(imgs):
    p = 32
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0]*h*w, 3, p, p))
    return x
    
teacher = teacher.to(device)
z = patchify(images).to(device)
clip_features = teacher.infer_image({"image": [z]}) 
patch_embed = PatchEmbed(224, 32, 3, 512).to(device)

#encoding isnide
mymask = model.gen_random_mask(images, 0.6)
myx = model.encoder(images, mymask)
#decoder inside
dex = model.proj(myx) #32, 512, 7, 7
n, c, h, w = dex.shape
demask = mymask.reshape(-1, h, w).unsqueeze(1).type_as(dex)
mask_token = model.mask_token.repeat(dex.shape[0], 1, dex.shape[2], dex.shape[3]) #32,512,7,7 
