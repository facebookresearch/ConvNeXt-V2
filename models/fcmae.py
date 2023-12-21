# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import wandb
from time import time
import numpy as np
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork,LastLevelMaxPool

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)

from timm.models.layers import trunc_normal_
from .convnextv2_sparse import SparseConvNeXtV2
from .convnextv2 import Block

class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
                self,
                img_size=224,
                in_chans=1,
                depths=[3, 3, 9, 3],
                dims=[96, 192, 384, 768],
                decoder_depth=1,
                decoder_embed_dim=512,
                patch_size=32,
                mask_ratio=0.6,
                norm_pix_loss=False,
                use_fpn=False,
                sigmoid = False
        ):
        super().__init__()
        
        # Patches must fit the image size
        assert (img_size % patch_size == 0), "Patch size must fit image size."

        # configs
        self.img_size = img_size
        self.time_since_last_img_save = time()
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.channels = in_chans
        self.use_fpn = use_fpn

        # encoder
        if use_fpn:
            proj_input = 256
            pred_output = 16
        else:
            proj_input=1024
            pred_output = patch_size ** 2
        self.encoder = SparseConvNeXtV2(
            patch_size=patch_size,in_chans=in_chans, depths=depths, dims=dims, D=2,use_fpn=use_fpn)
        # decoder
        self.proj = nn.Conv2d(
            in_channels=proj_input, 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=pred_output,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs,p=None):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        if p is None:
            p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        return x

    def unpatchify(self, x,p=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if p is None:
            p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.channels, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2 # number of patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    
    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask
    
    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        mask = self.upsample_mask(mask,int((x.shape[2] / (mask.shape[1] ** .5)))).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred
    
    def save_imgs(self,imgs,pred,combined,index=0,unpatch_factor=None):
        target_img = imgs[index].permute(1,2,0).detach().cpu().numpy()
        pred_img = self.unpatchify(pred,p=unpatch_factor)[index].permute(1,2,0).detach().cpu().numpy()
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
        combined_img = combined[index].permute(1,2,0).detach().cpu().numpy()
        target_img = (target_img - np.min(target_img))/np.ptp(target_img)
        pred_img = (pred_img - np.min(pred_img))/np.ptp(pred_img)
        combined_img = (combined_img - np.min(combined_img))/np.ptp(combined_img)
        x,y,c = target_img.shape
        combined_image = np.zeros((x,y*3,1))
        combined_image[:,:y,:] = target_img
        combined_image[:,y:y*2,:] = pred_img
        combined_image[:,y*2:,:] = combined_img
        images = wandb.Image(
            combined_image, 
            caption="Left: Target, Middle: Pred, Right: Combined"
            )  
        wandb.log({"Example": images})

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)
        if self.use_fpn:
            mask = self.upsample_mask(mask,8).reshape(mask.shape[0],-1)
            target = self.patchify(imgs,p=4)
        else:
            target = self.patchify(imgs)
        if time()-self.time_since_last_img_save > 1800: # 3600
            unpatch_factor = None
            upsample_factor = 32
            if self.use_fpn:
                unpatch_factor = 4
                upsample_factor = 4
            imgs_write = (imgs * (1-self.upsample_mask(mask,upsample_factor).unsqueeze(1)))
            preds_write = (self.unpatchify(pred,p=unpatch_factor) * self.upsample_mask(mask,upsample_factor).unsqueeze(1))
            preds_write = (preds_write - preds_write.min()) / (preds_write.max() - preds_write.min())
            combined = imgs_write + preds_write
            self.save_imgs(imgs,pred,combined,unpatch_factor=unpatch_factor)
            self.time_since_last_img_save = time()
# import cv2
# combined = (imgs * (1-self.upsample_mask(mask,32).unsqueeze(1))) + (self.unpatchify(pred) * self.upsample_mask(mask,32).unsqueeze(1))
# for i in range(len(combined)):
#    cv2.imwrite(f'comb_{i}.png',combined[i].permute(1,2,0).detach().cpu().numpy()*255)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # regularization terms
        l2_reg = None
        for name,W in self.named_parameters():
            if not W.requires_grad or 'encoder' not in name:
                continue
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        wandb.log({
            'l2_reg':(l2_reg*0.0000002),
            'loss: loss':loss
        })
        total_loss = loss + (l2_reg*0.0000002)
        return total_loss

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def convnextv2_atto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = FCMAE(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = FCMAE(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model