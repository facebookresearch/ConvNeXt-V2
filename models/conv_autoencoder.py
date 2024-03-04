import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time
import wandb

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_since_last_img_save = time()
        nfe = 64 # input feat maps
        nz=128 # latent vector
        nfd = 65 # decoder feat maps

        self.encoder = nn.Sequential(
            # input (nc) x 128 x 128
            nn.Conv2d(1, nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe),
            nn.LeakyReLU(True),
            # input (nfe) x 64 x 64
            nn.Conv2d(nfe, nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 32 x 32
            nn.Conv2d(nfe * 2, nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 16 x 16
            nn.Conv2d(nfe * 4, nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 8 x 8
            nn.Conv2d(nfe * 8, nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe * 16),
            nn.LeakyReLU(True),
            # input (nfe*16) x 4 x 4
            nn.Conv2d(nfe * 16, nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(True)
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(nz, nz * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nz * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            nn.ConvTranspose2d(nz * 16, nz * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            nn.ConvTranspose2d(nz * 8, nz * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            nn.ConvTranspose2d(nz * 4, nz * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            nn.ConvTranspose2d(nz * 2, nz, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            nn.ConvTranspose2d(nz, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # output (nc) x 128 x 128
        )
        
    def save_imgs(self,imgs,pred,index=0):
        target_img = imgs[index].permute(1,2,0).detach().cpu().numpy()
        pred_img = pred[index].permute(1,2,0).detach().cpu().numpy()
        x,y,c = target_img.shape
        combined_image = np.zeros((x,y*2,1))
        combined_image[:,:y,:] = target_img
        combined_image[:,y:y*2,:] = pred_img
        images = wandb.Image(
            combined_image, 
            caption="Left: Target, Right: Pred"
            )  
        wandb.log({"Example": images})

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        emb = self.encoder(imgs)
        output = self.decoder(emb)
        loss = F.mse_loss(output, imgs)
        if time()-self.time_since_last_img_save > 60: # 3600
            self.time_since_last_img_save = time()
            self.save_imgs(imgs,output,0)
        return loss, None, None
