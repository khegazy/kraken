import os
import torch
import numpy as np
import scOT.model as model

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def forward_pass():
    img_size        = (128,128)
    patch_size      = (16,16)
    out_channels    = 20
    in_channels     = 3
    ntime           = 10
    n_patches = [img_size[i] / patch_size[i] for i  in np.arange(len(img_size)) ]
    
    embed_dim       = out_channels * patch_size[0]
    patch = model.PatchEmbed(   img_size    = img_size, 
                                patch_size  = patch_size, 
                                in_chans    = in_channels, 
                                embed_dim   = embed_dim, 
                                out_dim     = embed_dim,
                                act         = 'gelu',)
    # B x H x W x T x C
    x           = torch.randn((1,img_size[0],img_size[1],ntime,in_channels))
    x = rearrange(x, 'b x y t c -> (b t) c x y')
    patch_out   = patch(x)
    print(patch_out.shape)

    patch_out = rearrange(patch_out, 'b c x y -> x y b c')

    tagg = model.TimeAggregator(n_channels      = in_channels, 
                                n_timesteps     = ntime,
                                out_channels    = embed_dim, 
                                type='mlp')
    
    tagg_out = tagg(patch_out)
    print(tagg_out.shape)


if __name__ == '__main__':
    forward_pass()



