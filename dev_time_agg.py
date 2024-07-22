import os
import torch
import numpy as np
import scOT.model as model

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def dev_forward_pass():
    img_size        = (128,128)
    patch_size      = (16,16)
    out_channels    = 20
    in_channels     = 3
    ntime           = 10
    nbatch          = 2
    n_patches = [img_size[i] / patch_size[i] for i  in np.arange(len(img_size)) ]
    
    embed_dim       = out_channels * patch_size[0]
    patch = model.DPOTPatchEmbed(   img_size    = img_size, 
                                patch_size  = patch_size, 
                                in_chans    = in_channels, 
                                embed_dim   = embed_dim, 
                                out_dim     = embed_dim,
                                act         = 'gelu',)
    # B x H x W x T x C
    x           = torch.randn((nbatch, img_size[0], img_size[1], ntime, in_channels))
    x = rearrange(x, 'b x y t c -> (b t) c x y')
    patch_out   = patch(x)
    print(patch_out.shape)

    patch_out_new   = rearrange( patch_out, '(b t) c x y -> b x y t c', b=nbatch, t=ntime)


    tagg = model.DPOTTimeAggregator(n_channels      = in_channels, 
                                n_timesteps     = ntime,
                                out_channels    = embed_dim, 
                                type='mlp')
    
    tagg_out = tagg(patch_out_new)
    print(tagg_out.shape)


def test_layer_square_mlp():
    img_size        = (128,128)
    patch_size      = (16,16)
    out_channels    = 20
    in_channels     = 3
    ntime           = 10
    nbatch          = 2
    embed_dim       = out_channels * patch_size[0]
    activation      = 'gelu'
    time_agg_type   = 'mlp'
    n_patches       = tuple([int(img_size[i] / patch_size[i]) for i  in np.arange(len(img_size)) ])
    expected_shape  = (nbatch, *n_patches, embed_dim )

    layer = model.DPOTPatchTimeAggregate(   image_size      = img_size,
                                            patch_size      = patch_size,
                                            embed_dim       = embed_dim,
                                            in_channels     = in_channels,
                                            num_timesteps   = ntime,
                                            activation      = activation,
                                            time_agg_type   = time_agg_type,)
    
    x = torch.randn((nbatch, img_size[0], img_size[1], ntime, in_channels))
    tagg_out = layer(x)

    assert tagg_out.shape == expected_shape, 'Test failed, expected output shape {}, but got {}'.format(
        expected_shape, tagg_out.shape
    )
    print('Test passed, expected output shape {}, got {}'.format(
        expected_shape, tagg_out.shape
    ))


def test_layer_square_exp_mlp():
    img_size        = (128,128)
    patch_size      = (16,16)
    out_channels    = 20
    in_channels     = 3
    ntime           = 12
    nbatch          = 13
    embed_dim       = out_channels * patch_size[0]
    activation      = 'gelu'
    time_agg_type   = 'exp_mlp'
    n_patches       = tuple([int(img_size[i] / patch_size[i]) for i  in np.arange(len(img_size)) ])
    expected_shape  = (nbatch, *n_patches, embed_dim )

    layer = model.DPOTPatchTimeAggregate(   image_size      = img_size,
                                            patch_size      = patch_size,
                                            embed_dim       = embed_dim,
                                            in_channels     = in_channels,
                                            num_timesteps   = ntime,
                                            activation      = activation,
                                            time_agg_type   = time_agg_type,)
    
    x = torch.randn((nbatch, img_size[0], img_size[1], ntime, in_channels))
    tagg_out = layer(x)

    assert tagg_out.shape == expected_shape, 'Test failed, expected output shape {}, but got {}'.format(
        expected_shape, tagg_out.shape
    )
    print('Test passed, expected output shape {}, got {}'.format(
        expected_shape, tagg_out.shape
    ))


    
if __name__ == '__main__':
    # test_layer_square_mlp()
    test_layer_square_exp_mlp()