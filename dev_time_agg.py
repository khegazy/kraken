import os
import torch
import json
import numpy as np
import scOT.model as model

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def dev_forward_pass():
    img_size        = (128,128)
    patch_size      = (4,4)
    out_channels    = 1
    in_channels     = 4
    ntime           = 10
    nbatch          = 1
    n_patches = [img_size[i] / patch_size[i] for i  in np.arange(len(img_size)) ]
    
    # embed_dim       = out_channels * patch_size[0]
    embed_dim = 48
    patch = model.DPOTPatchEmbed(   img_size    = img_size, 
                                patch_size  = patch_size, 
                                in_chans    = in_channels, 
                                embed_dim   = embed_dim, 
                                out_dim     = embed_dim,
                                act         = 'gelu',)
    # # B x H x W x T x C
    # x           = torch.randn((nbatch, img_size[0], img_size[1], ntime, in_channels))
    # x = rearrange(x, 'b x y t c -> (b t) c x y')
    x = torch.randn((nbatch, in_channels, img_size[0], img_size[1], ntime))
    x = rearrange(x, 'b c x y t -> (b t) c x y')
    patch_out   = patch(x)
    print(patch_out.shape)

    patch_out_new  = rearrange(patch_out, '(b t) c x y -> b (x y) t c', b=nbatch, t=ntime)

    # patch_out_new   = rearrange( patch_out, '(b t) c x y -> b x y t c', b=nbatch, t=ntime)

    tagg = model.DPOTTimeAggregator(n_channels      = in_channels, 
                                n_timesteps     = ntime,
                                out_channels    = embed_dim, 
                                type='mlp')
    
    tagg_out = tagg(patch_out_new)
    print(tagg_out.shape)


def test_layer_square_mlp():
    img_size        = (128,128)
    patch_size      = (4,4)
    out_channels    = 48
    in_channels     = 3
    ntime           = 10
    nbatch          = 2
    embed_dim       = out_channels
    activation      = 'gelu'
    time_agg_type   = 'mlp'
    n_patches       = tuple([int(img_size[i] / patch_size[i]) for i  in np.arange(len(img_size)) ])
    n_patches_tot   = n_patches[0] * n_patches[1]
    expected_shape  = (nbatch, n_patches_tot, embed_dim )

    layer = model.DPOTPatchTimeAggregate(   image_size      = img_size,
                                            patch_size      = patch_size,
                                            embed_dim       = embed_dim,
                                            in_channels     = in_channels,
                                            num_timesteps   = ntime,
                                            activation      = activation,
                                            time_agg_type   = time_agg_type,)
    
    x = torch.randn((nbatch, in_channels, img_size[0], img_size[1], ntime, ))
    tagg_out, shape_out = layer(x)

    assert tagg_out.shape == expected_shape, 'Test failed, expected output shape {}, but got {}'.format(
        expected_shape, tagg_out.shape
    )
    print('Test passed, expected output shape {}, got {}'.format(
        expected_shape, tagg_out.shape
    ))


def test_layer_square_exp_mlp():
    img_size        = (128,128)
    patch_size      = (4,4)
    out_channels    = 48
    in_channels     = 3
    ntime           = 12
    nbatch          = 13
    embed_dim       = out_channels
    activation      = 'gelu'
    time_agg_type   = 'exp_mlp'
    n_patches       = tuple([int(img_size[i] / patch_size[i]) for i  in np.arange(len(img_size)) ])
    n_patches_tot   = n_patches[0] * n_patches[1]
    expected_shape  = (nbatch, n_patches_tot, embed_dim )

    layer = model.DPOTPatchTimeAggregate(   image_size      = img_size,
                                            patch_size      = patch_size,
                                            embed_dim       = embed_dim,
                                            in_channels     = in_channels,
                                            num_timesteps   = ntime,
                                            activation      = activation,
                                            time_agg_type   = time_agg_type,)
    
    x = torch.randn((nbatch, in_channels, img_size[0], img_size[1], ntime, ))
    tagg_out, shape_out = layer(x)

    assert tagg_out.shape == expected_shape, 'Test failed, expected output shape {}, but got {}'.format(
        expected_shape, tagg_out.shape
    )
    print('Test passed, expected output shape {}, got {}'.format(
        expected_shape, tagg_out.shape
    ))


def scot_embeddings():
    nbatch = 1
    fn_config = os.path.join( os.environ['SCIFM_PATH'], 'poseidon_pretrained_models', 
                             'Poseidon-T', 'config.json')
    config_data = json.load( open(fn_config, 'r'))
    config  = model.ScOTConfig(**config_data)
    layer = model.ScOTEmbeddings(config)
    x = torch.randn((nbatch, config.num_channels, config.image_size, config.image_size,))
    t = torch.randn((nbatch,1))

    patch_out, embed_dim = layer( pixel_values = x ,
                      time = t)
    test = 1


def dev_build_model():
    img_size        = (128,128)
    patch_size      = (4,4)
    out_channels    = 48
    in_channels     = 4
    ntime           = 10
    nbatch          = 1
    embed_dim       = out_channels
    activation      = 'gelu'
    time_agg_type   = 'exp_mlp'
    n_patches       = tuple([int(img_size[i] / patch_size[i]) for i  in np.arange(len(img_size)) ])

    fn_config = os.path.join( os.environ['SCIFM_PATH'], 'poseidon_pretrained_models', 
                             'Poseidon-T', 'config.json')
    config_data = json.load( open(fn_config, 'r'))

    config = model.ScOTConfigTimeAggregate(num_timesteps    = ntime,
                                           time_agg_type    = 'mlp',
                                           **config_data,)



    # config = model.ScOTConfigTimeAggregate( image_size          = img_size[0],
    #                                         patch_size          = patch_size[0],
    #                                         num_channels        = in_channels,            # number of input channels
    #                                         num_out_channels    = embed_dim,           # number of channels output from first cnn embedding layer
    #                                         num_timesteps       = 10,           # number of input timesteps
    #                                         time_agg_type       = 'mlp',        # 'mlp' or 'exp_mlp', whether or not to include exponential weighting on 
    #                                         embed_dim           = embed_dim,           # embedding dimension output from patching/time aggregator
    #                                         depths              = [2, 2, 6, 2],
    #                                         num_heads           = [3, 6, 12, 24],
    #                                         skip_connections    = [True, True, True],
    #                                         window_size         = 7,
    #                                         mlp_ratio           = 4.0,
    #                                         qkv_bias            = True,
    #                                         hidden_dropout_prob = 0.0,
    #                                         attention_probs_dropout_prob = 0.0,
    #                                         drop_path_rate      = 0.1,
    #                                         hidden_act          = "gelu",
    #                                         use_absolute_embeddings = False,
    #                                         initializer_range   = 0.02,
    #                                         layer_norm_eps      = 1e-5,
    #                                         p                   = 1,  # for loss: 1 for l1, 2 for l2
    #                                         channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
    #                                         residual_model="convnext",  # "convnext" or "resnet"
    #                                         use_conditioning=False,
    #                                         learn_residual=False,)
    net = model.ScOTDPOTTimeAggregate(config)
    print(net)

    x = torch.randn((nbatch, in_channels, img_size[0], img_size[1], ntime,))
    t = torch.randn((nbatch,1))
    out_ = net(x,t)
    print(out_.shape)

    test = 1


    
if __name__ == '__main__':
    # dev_forward_pass()
    # scot_embeddings()
    dev_build_model()
    # test_layer_square_mlp()
    # test_layer_square_exp_mlp()
