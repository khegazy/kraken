import torch
from torch import nn
import torch.nn.functional as F
from scOT.layers.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from scOT.layers.self_attention import FullAttention, AttentionLayer
from scOT.layers.normalization import ConditionalLayerNorm
from scOT.layers.embed_transformer import ContinuousTemporalEmbedding


class TransformerTS(nn.Module):
    def __init__(self, configs, embed_dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        #self.output_attention = configs.output_attention
        self.output_attention = False

        # Temporal Embedding
        self.embedding = ContinuousTemporalEmbedding(embed_dim)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.hidden_dropout_prob,
                                      output_attention=self.output_attention), embed_dim, configs.num_res_attn_heads),
                    embed_dim,
                    embed_dim,
                    dropout=configs.hidden_dropout_prob,
                    activation="relu"
                ) for l in range(configs.num_residual_layers)
            ],
            norm_layer=ConditionalLayerNorm(embed_dim)
            #norm_layer=torch.nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x, time):
        #print("INPUT ATTN", x.shape, time.shape, self.embedding(time).shape)
        embeds = x + self.embedding(time)
        enc_out, attns = self.encoder(embeds, time, attn_mask=None)
        #print("OUTPUT ATTN", enc_out.shape)
        return enc_out


