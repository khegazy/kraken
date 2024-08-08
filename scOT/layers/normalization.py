import torch
from torch import nn

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, time):
        return super().forward(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Linear(1, dim)
        self.bias = nn.Linear(1, dim)

    def forward(self, x, time):
        #print("conditional layer norm!!!!!!", x.shape, time.shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x**2).mean(dim=-1, keepdim=True) - mean**2
        x = (x - mean) / (var + self.eps).sqrt()
        time = time.unsqueeze(-1).type_as(x)
        weight = self.weight(time)
        bias = self.bias(time)
        if not (x.dim() == 3 and time.dim() == 3):
            weight = weight.unsqueeze(-2)
            bias = bias.unsqueeze(-2)
        if x.dim() == 5:
            weight = weight.unsqueeze(-2)
            bias = bias.unsqueeze(-2)
        #print(x.shape, weight.shape, bias.shape, time.shape)
        return weight * x + bias
        #conditional layer norm!!!!!! torch.Size([9, 8, 1024, 96]) torch.Size([9, 8])

    def __forward(self, x, time):
        #print("conditional layer norm!!!!!!", x.shape, time.shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x**2).mean(dim=-1, keepdim=True) - mean**2
        x = (x - mean) / (var + self.eps).sqrt()
        time = time.unsqueeze(-1).type_as(x)
        weight = weight.unsqueeze(-2).type_as(x)
        bias = bias.unsqueeze(-2).type_as(x)
        if x.dim() == 5:
            weight = weight.unsqueeze(-2)
            bias = bias.unsqueeze(-2)
        #print(x.shape, weight.shape, bias.shape, time.shape)
        return weight * x + bias
