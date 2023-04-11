import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.TransformerEncoder import *


class get_model(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.num_points = args.num_point
        self.heads = args.num_heads
        self.channels_per_head = args.num_features // args.num_heads
        self.hidden = args.hidden
        # input batch is B, 1, 3, N

        # Transformer Embeddings for Q, K and V
        self.tokenizer = VNLinear(1, args.num_features)


        # Transformer
        T = []
        for _ in range(args.num_layers):
            T.append(TransformerEncoder(args))
        self.transformer = nn.ModuleList(T)

        # Invariant Block
        self.invariant = VNLinearAndLeakyReLU(self.channels_per_head * self.heads, 3, use_batchnorm=None)

        # Decoder
        # self.MLP = VNLinearLeakyReLU(self.channels_per_head * self.heads * 3, self.hidden)
        self.MLP = nn.Sequential(nn.Linear(self.channels_per_head * self.heads * 3, self.hidden),#, bias=False),
                                 nn.LeakyReLU(),
                                 nn.Linear(self.hidden, num_class)) #, bias=False))

    def forward(self, x):
        
        x = self.tokenizer(x)  # B, C, 3, N
        B, C, _, N = x.shape
        
        for layers in self.transformer:
            x = layers(x)  # B, C, 3, N

        inv = self.invariant(x)  # B 3 3 N
        # x --> B C 3 N inv --> B 3 3 N
        x = torch.einsum('...cp, ...mp->...cm', x.permute(0, -1, 1, 2), inv.permute(0, -1, 1, 2))  # B, N, C, 3
        x = torch.flatten(x, start_dim=-2)  # B, N, C * 3
        x = x.mean(1)  # B C*3
        x = self.MLP(x)

        trans_feat = None
        return x, trans_feat

