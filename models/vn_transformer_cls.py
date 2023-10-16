import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.TransformerEncoder import *
from einops import rearrange
from .utils.vn_dgcnn_util import get_graph_feature
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import random
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class get_model(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.num_points = args.num_point
        self.heads = args.num_heads
        self.channels_per_head = args.num_features // args.num_heads
        self.hidden = args.hidden
        self.upsampler = torch.nn.Linear(1, args.num_features)
        # input batch is B, 1, 3, N
        # Transformer Embeddings for Q, K and V

        T = []
        C = []
        P = []
        counter = 0
        self.k = args.n_knn
        self.factor = 1
        use_batchnorm = None
        for _ in range(args.num_layers):
            if counter == 0:

                C.append(nn.Sequential(VNLinear(self.factor * self.k, args.num_features)
                                       ))

            else:

                C.append(nn.Sequential(VNLinearLeakyReLU(args.num_features, args.num_features)))
            T.append(TransformerEncoder(args))
            # P.append(nn.Sequential(VNLinear(2,1)))
            counter += 1

        self.transformers = nn.ModuleList(T)
        self.conv = nn.ModuleList(C)
        # self.pointconv = nn.ModuleList(P)

        # Invariant Block
        self.inv_dim = 3
        self.invariant = nn.Sequential(
            VNLinearAndLeakyReLU(self.channels_per_head * self.heads, self.channels_per_head * self.heads, dim=4,
                                 negative_slope=0.2, use_batchnorm=True),
            VNLinear(self.channels_per_head * self.heads, self.inv_dim),
            )

        self.MLP = nn.Sequential(
            nn.Linear(self.channels_per_head * self.heads * self.inv_dim, self.channels_per_head * self.heads),
            # , bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.channels_per_head * self.heads),
            nn.Linear(self.channels_per_head * self.heads, num_class))  # , bias=False))

    def down_sample(self, x, num_sample):
        B, C, _, N = x.shape
        out = torch.zeros(B, C, _, num_sample).cuda()
        for b in range(B):
            index = torch.LongTensor(random.sample(range(N), num_sample))
            out[b] = x[b][:, :, index]

        return out

    def forward(self, x):
        B, D, N, K = x.shape
        i = 0
        x = x.unsqueeze(1)
        x = rearrange(x, 'b c d n k-> b (c k) d n')  # d=3
        for layer in self.transformers:

            x = self.conv[i](x)
            '''
            if i==0:
                x = self.down_sample(x,256)
                N = 256
            '''
            # x = self.pool[i](x)
            x = layer(x)  # B, C, 3, N
            i += 1

        inv = self.invariant(x)  # B 3 3 N
        # x --> B C 3 N inv --> B 3 3 N
        x = torch.einsum('...cp, ...mp->...cm', x.permute(0, -1, 1, 2), inv.permute(0, -1, 1, 2))  # B, N, C, 3

        # x_max = x.view(B,N,-1).max(1).values  # B C*3
        # x_mean = x.view(B,N,-1).mean(1)
        # x_min = x.view(B,N,-1).min(1).values
        # x = torch.cat([x_max,x_min,x_mean],dim=-1)
        x = x.view(B, N, -1).mean(1)  # B C*3
        # x = self.pool(x.view(B,N,-1))
        # x = x.view(B,-1)
        x = self.MLP(x)

        trans_feat = None

        return x, trans_feat
