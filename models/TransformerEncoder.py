from models.vn_layers import *
import torch
import torch.nn as nn
from math import *
from models.kernelization import *
from models.utils.vn_dgcnn_util import get_graph_feature


class TransformerEncoder(nn.Module):

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.heads = args.num_heads
        self.num_features = args.num_features
        self.channels_per_head = self.num_features // self.heads

        self.ln_attn = VNLayerNorm(self.channels_per_head * self.heads)

        self.mlp = nn.Sequential( VNLinearAndLeakyReLU(self.heads * self.channels_per_head, args.hidden, dim=4, negative_slope=0.2,use_batchnorm = None),
                                  VNLinear(args.hidden, self.heads * self.channels_per_head))

        self.ln_mlp = VNLayerNorm(self.heads * self.channels_per_head)

        self.wq = VNLinear(self.num_features, self.channels_per_head * self.heads)
        self.wk = VNLinear(self.num_features, self.channels_per_head * self.heads)
        self.wv = VNLinear(self.num_features, self.channels_per_head * self.heads)

        self.kernel = args.kernel
        self.antithetic = args.antithetic
        self.num_random = args.num_random
        self.kernel_channel = self.channels_per_head * 3
        if self.kernel:
            if self.antithetic:
                w = gaussian_orthogonal_random_matrix(nb_rows=self.num_random // 2, nb_columns=self.kernel_channel)
                self.w = torch.cat([w, -w], dim=0).cuda()
            else:
                self.w = gaussian_orthogonal_random_matrix(nb_rows=self.num_random, nb_columns=self.kernel_channel)


    def forward(self, x):
        '''
        q: query B, C, 3, N
        k: key   B, C, 3, N
        v: value B, C, 3, N
        '''

        skip = x # skip is defined as input
        B, C, _, N = x.shape

        # q(k) --> B, C//H * H, 3, N
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # --> B, H, N, C // H * 3
        q = torch.stack(q.transpose(1, -1).split(self.channels_per_head, -1), 3)  ## B N 3 C --> B N 3 H C/H
        k = torch.stack(k.transpose(1, -1).split(self.channels_per_head, -1), 3)
        v = torch.stack(v.transpose(1, -1).split(self.channels_per_head, -1), 3)
        # v  --> B, H, N, C//H * 3
        q = q.permute(0, 3, 1, -1, 2)
        k = k.permute(0, 3, 1, -1, 2)
        v = v.permute(0, 3, 1, -1, 2)
        q = q.flatten(-2)
        k = k.flatten(-2)

        # separated attention
        # B, H, N, N
        if self.kernel:
            if self.training:
                if self.antithetic:
                    w = gaussian_orthogonal_random_matrix(nb_rows=self.num_random // 2, nb_columns=self.kernel_channel)
                    w = torch.cat([w, -w], dim=0).cuda()
                else:
                    w = gaussian_orthogonal_random_matrix(nb_rows=self.num_random, nb_columns=self.kernel_channel)

            else:
                w = self.w

            k = softmax_kernel(data=k, projection_matrix=w, is_query=False)
            q = softmax_kernel(data=q, projection_matrix=w, is_query=True)
            out = compute_attn(q, k, v).contiguous()

        else:
            div = 1 / sqrt(q.shape[-1])
            attn = torch.einsum('...nc, ...mc->...nm', q, k)
            attn *= div
            attn = attn.softmax(-1)

            # B, H, N, C//H, 3 --> B, C//H * H, 3, N
            out = torch.einsum('...nm, ...mcp->...ncp', attn, v)  # B, H, N, C//H, 3

        out = out.permute(0, -1, 2, 3, 1).contiguous()
        out = out.view(B, -1, N, self.channels_per_head*self.heads)  # B, 3, N, C//H * H
        out = out.permute(0, -1, 1, 2)

        # add and norm

        out = self.ln_attn(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B, N, C, 3
        out += skip

        skip = out  # B C, 3, N

        # MLP
        out = self.mlp(out)  # B, C, 3, N

        out = self.ln_mlp(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # B, N, C, 3 --> B, C, 3, N
        out += skip  # B, C, 3, N

        return out

