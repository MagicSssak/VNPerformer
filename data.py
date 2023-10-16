import torch
import numpy as np
from numpy import linalg as LA
import knn_cuda


# given pc = torch.tensor(N, 3)

# # Point normalization
# N = 1024  # Num of points
# pc = torch.random(N, 3)
# mean, std = torch.mean(pc, 0), torch.std(pc, 0)
# normalize = (pc - mean) / std

# # Point selection
# candidate = torch.sqrt((pc ** 2).sum(-1))  # N, 1


# # select 2+1+1

# a computer


def farthest_n_points(xyz, npoint=3):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    dist = xyz.norm(dim=-1)
    _, indices = torch.sort(dist)  # B, N
    indices = indices[:, -npoint:].unsqueeze(-1)

    idx_base = torch.arange(
                            0, B).view(-1, 1, 1) * N
    idx_base = idx_base.to(device)
    indices = indices + idx_base

    indices = indices.view(-1)
    groundings = xyz.view(B * N, -1)[indices, :].view(B, -1, C)
    return groundings



class point:
    """
    Input: A batch of points(torch.Tensor): B, 1, 3
    Output: A batch of coordinates(torch.Tensor): B, 3, 3
    """

    def __init__(self, *args, **kargs):

        try:
            lst = kargs['list']
            self.x = lst[:, 0]
            self.y = lst[:, 1]
            self.z = lst[:, 2]
        except KeyError as e:
            # print(f'No list in the kargs: {e}')
            pass

        if 'list' not in kargs.keys():
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]

    def getList(self):
        return torch.stack([self.x, self.y, self.z]).T

    def get_slice(self, num_slice: int):
        return 1 / num_slice * self.getList()


def compute_coordinates(point1, point2, point3):
    """
    compute the coordinate and corresponding middle points
    """

    direct_x = point1.getList() / point1.getList().norm(dim=-1).unsqueeze(-1)
    direct_x = point(direct_x[:, 0], direct_x[:, 1], direct_x[:, 2])

    distance_lines = point2.getList().norm(dim=-1).unsqueeze(-1) \
                     * (torch.einsum('bs,bs->b', point1.getList(), point2.getList()).unsqueeze(-1)
                        /
                        (point1.getList().norm(dim=-1).unsqueeze(-1) *
                         point2.getList().norm(dim=-1).unsqueeze(-1)))
    # distance_lines = point2.getList().norm(-1) * (torch.dot(point1.getList(), point2.getList()) /
    #                                               (point1.getList().norm(-1) * point2.getList().norm(-1)))
    foot1 = point1.getList() * distance_lines / point1.getList().norm(dim=-1).unsqueeze(-1)
    foot_1 = point(foot1[:, 0], foot1[:, 1], foot1[:, 2])

    direct_y = (point2.getList() - foot_1.getList()) / (point2.getList() - foot_1.getList()).norm(dim=-1).unsqueeze(-1)
    direct_y = point(direct_y[:, 0], direct_y[:, 1], direct_y[:, 2])
    a_times_b = point(direct_x.y * direct_y.z - direct_x.z * direct_y.y,
                      direct_x.z * direct_y.x - direct_x.x * direct_y.z,
                      direct_x.x * direct_y.y - direct_x.y * direct_y.x)

    flag = 2 * \
           ((a_times_b.x * point3.x + a_times_b.y * point3.y + a_times_b.z * point3.z >= 0).long() - 1 / 2).unsqueeze(
               -1)
    direct_z = point(list=flag * a_times_b.getList())

    return direct_x, direct_y, direct_z


def split_layers(coor_x, coor_y, coor_z, split: int):
    unit_x = coor_x.getList() / split
    unit_y = coor_y.getList() / split
    unit_z = coor_z.getList() / split
    # there would be 2*split + 1 layers
    # for each layer there should be 2*split+1 bars
    # for each bar there should be 2*split+1 centers
    pc = []
    for i in range(-split, split + 1):
        layer = []
        for j in range(-split, split + 1):
            bar = []
            for k in range(-split, split + 1):
                center = k * unit_z + j * unit_y + i * unit_x
                bar.append(center)
            bar = torch.stack(bar)
            layer.append(bar)
        layer = torch.stack(layer)
        pc.append(layer)

    return torch.stack(pc)


def wrap_knn(centers, ref, min_dist, k=10, transpose_mode=True):
    min_dist = min_dist
    knn = knn_cuda.KNN(k, transpose_mode)

    ref = ref .to('cuda')  # B, N, 3
    centers = centers.to('cuda')  # B, N, 3
    B, N, _ = ref.shape
    centers = centers.view(B, -1, _)  # B, n, 3
    n = centers.shape[1]
    dist, indices = knn(ref, centers)  # B, n, k
    dist = (dist <= min_dist).unsqueeze(-1)  # B, n, k, 1

    idx_base = torch.arange(
        0, B).view(-1, 1, 1) * N
    idx_base = idx_base.to(indices.device)
    indices = indices + idx_base
    indices = indices.view(-1)
    neighborhood = ref.view(B * N, -1)[indices, :]
    neighborhood = neighborhood.view(B, n, k, -1)
    neighborhood *= dist

    return neighborhood  # B, x, x, x, k, 3


# if __name__ == '__main__':
#     pc = torch.normal(0, 1, [32, 1024, 3]).to('cuda')
#     points = farthest_n_points(pc, 3)
#     print(pc[:, 0, 0].shape, points.shape)
#
#     p1 = point(pc[:, 0, 0], pc[:, 0, 1], pc[:, 0, 2])
#     p2 = point(pc[:, 1, 0], pc[:, 1, 1], pc[:, 1, 2])
#     p3 = point(pc[:, 2, 0], pc[:, 2, 1], pc[:, 2, 2])
#     coor_1, coor_2, coor_3 = compute_coordinates(p1, p2, p3)
#     centers = split_layers(coor_1, coor_2, coor_3, split=3)
#     n = wrap_knn(centers, pc, min_dist=1/3, k=10)
#     try_ = (n==0)
#     print(try_.sum())
#     print('done')
