import sys
import numpy as np
import torch
import os
import pickle
import time
import h5py
import random


DTYPE = np.float32

#from .provider import random_scale_point_cloud, jitter_point_cloud, rotate_perturbation_point_cloud


def loadh5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).to(torch.float32)


class PC3DDataset(torch.utils.data.Dataset):
    def __init__(self, FLAGS, split):
        """Create a dataset object"""

        # Data shapes:
        #  points :: [samples, points, 3]
        #  label  :: [samples, 1]

        self.FLAGS = FLAGS
        self.split = split
        self.n_points = FLAGS.num_point
        self.data_name = FLAGS.data_name


        assert split in ["test", "train"]
        if split == "train":
            if self.data_name == 'ModelNet40':
                filename = '/mnt/c/users/sssak/desktop/h5_files/ModelNet40/ModelNet40_trainset.h5'
            else:
                filename = '/mnt/c/users/sssak/desktop/h5_files/split1_nobg/training_objectdataset.h5'
        else:
            if self.data_name == 'ModelNet40':
                filename = '/mnt/c/users/sssak/desktop/h5_files/ModelNet40/ModelNet40_testset.h5'
            else:
                filename = '/mnt/c/users/sssak/desktop/h5_files/split1_nobg/test_objectdataset.h5'

        # data.shape => 11481, 2048, 3
        # label.shape => 11481, 2048,
        points, label = loadh5(filename)  # data = f['data'][:]
        # label = f['label'][:]

        if self.split == 'test':
            # points = far_sampling(points, self.n_points)
            print('test points sampled')

        points, label = torch.tensor(points).to(torch.float), torch.tensor(label)
        if self.split == 'test':
            r = RandomRotation()
            points = r(points)

        data = {'points': points,
                'label': label}

        self.data = data
        self.len = data['points'].shape[0]

    def __len__(self):
        return self.len



    def __getitem__(self, idx):

        # select a start and a target frame

        x_0 = self.data['points'][idx]  # 2048 * 3
        #x_0 = pc_normalize(x_0)
        # if self.split == 'train':
        #
        #     x_0 = x_0.unsqueeze(0).numpy()
        #     x_0 = jitter_point_cloud(x_0, 0.005)
        #     x_0 = random_scale_point_cloud(x_0,0.9,1.1)
        #     x_0 = torch.tensor(x_0)
        #     x_0 = x_0.squeeze(0).to(torch.float32)

        P, D = x_0.shape

        index = torch.LongTensor(random.sample(range(P), self.n_points))

        x_sample = x_0[index]
        label_0 = self.data['label'][idx]
        # label = np.zeros(self.FLAGS.num_class)
        # label[self.data['label'][idx]] = 1
        # label_0 = torch.tensor(label.astype(DTYPE))
        data = x_sample

        return data, label_0
