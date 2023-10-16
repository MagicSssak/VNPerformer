import sys
import numpy as np
import torch
import os
import pickle
import time
import h5py
import random


DTYPE = np.float32

from .provider import random_scale_point_cloud, jitter_point_cloud, shift_point_cloud, normalize_data


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


def far_sampling(data, n_points):
    N, P, __ = data.shape

    r = torch.zeros(N, n_points, __)
    for i in range(N):
        source = data[i] ## P 3
        id = np.random.randint(low=0, high=P, size=1)
        sample_id = [id.item()]

        for j in range(n_points-1):

            tmp = source[sample_id].reshape(len(sample_id),1,3)
            dist = ((source - tmp)**2).sum(-1) ## len(sample_id) P
            dist = dist.min(0) ## P
            far_id = np.argmax(dist).item()
            sample_id.append(far_id)


        sample_id = np.array(sample_id) ## n_points farthest id
        sample_id = torch.tensor(sample_id).to(torch.long)

        r[i] = torch.tensor(source)[sample_id]

    return r


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
                filename = '/home/sssak/h5_files/ModelNet40/ModelNet40_trainset.h5'
            else:
                filename = '/home/sssak/h5_files/split1_nobg/training_objectdataset.h5'
        else:
            if self.data_name == 'ModelNet40':
                filename = '/home/sssak/h5_files/ModelNet40/ModelNet40_testset.h5'
            else:
                filename = '/home/sssak/h5_files/split1_nobg/test_objectdataset.h5'


        # data.shape => 11481, 2048, 3
        # label.shape => 11481, 2048,
        points, label = loadh5(filename)  # data = f['data'][:]
        points = normalize_data(points)
        # label = f['label'][:]


        root = os.path.split(filename)[0]

        self.test_point = self.n_points


        if self.split == 'test' and self.n_points < 2048:

            try:
                hf = h5py.File(os.path.join(root,f'fps_{self.test_point}.h5'), 'r')
                points = hf['points'][:]
                print('test points loaded')
            except:
                points = far_sampling(points, self.test_point)
                f = h5py.File(os.path.join(root,f'fps_{self.test_point}.h5'), 'w')
                f.create_dataset('points', data=points)
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
        P, D = x_0.shape

        if self.split == 'train':
        
            #x_0 = x_0.unsqueeze(0).numpy()
            #x_0 = jitter_point_cloud(x_0, 0.01,0.05)
            #x_0 = random_scale_point_cloud(x_0,0.9,1.1)
            #x_0 = torch.tensor(x_0)
            #x_0 = x_0.squeeze(0).to(torch.float32)


            index = torch.LongTensor(random.sample(range(P), self.n_points))


            x_sample = x_0[index]
        else:
            x_sample = x_0

        label_0 = self.data['label'][idx]
        # label = np.zeros(self.FLAGS.num_class)
        # label[self.data['label'][idx]] = 1
        # label_0 = torch.tensor(label.astype(DTYPE))

        data = (x_sample - x_sample.mean(0,keepdims = True)).detach()


        return data, label_0
