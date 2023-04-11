"""
Author: Xuande Feng
Contact: xf2219@columbia.edu
Date: March 2022
"""
from data_utils.DataLoader_cls import PC3DDataset
import argparse
import numpy as np
import os
import torch
import wandb
import datetime
import sys
import provider
import warnings
from models.vn_transformer_cls import get_model
from tqdm import tqdm
import time

warnings.simplefilter(action='ignore', category=FutureWarning)



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', default='vn_transformer_cls',
                        help='Model name [default: vn_dgcnn_cls]',
                        choices=['pointnet_cls', 'vn_pointnet_cls', 'dgcnn_cls', 'vn_dgcnn_cls', 'vn_transformer_cls'])
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=2000, type=int,
                        help='Number of epoch in training [default: 250]')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='Decay rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Pptimizer for training [default: SGD]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=256,
                        help='Point Number [default: 1024]')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Point Number [default: 1024]')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Point Number [default: 1024]')
    parser.add_argument('--num_features', type=int, default=512,
                        help='Feature Number [default: 1024]')
    parser.add_argument('--hidden', type=int, default=512,
                        help='Point Number [default: 1024]')
    parser.add_argument('--num_class', type=int, default=40,
                        help='Point Number [default: 1024]')

    parser.add_argument('--kernel', action='store_true', default=False,
                        help='Whether to use performer')
    parser.add_argument('--antithetic', action='store_true', default=False,
                        help='Whether to use antithetic sampling')
    parser.add_argument('--num_random', type=int, default=20,
                        help='Number of random features')


    parser.add_argument('--data_name', type=str, default='ModelNet40',
                        help='Point Number [default: 1024]')
    parser.add_argument('--save_dir', type=str, default='/mnt/c/users/sssak/desktop/models/VNPerformer',
                        help='Experiment root [default: saved_models]')
    parser.add_argument('--restore', type=str, default=None,
                        help='Model path to restore [default: None]')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--name', type=str, default='4090',
                        help='device name')


    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--rot', type=str, default='aligned',
                        help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--pooling', type=str, default='mean',
                        help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--n_knn', default=20, type=int,
                        help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')


    return parser.parse_args()


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).to(torch.float32)


def test(model, loader, criterion, num_class=40):
    r = RandomRotation()
    loss_epoch = 0
    num_correct = 0
    count = 0
    class_acc = np.zeros((num_class,3))

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        
        trot = None
        # if args.rot == 'z':
        #         trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
        # elif args.rot == 'so3':
        #     trot = Rotate(R=random_rotations(points.shape[0]))
        # if trot is not None:
        #     points = trot.transform_points(points)
        #target = target[:, 0]
        points = r(points)
        points = points.transpose(2, 1)
        points = points.unsqueeze(dim=1)
        points, target = points.cuda(), target.squeeze().cuda()

        classifier = model.eval()
        pred, _ = classifier(points)

        test_loss = criterion(pred, target.long().squeeze())
        loss_epoch += test_loss / len(loader)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item()
            class_acc[cat, 1] += points[target == cat].shape[0]

        num_correct += pred_choice.eq(target.long().data).cpu().sum()
        count += points.shape[0]
    epoch_acc = num_correct / count

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])

    return epoch_acc, class_acc


def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))


    '''DATA LOADING'''
    print('Load dataset ...')
    TRAIN_DATASET = PC3DDataset(args, split='train')
    TEST_DATASET = PC3DDataset(args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2)

    # TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    #MODEL = importlib.import_module(args.model)
    num_class = args.num_class
    classifier = get_model(args, num_class, normal_channel=args.normal)
    # criterion = MODEL.get_loss().cuda()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.2).cuda()


    if args.kernel:
        model_name = f'Performer_{args.data_name}_{args.name}_{args.num_layers}_{args.num_heads}_{args.num_point}_{args.batch_size}_{args.num_features}_{args.hidden}'
    else:
        model_name = f'Transformer_{args.data_name}_{args.name}_{args.num_layers}_{args.num_heads}_{args.num_point}_{args.batch_size}_{args.num_features}_{args.hidden}'
    save_path = os.path.join(args.save_dir, model_name + '.pt')

    if args.restore is not None:
        classifier.load_state_dict(torch.load(args.restore))
        print('Restore model')
    classifier = classifier.to(args.device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer =='SGD':
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.decay_rate)
    factor = len(trainDataLoader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=20*factor)
    global_epoch = 0
    global_step = 0
    best_test_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []


    '''TRANING'''
    print('Start training...')
    for epoch in range(args.epoch):
        epoch_loss = 0
        correct = 0
        count = 0
        grad_acc = 2
        print(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')

        for batch_id, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
            points, target = data
            
            #points = points.data.numpy()
            #points = provider.point_dropout(points)
            #points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            #points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            #points = torch.Tensor(points)
            target = target.unsqueeze(-1)

            # B, N, 3 --> B, 3, N
            points = points.transpose(2, 1)
            points = points.unsqueeze(dim=1)## B 1 3 N
            points, target = points.cuda(), target.cuda()

            wandb.log({'lr': optimizer.param_groups[0]['lr']}, commit=False)
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long().squeeze())
            loss /= grad_acc

            pred_choice = pred.data.max(1)[1]
            correct += pred_choice.eq(target.long().squeeze()).cpu().sum()
            epoch_loss += loss.item()
            count += points.size(0)

            loss.backward()
            if (batch_id + 1) % grad_acc == 0 or batch_id + 1 == len(trainDataLoader):
                optimizer.step()

            if batch_id % 25 == 0:
                wandb.log({'Train Batch Loss': epoch_loss/(batch_id+1)}, commit=True)

            scheduler.step()
            global_step += 1
        acc_epoch = correct / count
        epoch_loss /= len(trainDataLoader)
        wandb.log({'Train Epoch Loss': epoch_loss}, commit=False)
        wandb.log({'Train Acc': acc_epoch}, commit=False)
        print(f'Train Epoch Accuracy: {acc_epoch}')
        print(f"Train Epoch Loss: {epoch_loss}")

        with torch.no_grad():
            classifier.eval()

            test_start = time.time()
            test_acc, class_acc = test(classifier, testDataLoader, criterion, num_class)
            inf_time = time.time() - test_start
            print(f'TEST (inference) time: {inf_time}s')
            wandb.log({'Inference Time': inf_time}, commit=False)
            if test_acc >= best_test_acc:
                torch.save(classifier.state_dict(), save_path)
                best_test_acc = test_acc

            if class_acc >= best_class_acc:
                best_class_acc = class_acc
            print(f'Test Instance Accuracy: {test_acc}, Class Accuracy: {class_acc}')
            print(f'Best Instance Accuracy: {best_test_acc}, Class Accuracy: {best_class_acc}')

            global_epoch += 1
            wandb.log({'Test Acc': test_acc}, commit=False)
            wandb.log({'Test Best Acc': best_test_acc}, commit=False)
            wandb.log({'Test Best Class Acc': best_class_acc}, commit=False)

    print('End of training...')


if __name__ == '__main__':
    args = parse_args()
    args.num_class = 40 if args.data_name == "ModelNet40" else 15

    # logg to wandb
    if args.kernel:
        k = 'Performer_'
    else:
        k = 'Transformer_'
    name = k + f'{args.data_name}_{args.name}_{args.num_layers}_{args.num_heads}_{args.num_point}_{args.batch_size}_{args.num_features}_{args.hidden}'
    project = 'ModelNet40_VN' if args.data_name == 'ModelNet40' else 'ScanObjectNN_VN'
    wandb.init(project=project, name=name, config=args, entity='orcs4529')
    wandb.save('*.txt')
    main(args)

