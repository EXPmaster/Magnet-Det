import argparse
import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from unet import UNet
from dataset import train_test_split, MagnetDataset


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count if self._count != 0 else 0

    def getval(self):
        return self._avg

    def __str__(self):
        if not hasattr(self, 'val'):
            return 'None.'
        return str(self.getval())


def train(epoch, args, loader, model, loss_fn1, loss_fn2, optimizer):
    model.train()
    for itr, (data, data_txt, label, label_txt) in enumerate(loader):
        data, data_txt, label, label_txt =\
        data.to(args.device), data_txt.to(args.device), label.to(args.device), label_txt.to(args.device)
        data = data.unsqueeze(1)
        optimizer.zero_grad()
        predict_img, predict_txt = model(data, data_txt)
        loss1 = loss_fn1(predict_img, label)
        loss2 = loss_fn2(predict_txt, label_txt)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        if itr % 100 == 0:
            print(f'Epoch: {epoch}, train img loss: {loss1.item()}, train txt loss: {loss2.item()}')


@torch.no_grad()
def evaluate(epoch, args, loader, model, metric_fn):
    model.eval()
    metric_img = AverageMeter()
    metric_txt = AverageMeter()
    for itr, (data, data_txt, label, label_txt) in enumerate(loader):
        data, data_txt, label, label_txt =\
        data.to(args.device), data_txt.to(args.device), label.to(args.device), label_txt.to(args.device)
        data = data.unsqueeze(1)
        predict_img, predict_txt = model(data, data_txt)
        loss_img = metric_fn(predict_img.flatten(1), label.flatten(1))
        loss_txt = torch.abs(predict_txt - label_txt).mean()
        metric_img.update(loss_img.item())
        metric_txt.update(loss_txt.item())
    print(f'Average evaluation Error => IMG: {metric_img.getval()}, TXT: {metric_txt.getval()}')
    print()
    return metric_img.getval()


@torch.no_grad()
def save_results(args, loader, model):
    print('Saving...')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    model.eval()
    outputs = []
    targets = []
    for itr, (data, data_txt, label, label_txt) in enumerate(loader):
        if itr >= 100: break
        data, data_txt = data.to(args.device), data_txt.to(args.device)
        data = data.unsqueeze(1)
        predict_img, predict_txt = model(data, data_txt)
        predict = predict_img.cpu().numpy()
        outputs.append(predict)
        targets.append(label)
    outputs = np.concatenate(outputs, 0)
    targets = np.concatenate(targets, 0)
    for idx, (data, label) in enumerate(zip(outputs, targets)):
        # data = data.transpose(1, 2, 0)
        # label = label.transpose(1, 2, 0)
        data = 255 - data * 255.0
        label = 255 - label * 255.0
        cv2.imwrite(os.path.join(args.output_path, f'{idx}_predict.png'), data)
        cv2.imwrite(os.path.join(args.output_path, f'{idx}_label.png'), label)
    

def main(args):
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(args.device)
    loss_fn1 = nn.SmoothL1Loss()  # nn.L1Loss()  # nn.MSELoss() # nn.BCELoss()
    loss_fn2 = nn.MSELoss()
    metric_fn = nn.L1Loss()  # nn.MSELoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    trainset, testset = train_test_split(args.data_path)
    trainset = MagnetDataset(trainset, imgsz=args.imgsz)
    testset = MagnetDataset(testset, imgsz=args.imgsz)
    print(f'Data length: {len(trainset)}, {len(testset)}')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    best_metric = 0.015
    for i in range(args.epochs):
        train(i, args, train_loader, model, loss_fn1, loss_fn2, optimizer)
        metric = evaluate(i, args, test_loader, model, metric_fn)
        scheduler.step(metric)

        if metric < best_metric:
            best_metric = metric
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if not os.path.exists(args.weight_path):
                os.mkdir(args.weight_path)
            torch.save(ckpt, os.path.join(args.weight_path, 'best.pt'))

    save_results(args, test_loader, model)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data-path', default='./data/dataset_new', type=str)
    parser.add_argument('--output-path', default='./data/outputs', type=str)
    parser.add_argument('--weight-path', default='./weights', type=str)
    parser.add_argument('--epochs', '-e', type=int, default=40, help='Number of epochs')
    parser.add_argument('--imgsz', type=tuple, default=(512, 512), help='Image size')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', '-l', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of channels')
    parser.add_argument('--gpus', type=str, default='0')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)

    main(args)
