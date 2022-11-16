import argparse
import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from dataset import train_test_split, MagnetDataset


@torch.no_grad()
def save_results(args, loader, model):
    print('Saving...')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    model.eval()
    outputs = []
    targets = []
    for itr, (data, label) in enumerate(loader):
        if itr >= 100: break
        data = data.to(args.device)
        data = data.unsqueeze(1)
        predict = model(data).cpu().squeeze().numpy()
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
    

@torch.no_grad()
def infer(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(args.device)
    model.eval()
    weights = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(weights['model'])
    trainset, testset = train_test_split(args.data_path)
    print('Start inference...')
    for idx, name in enumerate(testset):
        if idx > 100: break
        data_name = os.path.join(args.data_path, name + '_Ori.png')
        label_name = os.path.join(args.data_path, name + '_ResCo.png')
        img = cv2.imread(data_name, 0)[50:-50, 50:-50]
        org_img = img.copy()
        label = cv2.imread(label_name)[50:-50, 50:-50]
        label = cv2.resize(label, args.imgsz)
        img = cv2.resize(img, args.imgsz) / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = img[None, None, :].to(args.device)
        predict = model(img).cpu().squeeze().numpy()
        predict = (predict / predict.max() * 255).astype(np.uint8)
        predict = cv2.applyColorMap(predict, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.output_path, f'{idx}_predict.png'), predict)
        cv2.imwrite(os.path.join(args.output_path, f'{idx}_label.png'), label)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data/dataset', type=str)
    parser.add_argument('--output-path', default='./data/outputs_infer', type=str)
    parser.add_argument('--weight-path', default='./weights/best.pt', type=str)
    parser.add_argument('--imgsz', type=tuple, default=(512, 512), help='Image size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of channels')
    parser.add_argument('--gpus', type=str, default='0')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)

    infer(args)
