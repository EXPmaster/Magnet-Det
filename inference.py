import argparse
import time
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
def infer(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists('./data/outputs_10'):
        os.mkdir('./data/outputs_10')
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(args.device)
    model.eval()
    weights = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(weights['model'])
    datalist = os.listdir(args.data_path)
    datalist = list(filter(lambda x: x.endswith('Ori.png') and int(x.split('_')[2]) > 5, datalist))
    datalist = list(map(lambda x: '_'.join(x.split('_')[:-1]), datalist))
    testset = MagnetDataset(datalist, imgsz=args.imgsz)
    loader = DataLoader(testset, batch_size=1)
    print('Start inference...')

    for idx, (data, data_txt, label, label_txt) in enumerate(loader):
        start = time.time()
        prefix = '_'.join(str(int(data_txt[0][i].item())) for i in range(4))
        data, data_txt_gpu = data.to(args.device), data_txt.to(args.device)
        data = data.unsqueeze(1)
        data_txt_gpu = data_txt_gpu[None] #  / 400.
        predict_img, predict_txt = model(data, data_txt_gpu)
        predict_txt = predict_txt.squeeze().cpu().numpy()
        M, L1, Q1 = predict_txt[0:3]
        M = M * (6.3442e-05 - 2.9278e-08) + 2.9278e-08
        L1 = L1 * (4.6380e-02 - 6.5287e-07) + 6.5287e-07
        Q1 = Q1 * (3.6427e+02 - 4.9272e+00) + 4.9272e+00
        L2 = 142.216 * 1e-6
        Q2 = 19.236
        eff = cal_efficiency(M, L1, L2, Q1, Q2)
        end = time.time()
        total_time = end - start
        prefix += '_' + str(round(eff, 3)) + '_' + str(round(total_time, 3))
        predict = predict_img.squeeze().cpu().numpy()
        line_mask = draw_line(predict, predict_txt[-1], predict_txt[-2])
        predict = (predict / predict.max() * 255).astype(np.uint8)
        # label = label.numpy()
        # label = (label / label.max() * 255).astype(np.uint8)
        predict = cv2.applyColorMap(predict, cv2.COLORMAP_JET)
        predict[line_mask == 255] = [255, 255, 255]
        # label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
        if data_txt[0][0] == 60 and data_txt[0][1] == 10 and data_txt[0][2] == 1 and data_txt[0][4] == 2:
            cv2.imwrite(os.path.join(args.output_path, f'{prefix}_predict.png'), predict)
            # cv2.imwrite(os.path.join(args.output_path, f'{prefix}_label.png'), label)
        elif data_txt[0][0] == 60 and data_txt[0][1] == 10 and data_txt[0][2] == 1 and data_txt[0][4] == 10:
            cv2.imwrite(os.path.join('./data/outputs_10', f'{prefix}_predict.png'), predict)


def cal_efficiency(M, L1, L2, Q1, Q2):
    k2 = M ** 2 / (L1 * L2)
    return k2 * Q1 * Q2 / (1 + k2 * Q1 * Q2)


def draw_line(img_gray, min_val, max_val):
    img_gray = (img_gray * 255).astype(np.uint8)
    mask = np.ones_like(img_gray) * 255
    min_val = min_val * (4.75949139e-09 - 1.22639357e-12) + 1.22639357e-12
    max_val = max_val * (1.33987507e-02 - 1.49683161e-05) + 1.49683161e-05
    ratio = int((27 * 1e-6 - min_val) / (max_val - min_val) * 255)
    mask[img_gray == ratio] = 0
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 5))
    msk = cv2.medianBlur(mask, 3)
    msk = cv2.erode(msk, kernel1)
    msk = cv2.erode(msk, kernel2)
    # Skeletonization-like operation in OpenCV
    msk = cv2.ximgproc.thinning(~msk)
    msk[0, :] = 0
    msk[:, 0] = 0
    msk[len(msk) - 1, :] = 0
    msk[:, len(msk) - 1] = 0
    
    return msk


def test_draw():
    data_name = './data/dataset_new/LLsim00188_60_10_1_64_2_150_218_Ori.png'
    label_name = './data/dataset_new/LLsim00188_60_10_1_64_2_150_218_ResBW.png'
    img = cv2.imread(data_name, 0)[50:-50, 50:-50]
    label = cv2.imread(label_name, 0)[50:-50, 50:-50]
    label = img - label
    txt_name = './data/dataset_new/LLsim00188_60_10_1_64_2_150_218_TabSave.txt'
    with open(txt_name, 'r') as f:
        string = f.readline().strip().split(',')
    max_val = (float(string[12]) - 1.49683161e-05) / (1.33987507e-02 - 1.49683161e-05)
    min_val = (float(string[14]) - 1.22639357e-12) / (4.75949139e-09 - 1.22639357e-12)
    line_mask = draw_line(label / 255., min_val, max_val)

    predict = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    predict[line_mask == 255] = [255, 255, 255]
    cv2.imwrite('tmp.png', predict)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data/dataset_new', type=str)
    parser.add_argument('--output-path', default='./data/outputs_02', type=str)
    parser.add_argument('--weight-path', default='./weights/best.pt', type=str)
    parser.add_argument('--imgsz', type=tuple, default=(512, 512), help='Image size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of channels')
    parser.add_argument('--gpus', type=str, default='0')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)

    infer(args)
    # test_draw()
