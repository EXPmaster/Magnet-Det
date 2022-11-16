import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import cv2


class MagnetDataset(Dataset):

    def __init__(self, data_list, imgsz=(256, 256)):
        self.data_list = data_list
        self.suffix_data = '_Ori.png'
        self.suffix_label = '_ResBW.png'
        self.imgsz = imgsz

    def __getitem__(self, idx):
        data_name = os.path.join('./data/dataset', self.data_list[idx] + self.suffix_data)
        label_name = os.path.join('./data/dataset', self.data_list[idx] + self.suffix_label)
        img = cv2.imread(data_name, 0)[50:-50, 50:-50]
        label = cv2.imread(label_name, 0)[50:-50, 50:-50]
        # label = 255 - label
        # label = (label.astype(np.float) - img.astype(np.float))
        label = img - label
        # cv2.imwrite('tmp.png', label)
        # assert False
        img = cv2.resize(img, self.imgsz) / 255.0
        label = cv2.resize(label, self.imgsz) / 255.0
        # img = img.transpose(2, 0, 1) / 255.0
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRA)
        # label = label.transpose(2, 0, 1) / 255.0
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data_list)


def train_test_split(dataset_path, ratio=0.7):
    datalist = os.listdir(dataset_path)
    datalist = list(filter(lambda x: x.endswith('Ori.png') and int(x.split('_')[2]) > 5, datalist))
    print(len(datalist))
    data_names = list(map(lambda x: '_'.join(x.split('_')[:-1]), datalist))
    train_len = int(len(data_names) * ratio)
    trainset, testset = random_split(dataset=data_names, lengths=[train_len, len(data_names) - train_len])
    return list(trainset), list(testset)


if __name__ == '__main__':
    trainset, testset = train_test_split('./data/dataset')
    # dataset = MagnetDataset(trainset)
    # data, label = next(iter(dataset))
    # print(data.shape, label.shape)
    assert False
    cv2.imwrite('tmp_raw.png', label)
    # data: 512 * 512 * 3
    # color_map: 256 * 3
    gray_values = np.arange(256, dtype=np.uint8)
    # color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    # color_to_gray_map = dict(zip(color_values, gray_values))
    color_map = cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).squeeze()
    label = gray_values[np.argmin((np.abs(label[:, :, None, :] - color_map[None, None, :, :])).sum(-1), -1)]
    # label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    # label = cv2.LUT(label, color_map)
    cv2.imwrite('tmp.png', label)
    img = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    cv2.imwrite('tmp_rec.png', img)
    # print(color_map)
