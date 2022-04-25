
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import cv2
# import scipy.misc
# from tool import imutils
from torchvision import transforms

IMG_FOLDER_NAME = "/data/sonnh8/SEAM/TrainValid/Images"
ANNO_NAME = "/data/sonnh8/SEAM/TrainValid/label.txt"

CAT_LIST = ['normal', 'polyp']

def parse_txt(root, txt):
    with open(txt, 'r') as f:
        raws = f.read().split('\n')
    labels = [r.split('\t') for r in raws if r != '']
    labels = [{'name':os.path.join(root, r[0]), 'label':1-int(r[1])} for r in labels]
    return labels

class PolypImageDataset(Dataset):

    def __init__(self, transform=None):
        self.data = parse_txt(IMG_FOLDER_NAME, ANNO_NAME)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        img = PIL.Image.open(d['name']).convert("RGB")
#         print(img)

        if self.transform:
            img = self.transform(img)
        label = [1.0, 0.0] if d['label'] == 0 else [0.0, 1.0]
        label = torch.from_numpy(np.array(label))

        return d['name'], img, label

if __name__ == '__main__':
#     cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()
#     print(cls_labels_dict)
    PID = PolypImageDataset()
    _, img, _ = PID[100]
    img = np.array(img)
    print(img.shape)
