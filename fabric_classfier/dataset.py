import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from random import random
from PIL import Image
import os
import json

class MyDataset(Dataset):

    def __init__(self, root, train=True, transform = None, target_transform=None):
        super(MyDataset, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        #如果是训练则加载训练集，如果是测试则加载测试集
        if self.train :
            file_annotation = self.root+'cls_ann/train.txt'
            
        else:
            file_annotation = self.root+'cls_ann/val.txt'
            
        img_folder_temp = self.root+ '/crop_temp/'
        img_folder_trgt = self.root+ '/crop_trgt/'

        self.filenames = []
        self.labels = []

        with open(file_annotation, 'r') as f:
            for l in f.readlines():
                l = l.strip()
                file_name = l.split(' ')[0]
                label = l.split(' ')[1]
                # 有缺陷，添加数据集
                if(label=='1'):
                    self.filenames.append(img_folder_temp + file_name[:-4] + '_template' + file_name[-4:])
                    self.labels.append(0)
                    self.filenames.append(img_folder_trgt + file_name)
                    self.labels.append(1)
                elif(label=='0'):
                    if self.train :
                        # 训练加入一部分负样本
                        if(random()>=0):
                            self.filenames.append(img_folder_temp + file_name[:-4] + '_template' + file_name[-4:])
                            self.labels.append(0)
                    else:
                        self.filenames.append(img_folder_temp + file_name[:-4] + '_template' + file_name[-4:])
                        self.labels.append(0)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name).convert('RGB')
        # img = img.resize((384,384))
        # img = (plt.imread(img_name) / 255).astype(np.float32)
        img = self.transform(img)   

        return img, label

    def __len__(self):
        return len(self.filenames)
        



class ObjectDataset(Dataset):
    def __init__(self, root, train=True, transforms= None):
        self.root = root
        self.transforms = transforms
        if train:
            # root = 'fabric_defect'
            self.label_file = os.path.join(root, "annotations/train.json")
        else:
            self.label_file = os.path.join(root, "annotations/val.json")
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.labels = []
        with open(self.label_file, 'r') as f:
            img_json = json.loads(''.join(f.readlines()))
            img_list = img_json['annotations']
            img_full_list = img_json['images']
        for img in img_list:
            index = [ i['id'] for i in img_full_list ].index(img['image_id'] )
            if(index != -1):
                self.imgs.append(os.path.join('fabric_defect/crop_trgt',img_full_list[index]['file_name']) )
                self.labels.append(img)
        
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        boxes = []
        bbox = label['bbox']
        xmin = np.min(bbox[0])
        xmax = np.max(bbox[0]+bbox[2])
        ymin = np.min(bbox[1])
        ymax = np.max(bbox[1]+bbox[3])
        boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((label['category_id'],), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = label['area']
        # suppose all instances are not crowd
        iscrowd = torch.zeros((label['iscrowd'],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':
    a = [{'a':1,'b':2},{'a':3,'b':4}]
    print( a[[x['a'] for x in a].index(3)])

        
