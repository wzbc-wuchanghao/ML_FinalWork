import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
from dataset import MyDataset
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from model.resnet_cbam import resnet50_cbam
import torch.optim as optim
from torch.optim import lr_scheduler
from model import vit_example
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

f_res = open(str(time.time())+'.txt', 'w')

batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

print(f"Using {device} device",file=f_res)

data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(320),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# 数据读取
train_dataset = MyDataset(root='/home/daslab/nfs/wch/homework/ml/fabric_defect/',train=True, transform = data_transforms['train'])
test_dataset = MyDataset(root='/home/daslab/nfs/wch/homework/ml/fabric_defect/',train=False, transform = data_transforms['val'])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# model_ft = models.vgg16(pretrained=True)
# num_ftrs = model_ft.classifier[6].in_features
# model_ft.classifier[6] = nn.Linear(num_ftrs, 2)

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = models.densenet121(pretrained=True)
# num_ftrs = model_ft.classifier.in_features
# model_ft.classifier = nn.Linear(num_ftrs, 2, bias=True)


# model_ft = torch.hub.load('facebookresearch/deit:main','deit_base_patch16_384', pretrained=True)
# num_ftrs = model_ft.head.in_features
# model_ft.head = nn.Linear(num_ftrs, 2)
print(model_ft)
print(model_ft,file=f_res)


model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) 
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[30, 60], gamma=0.1)

def train(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",file=f_res)
    scheduler.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n",file=f_res)

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print(f"Epoch {t+1}\n-------------------------------",file=f_res)
    train(train_loader, model_ft, criterion, optimizer_ft,exp_lr_scheduler)
    test(test_loader, model_ft, criterion)
print("Done!\n")
print("Done!\n",file=f_res)
torch.save(model_ft, 'resnet50.pth')
f_res.close()


