import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model.resnet_cbam import resnet50_cbam
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
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

model_ft = torch.load('model.pth')


# print(model_ft)

model_ft.eval()

target_layers = [model_ft.features.norm5]

methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

method = 'gradcam'

if method == "ablationcam":
    cam = methods[method](model=model_ft,
                                target_layers=target_layers,
                                use_cuda=True,
                                reshape_transform=vit_example.reshape_transform,
                                ablation_layer=vit_example.AblationLayerVit())
else:
    cam = methods[method](model=model_ft,
                                target_layers=target_layers,
                                use_cuda=True,
                                # reshape_transform=vit_example.reshape_transform
                                )
val_path = '/home/daslab/nfs/wch/homework/ml/fabric_defect/cls_ann/val.txt'
f = open(val_path,'r')
for path in f.readlines():
    label = path.strip().split(' ')[1]
    if(label=='0'):
        continue
    raw_path = path.strip().split(' ')[0]
    image_path = 'fabric_defect/crop_trgt/' + raw_path
    print(image_path)
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (400, 400))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    targets = None

    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=True,
                        aug_smooth=True)

    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    if(not os.path.exists('cam/'+method+'/')):
        os.mkdir('cam/'+method+'/')
    cv2.imwrite('cam/'+method+'/'+raw_path, cam_image)
f.close()