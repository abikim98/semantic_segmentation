#!/usr/bin/env python
# coding: utf-8

# 1.   model: unet

# 1-1 CelebAMask 이미지 라벨만들기
# 



import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = './CelebAdataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = './CelebAdataset/CelebAMask-HQ/CelebAMask_label'
img_num = 30000

for k in tqdm(range(img_num)):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = f"{folder_base}/{folder_num}/{k:0>5}_{label}.png"
        if (os.path.exists(filename)):
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_base[im != 0] = (idx + 1)

    filename_save = f"{folder_save}/{k}.png"
    cv2.imwrite(filename_save, im_base)


# 1-1-1 mask를 train valid 로 나누기


for path in glob('./CelebAdataset/*'):
    print(len(glob(path+'/*')))



all_mask = glob('./CelebAdataset/CelebAMask-HQ/CelebAMask_label/*')
print(len(all_mask))



import random
random.shuffle(all_mask)



cnt = int(len(all_mask)*0.9)
train = all_mask[:cnt]
valid = all_mask[cnt:]
print(len(train))
print(len(valid))




import shutil
for path in tqdm(train):
    shutil.copy(path, path.replace('./CelebAdataset/CelebAMask-HQ/CelebAMask_label', './CelebAdataset/CelebAMask-HQ/train_mask'))


import shutil
for path in tqdm(valid):
    shutil.copy(path, path.replace('./CelebAdataset/CelebAMask-HQ/CelebAMask_label', './CelebAdataset/CelebAMask-HQ/valid_mask'))


# 1-2 image dataset/loader 정의



import os
import copy
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms



import numpy as np



class CelebAMaskDataset():
    def __init__(self, img_path, label_path, transform_img, transform_label):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.dataset = glob(f'{label_path}/*')

    def __getitem__(self, index):
        label_path = self.dataset[index] 
        img_path = label_path.replace(self.label_path, self.img_path).replace('png', 'jpg')
        image = Image.open(img_path)
        label = Image.open(label_path)
        img, la = self.transform_img(image), self.transform_label(label)

        return img, la

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)


transform_Image = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
transform_Label = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])



train_dataset = CelebAMaskDataset('./CelebAdataset/CelebAMask-HQ/CelebA-HQ-img', './CelebAdataset/CelebAMask-HQ/train_mask', transform_Image, transform_Label)
valid_dataset = CelebAMaskDataset('./CelebAdataset/CelebAMask-HQ/CelebA-HQ-img', './CelebAdataset/CelebAMask-HQ/valid_mask', transform_Image, transform_Label)



train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
valid_dataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8)


dataset = iter(train_dataset)
img, label = next(dataset)
img, label = img.numpy().transpose((1, 2, 0)), label.numpy().transpose((1, 2, 0))

print('input image')
plt.imshow(np.int32(img*255))
plt.show()

print('mask image')
plt.imshow(np.int32(label*255).reshape(512,512))
plt.show()


# 1-3 model 정의


def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())
    return conv


class Unet(nn.Module):
    def __init__(self, num_class=1):
        super(Unet, self).__init__()

        ## encoder
        self.down1_1  = conv3x3(in_channels=3, out_channels=64)
        self.down1_2 = conv3x3(in_channels=64, out_channels=64)
        self.downpool1 = nn.MaxPool2d(kernel_size=2)

        self.down2_1 = conv3x3(in_channels=64, out_channels=128)
        self.down2_2 = conv3x3(in_channels=128, out_channels=128)
        self.downpool2 = nn.MaxPool2d(kernel_size=2)

        self.down3_1 = conv3x3(in_channels=128, out_channels=256)
        self.down3_2 = conv3x3(in_channels=256, out_channels=256)
        self.downpool3 = nn.MaxPool2d(kernel_size=2)

        self.down4_1 = conv3x3(in_channels=256, out_channels=512)
        self.down4_2 = conv3x3(in_channels=512, out_channels=512)
        self.downpool4 = nn.MaxPool2d(kernel_size=2)

        self.down5_1 = conv3x3(in_channels=512, out_channels=1024)

        ## decoder
        self.up5_1 = conv3x3(in_channels=1024, out_channels=512)

        self.uppool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.up4_2 = conv3x3(in_channels=2 * 512, out_channels=512)
        self.up4_1 = conv3x3(in_channels=512, out_channels=256)

        self.uppool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)        
        self.up3_2 = conv3x3(in_channels=2 * 256, out_channels=256)
        self.up3_1 = conv3x3(in_channels=256, out_channels=128)


        self.uppool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)        
        self.up2_2 = conv3x3(in_channels=2 * 128, out_channels=128)
        self.up2_1 = conv3x3(in_channels=128, out_channels=64)

        self.uppool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)        
        self.up1_2 = conv3x3(in_channels=2 * 64, out_channels=64)
        self.up1_1 = conv3x3(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        down1_1 = self.down1_1(x)
        down1_2 = self.down1_2(down1_1)
        downppool1 = self.downpool1(down1_2)

        down2_1 = self.down2_1(downppool1)
        down2_2 = self.down2_2(down2_1)
        downppool2 = self.downpool2(down2_2)

        down3_1 = self.down3_1(downppool2)
        down3_2 = self.down3_2(down3_1)
        downpool3 = self.downpool3(down3_2)

        down4_1 = self.down4_1(downpool3)
        down4_2 = self.down4_2(down4_1)
        downpool4 = self.downpool4(down4_2)

        down5_1 = self.down5_1(downpool4)

        ## decoder
        up5_1 = self.up5_1(down5_1)

        uppool4 = self.uppool4(up5_1)
        cat4 = torch.cat((uppool4, down4_2), dim=1)
        up4_2 = self.up4_2(cat4)
        up4_1 = self.up4_1(up4_2)

        uppool3 = self.uppool3(up4_1)
        cat3 = torch.cat((uppool3, down3_2), dim=1)
        up3_2 = self.up3_2(cat3)
        up3_1 = self.up3_1(up3_2)

        unpool2 = self.uppool2(up3_1)
        cat2 = torch.cat((unpool2, down2_2), dim=1)
        up2_2 = self.up2_2(cat2)
        up2_1 = self.up2_1(up2_2)

        unpool1 = self.uppool1(up2_1)
        cat1 = torch.cat((unpool1, down1_2), dim=1)
        up1_2 = self.up1_2(cat1)
        up1_1 = self.up1_1(up1_2)

        logit = self.fc(up1_1)
        return logit


import segmentation_models_pytorch as smp



import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smp.Unet(classes=19)
# model = Unet(19)
model.to(device)



import torch.nn.functional as F
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()  
    nt, ht, wt = target.size()  

    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)  

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) 
    target = target.view(-1)
    loss = F.cross_entropy(input, target)
    return loss


criterion = cross_entropy2d
optimizer = optim.Adam(model.parameters(), 0.0002, [.5, .999])
scheduler=None
num_epochs=50



def generate_label_plain(inputs, imsize=512):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())
                
    label_batch = np.array(label_batch)

    return label_batch



best_model_wts = copy.deepcopy(model.state_dict())
best_iou = 0.0

for epoch in range(num_epochs):
    running_loss = 0.0
    train_corrects = 0
    train_data_cnt = 0
    train_progress_bar = tqdm(train_dataloaders) 
    for inputs, labels in train_progress_bar:
        model.train()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        size = labels.size()
        labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
        labels_real_plain = labels[:, 0, :, :].cuda()

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels_real_plain.long())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        train_data_cnt += inputs.size(0)
        train_progress_bar.set_description(f" Epoch[{epoch+1}/{num_epochs}] train : runing_Loss {running_loss / train_data_cnt:.5f}")
        
    if scheduler:
        scheduler.step()

    valid_loss = 0
    valid_data_cnt = 0
    val_iou = 0
    valid_progress_bar = tqdm(valid_dataloaders)
    for inputs, labels in valid_progress_bar:    
        model.eval()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad ():
            outputs = model(inputs)

        labels_predict_plain = generate_label_plain(outputs)

        intersection = torch.logical_and(labels, torch.tensor(labels_predict_plain).cuda())
        union = torch.logical_or(labels, torch.tensor(labels_predict_plain).cuda())
        val_iou += torch.sum(intersection) / torch.sum(union) * inputs.size(0)

        size = labels.size()
        labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
        labels_real_plain = labels[:, 0, :, :].cuda()
        loss = criterion(outputs, labels_real_plain.long())
        
        valid_loss += loss.item() * inputs.size(0)
        valid_data_cnt += inputs.size(0)
        valid_progress_bar.set_description(f" Epoch[{epoch+1}/{num_epochs}] valid : valid_acc {valid_loss / valid_data_cnt} valid_iou {val_iou / valid_data_cnt}")
    
    plt.imshow(labels_predict_plain[0])
    plt.show()

    epoch_iou = val_iou / valid_data_cnt
    if epoch_iou > best_iou:
        best_iou = epoch_iou
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
    print('-'*10, f"best epoch : {best_epoch}", '-'*10)