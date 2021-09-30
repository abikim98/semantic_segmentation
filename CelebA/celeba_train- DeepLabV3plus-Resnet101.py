#!/usr/bin/env python
# coding: utf-8


# 1.   model: DeepLabV3plus

# 1-1 CelebAMask 이미지 라벨만들기
# 

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


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
print(label*255)


# 1-3 model


import segmentation_models_pytorch as smp
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smp.DeepLabV3Plus(encoder_name='resnet101',classes=19)
model.to(device)



import torch.nn.functional as F
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()  # 4, 19, 512, 512
    nt, ht, wt = target.size()  # 4, 512, 512

    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)  #4, 19, ht, wt

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) ## -1, c => ?, 19
    target = target.view(-1)
    loss = F.cross_entropy(input, target)
    return loss



criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), 0.0002, [.5, .999])
scheduler=None
num_epochs=25



def generate_label_plain(inputs, imsize=512):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        #pred = pred.reshape((1, 512, 512))
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