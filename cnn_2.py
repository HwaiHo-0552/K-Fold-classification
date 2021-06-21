#########################################################################
#                             训练网络                                  #
#                    思翠人工智能, 机器学习研发                          #
#                           2021.5.26                                   #
#########################################################################
#!/usr/bin/python
#-*- coding:UTF-8 -*-

from __future__ import print_function, division

import os
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler      
from neural_network import ResNet
from data_2 import data_set, K_Folder
from sklearn.metrics import accuracy_score

#########################################################################
#                  数据集地址; 训练网络的超参数设置                      #
#########################################################################
data_augmentation = True
learning_rate = 0.01
Momentum = 0.9
batch_size = 4
num_workers = 4
classes = 1                                                                             
epoch = 25
kfolder = int(5)
root_path = '/home/maxiaozhi/CNN/cnn_1/k_folder'
csv_path = '/home/maxiaozhi/CNN/cnn_1/k_folder/dataset.csv'
image_path = '/home/maxiaozhi/CNN/cnn_1/k_folder/data'
#########################################################################
#                           Train CNN                                   #
#########################################################################
def train_model(device, model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, num_epochs, KF):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        total_acc4feature_1 = 0.0
        for k in range(0, KF):
            dataset_num = 'dataset_' + str(k)
            print(dataset_num)
            
            s_1 = []
            l_1 = []

            for phase in ['train', 'val']:                                               ######模型训练、验证的计算设置######
                running_loss = 0.0
#                running_corrects = 0.0
                data_size = 0
                
                if phase == 'train':                                                     # 判断模型为训练or验证状态
                    load_data = dataloader_train[dataset_num]                            # 取得训练集数据
                    model.train()                                                        # 将模型设置为训练模式model, 可进行training使用
                else:
                    load_data = dataloader_val[dataset_num]                              # 取得验证集数据
                    model.eval()                                                         # 将模型设置为训练模式model, 可进行training使用

                for imgs, labels in zip(load_data['img'], load_data['label']):           # 取得数据集中的图像, 标签
                    imgs = imgs.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.float32)
                    
                    data_size += int(labels.size(0))                                     # 计算取得这批送入学习的数据集size总和

                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(imgs)
                        outputs = torch.sigmoid(outputs)
#                        preds, p_index = torch.max(outputs, 1)
                        gt, g_index = torch.max(labels, 1)
                        
                        for scores in outputs:
                            for s in scores:
                                s = float(s)
                                if s>=float(0.5):
                                    preds = int(1)
                                else:
                                    preds = int(0)
                                s_1.append(preds)

                        for gts in labels:
                            for g in gts:
                                gs = int(g)
                                l_1.append(gs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * imgs.size(0)
#                    running_corrects += torch.sum(p_index == labels.data)
                if phase == 'train':
                    scheduler.step()
            
                epoch_loss = running_loss / data_size
#                epoch_acc = running_corrects.double() / data_size
                epoch_acc = accuracy_score(s_1, l_1)
                    
                print('{} Loss: {:.4f} Acc4feature_1: {:.4f}'.format(phase, 
                                                                    epoch_loss, 
                                                                    epoch_acc))

                if phase == 'val':
                    total_acc4feature_1 = total_acc4feature_1 + epoch_acc

                # deep copy the model
                if phase == 'val' and total_acc4feature_1 > best_acc:
                    best_acc = total_acc4feature_1
                    best_model_wts = copy.deepcopy(model.state_dict())
   
            print()
        Mean_Accuracy4feature_1 = total_acc4feature_1/float(KF)
        print('Mean_Accuracy for evaluation: feature_1={:.4f}'.format(Mean_Accuracy4feature_1))

        print('save model epoch {}'.format(epoch))
        pth = '/home/maxiaozhi/CNN/cnn_1/checkpoint'
        fil = 'epoch_' + str(epoch) +'.pth'
        save_pth = os.path.join(pth, fil)
        torch.save(model.state_dict(), save_pth)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
            
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#########################################################################
#                           主函数                                      #
#########################################################################
def main():

    #设置是否为GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #数据集的读取
    KFolder_path = os.path.join(root_path, 'folders')

    work_1 = K_Folder(csv_path, image_path, kfolder, KFolder_path)
    dict_image = work_1.read_data()
    K_dataset = work_1.make_data(dict_image)

    if not os.path.exists(KFolder_path):
        work_1.make_txt(K_dataset)

    work_2 = data_set(KFolder_path, batch_size, num_workers)
    dataloader_train, dataloader_val = work_2.reading_txt()
    
    #神经网络的构建
    DCNN = ResNet(classes)
    model_ft = DCNN.module()

    #设定一些训练网络的计算方式
    model_ft = model_ft.to(device)
#    criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()        
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=Momentum)    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #训练网络，并显示运行计算的过程、统计结果
    train_model(device, model_ft, dataloader_train, dataloader_val, criterion, optimizer_ft, exp_lr_scheduler, epoch, kfolder)

if __name__ == '__main__':
    main()
