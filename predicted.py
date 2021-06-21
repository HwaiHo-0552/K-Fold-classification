#########################################################################
#                             训练网络                                  #
#                       SAIL LAB, 机器学习研发                          #
#                           2021.5.26                                   #
#########################################################################
#!/usr/bin/python
#-*- coding:UTF-8 -*-

import os
import torch
from neural_network import ResNet
from images import data_set, read_data

data_augmentation = False
learning_rate = 0.01
Momentum = 0.9
num_workers = 4
classes = 1
dataset_path = '/home/maxiaozhi/CNN/cnn_1/predicted_dataset'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preds = []
    GT = []

    # 将制作好的数据集读入
    dataset_txt = os.path.join(dataset_path, 'images.txt')
    if not os.path.exists(dataset_txt):
        work_1 = read_data(dataset_path)
        dict_image = work_1.reading()
        dataset_txt = work_1.make_txt(dict_image)
    
    work_2 = data_set(dataset_path)
    dataloader_predicted = work_2.reading_txt()

    # 定义网络
    DCNN = ResNet(classes)
    model_ft = DCNN.module()
    model_ft = model_ft.to(device)
    
    # 单GPU运算, 将之前保存的权值加载至网络, 进行预测
    weighted = '/home/maxiaozhi/CNN/cnn_1/checkpoint/epoch_20.pth'
    model_ft.load_state_dict(torch.load(weighted), strict=True)

    print('网络进行预测计算')
    model_ft.eval()
    
    with torch.no_grad():
        for imgs, labels in zip(dataloader_predicted['img'], dataloader_predicted['label']):
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model_ft(imgs)
            outputs = torch.sigmoid(outputs)
            if outputs >= float(0.5):
                score = int(1)
            else:
                score = int(0)
            gt = int(labels)
            
            GT.append(gt)  
            preds.append(score)
    
    print('预测集数据共:{}'.format(len(preds)))

if __name__ == '__main__':

    main()
