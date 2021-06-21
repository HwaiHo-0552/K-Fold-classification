#########################################################################
#                           读入数据                                    #
#                    SAIL LAB, 机器学习研发                             #
#                           2021.5.26                                   #
#########################################################################
#!/usr/bin/python
#-*- coding:UTF-8 -*-

from __future__ import print_function, division

import os
import torch
import numpy as np  
import pandas as pd
from PIL import Image
from PIL import ImageFile

root_path = '/home/maxiaozhi/CNN/cnn_1/predicted_dataset'

class data_set:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def reading_txt(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        dataloader_prdicted = {}
        for txt_file in os.listdir(self.dataset_path):
            if txt_file.endswith('.txt'):
                predicted = os.path.join(self.dataset_path, txt_file)
                dataloader_prdicted = self.loading(predicted)

        return dataloader_prdicted

    def loading(self, list_pth):
        list_img = []
        list_label = []
        dict_dataset = {}
        with open(list_pth, 'r+', encoding='UTF-8') as file:
            lines = file.readlines()
        for content in lines:
            img = content.strip().split()[0]
            img = Image.open(img).convert('RGB')
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)
            list_img.append(img)

            label = content.strip().split()[1:]
            label = [int(i) for i in label]                                      # 遍历label的list并将其元素改为int型
            label = np.array(label)
            label = torch.tensor(label)
            list_label.append(label)
        
        dataloaders = torch.utils.data.DataLoader(dataset=list_img, 
                                                  batch_size=1, 
                                                  num_workers=1)
        dict_dataset['img'] = dataloaders
        dataloaders = torch.utils.data.DataLoader(dataset=list_label, 
                                                  batch_size=1, 
                                                  num_workers=1)
        dict_dataset['label'] = dataloaders

        return dict_dataset

class read_data:
    def __init__(self, root_pth):
        self.root_pth = root_pth
        self.csv_pth = os.path.join(root_path, 'dataset.csv')
        self.image_pth = os.path.join(root_path, 'images_predicted')
    
    def reading(self):
        df = pd.read_csv(self.csv_pth)
        dict_dataset = {}                                         # 新建一个dict_dataset字典，key是image的名字，而value是image的存放地址和image的标签.
        
        new_df = df[['image_name', 'label']]                      # 新建一个df, 只取源csv中, image的名字和image的标签.
        for i in range(0, len(new_df)):
            img_name = new_df.loc[i][0]
            img_path = os.path.join(self.image_pth, img_name)
            img_label = new_df.loc[i][1]
            dict_dataset[img_name] = [img_path, img_label]        # 后面在索引image的地址和标签时, 只需对应索引其list[0]和list[1]

        return dict_dataset

    def make_txt(self, dict_temp):                                # 这个函数将分好的数据写入txt文件中, 保存下来
        save_data = os.path.join(self.root_pth, 'images.txt')
        load_text = open(save_data, 'w', encoding='UTF-8')

        for i in dict_temp:
            info = dict_temp[i][0] + ' ' + str(dict_temp[i][1]) + '\n'
            load_text.writelines(info)  

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    work_1 = read_data(root_path)
    dict_image = work_1.reading()
    dataset_txt = work_1.make_txt(dict_image)

    work_2 = data_set(root_path)
    dataloader_predicted = work_2.reading_txt()

if __name__ == '__main__':
    main()