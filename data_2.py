#########################################################################
#                           读入数据                                    #
#                    思翠人工智能, 机器学习研发                          #
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

kfolder = int(5)
BS = 4
NW = 4
root_path = '/home/maxiaozhi/CNN/cnn_1/k_folder'
csv_path = '/home/maxiaozhi/CNN/cnn_1/k_folder/dataset.csv'
image_path = '/home/maxiaozhi/CNN/cnn_1/k_folder/data'

class data_set:
    def __init__(self, dataset_path, batch_size, num_workers):
        self.dataset_path = dataset_path
#        self.data_augmentation = data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def reading_txt(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        dataloader_train = {}
        dataloader_val = {}
        for root, folders, files in os.walk(self.dataset_path):
            for f in folders:                                          
                f_path = os.path.join(root, f)
                txt = os.listdir(f_path)
                train_path = os.path.join(f_path, txt[0])
                val_path = os.path.join(f_path, txt[1])

                dataloader_train[f] = self.loading(train_path)
                dataloader_val[f] = self.loading(val_path)

        return dataloader_train, dataloader_val

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
                                                  batch_size=self.batch_size, 
                                                  num_workers=self.num_workers)
        dict_dataset['img'] = dataloaders
        dataloaders = torch.utils.data.DataLoader(dataset=list_label, 
                                                  batch_size=self.batch_size, 
                                                  num_workers=self.num_workers)
        dict_dataset['label'] = dataloaders

        return dict_dataset

class K_Folder:
    def __init__(self, csv_pth, image_pth, k, save_txt_path):
        self.csv_pth = csv_pth
        self.image_pth = image_pth
        self.k = k
        self.save_txt_path = save_txt_path
        self.k_folder = {}
    
    def read_data(self):
        df = pd.read_csv(self.csv_pth)
        # 建一个dict_image字典, 存放K个数据集.
        dict_image = {}

        counter = 0
        dict_dataset = {}                                       # 新建一个dict_dataset字典，key是image的名字，而value是image的存放地址和image的标签.
        new_df = df[['image_name', 'f_8']]        # 新建一个df, 只取源csv中, image的名字和image的标签.
        for i in range(0, len(new_df)):
            img_name = new_df.loc[i][0]
            img_path = os.path.join(self.image_pth, img_name)
            img_label_1 = new_df.loc[i][1]
            dict_dataset[img_name] = [img_path, img_label_1]   # 后面在索引image的地址和标签时, 只需对应索引其list[0]和list[1]

        list_image = [ i for i in new_df['image_name'] ]
        for i in range(0, self.k):
            value_name = 'dataset_' + str(i)
            dict_image[value_name] = []
        dataset_0 = int(len(list_image)/self.k)                 # 将数据化为K份, 计算第一份的数量
        dataset_1 = int(2*dataset_0)                            # 计算第二份的数量
        dataset_2 = int(3*dataset_0)                            # 计算第三份的数量
        dataset_3 = int(4*dataset_0)                            # 计算第四份的数量

        for img in list_image:
            if counter<=dataset_0:
                dict_image['dataset_0'].append(dict_dataset[img])
                counter += 1            
            elif counter>dataset_0 and counter<=dataset_1:
                dict_image['dataset_1'].append(dict_dataset[img])
                counter += 1
            elif counter>dataset_1 and counter<=dataset_2:
                dict_image['dataset_2'].append(dict_dataset[img])
                counter += 1
            elif counter>dataset_2 and counter<=dataset_3:
                dict_image['dataset_3'].append(dict_dataset[img])
                counter += 1
            else:
                dict_image['dataset_4'].append(dict_dataset[img])
                counter += 1
        return dict_image

    def make_data(self, dict_temp):
        for i in dict_temp:
            self.k_folder[i] = [[], []]                   # 每个key对应一个value; 每个value是一个list; 而每个list中list[0]存放train, list[1]存放val

        for D in dict_temp:
            for d in dict_temp:
                if D==d:
                    self.k_folder[D][1] = dict_temp[d]                    
                else:
                    self.k_folder[D][0] += dict_temp[d]
        
        return self.k_folder

    def make_txt(self, dict_temp):                        # 这个函数将分好的数据写入txt文件中, 保存下来
        os.mkdir(self.save_txt_path)
        for files in dict_temp:
            sub_folder = os.path.join(self.save_txt_path, files)
            os.mkdir(sub_folder)

        for root, folders, files in os.walk(self.save_txt_path):
            for f in folders:
                for key in dict_temp:
                    if f==key:
                        save_folder = os.path.join(root, key)

                        dataset = dict_temp[key]
                        train_data = dataset[0]
                        val_data = dataset[1]
                    
                        save_train = os.path.join(save_folder, 'train.txt')
                        save_val = os.path.join(save_folder, 'val.txt')
                    
                        train_txt = open(save_train, 'w', encoding='UTF-8')
                        val_txt = open(save_val, 'w', encoding='UTF-8')
                        for contents in train_data:
                            for index in range(0, len(contents)):
                                if index == 0:
                                    info = contents[index]
                                else:
                                    info += ' '+str(contents[index])
                            info += '\n'
                            train_txt.writelines(info)
                        for contents in val_data:
                            for index in range(0, len(contents)):
                                if index == 0:
                                    info = contents[index]
                                else:
                                    info += ' '+str(contents[index])
                            info += '\n'
                            val_txt.writelines(info)    

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    KFolder_path = os.path.join(root_path, 'folders')

    work_1 = K_Folder(csv_path, image_path, kfolder, KFolder_path)
    dict_image = work_1.read_data()
    K_dataset = work_1.make_data(dict_image)

    if not os.path.exists(KFolder_path):
        work_1.make_txt(K_dataset)

    work_2 = data_set(KFolder_path, BS, NW)
    train, val = work_2.reading_txt()

if __name__ == '__main__':
    main()