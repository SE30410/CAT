import os
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
from torchvision import transforms                                                               
class MyDataset(Dataset):
    def __init__(self, type, img_size_unet, data_dir):
        self.name2label = {"IAD": 0, "NC": 1}
     
        self.img_size_unet = img_size_unet#120
        self.data_dir = data_dir
        
        self.data_list1 = []
        self.data_list2 = []
        self.data_list3 = []
        self.file_class = [cla for cla in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cla))]#[IAD,NC]
        self.file_class.sort()

        for sub in self.file_class:#[IAD NC]
            self.file_class_all=os.path.join(data_dir,sub)#./train/IAD
            
            self.file_class_all_01=os.path.join(self.file_class_all,'CSF')#./IAD/GM
            for file_01 in os.listdir(self.file_class_all_01):
                    self.data_list1.append(os.path.join(self.file_class_all_01, file_01))#./IAD/GM/*
            
            self.file_class_all_02=os.path.join(self.file_class_all,'GM')#./IAD/WM
            for file_02 in os.listdir(self.file_class_all_02):
                    self.data_list2.append(os.path.join(self.file_class_all_02, file_02))
           
            self.file_class_all_03=os.path.join(self.file_class_all,'WM')#./IAD/CSF
            for file_03 in os.listdir(self.file_class_all_03):
                    self.data_list3.append(os.path.join(self.file_class_all_03, file_03))
           
            print("Load {} Data Successfully!".format(type))
      

    def __len__(self):
        return min(len(self.data_list1),len(self.data_list2),len(self.data_list3))


    def __getitem__(self, item):
 
        file1= self.data_list1[item]
        file2= self.data_list2[item]
        file3= self.data_list3[item]
     
        
        
 
        img1 = Image.open(file1)
       
        img1 = img1.resize((self.img_size_unet[0],self.img_size_unet[1]))
       
        img2 = Image.open(file2)
       
        img2 = img2.resize((self.img_size_unet[0],self.img_size_unet[1]))
        img3 = Image.open(file3)
       
        img3 = img3.resize((self.img_size_unet[0],self.img_size_unet[1]))

       
        file_label_ALL=file1.split('/')[-3]
        label = self.name2label[file_label_ALL]
        image_to_tensor = ToTensor()

        img1 = image_to_tensor(img1)
        img2 = image_to_tensor(img2)
        img3 = image_to_tensor(img3)
    
        label = tensor(label)

        return img1,img2,img3,label



