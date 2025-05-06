import os
import cv2
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils import gaussian_noise, contrast, horizontal_flip, resize_and_pad

class VisdroneDataset(Dataset):
    # Set parameters for dataset
    def __init__(self, dataset_name, device, num_steps=None, stride=None, batch_size=32):
        self.dataset_name = dataset_name
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.stride = stride
        self.device = device

        # Load data information
        self.load_data()
    
    def load_data(self):
        if self.dataset_name == "train":
            data_path = os.path.join(r'C:\Users\suluh\Visdrone_YOLO\images\train\\')
            label_path = os.path.join(r'C:\Users\suluh\Visdrone_YOLO\labels\train\\')
            
            # Train dataset image and label load
            data_raw_list = sorted(os.listdir(data_path))
            label_raw_list = sorted(os.listdir(label_path))
            self.data_list = []
            self.label_list = []

            for data_raw in data_raw_list:
                self.data_list.append(data_path + data_raw)
            for label_raw in label_raw_list:
                self.label_list.append(label_path + label_raw)

        elif self.dataset_name == "val":
            data_path = os.path.join(r'C:\Users\suluh\Visdrone_YOLO\images\val\\')
            label_path = os.path.join(r'C:\Users\suluh\Visdrone_YOLO\labels\val\\')

            # Validation dataset image and label load
            data_raw_list = sorted(os.listdir(data_path))
            label_raw_list = sorted(os.listdir(label_path))
            self.data_list = []
            self.label_list = []

            for data_raw in data_raw_list:
                self.data_list.append(data_path + data_raw)
            for label_raw in label_raw_list:
                self.label_list.append(label_path + label_raw)

        elif self.dataset_name == "test":
            data_path = os.path.join(r'C:\Users\suluh\Visdrone_YOLO\images\test\\')
            label_path = os.path.join(r'C:\Users\suluh\Visdrone_YOLO\labels\test\\')

            # Test dataset image and label load
            data_raw_list = sorted(os.listdir(data_path))
            label_raw_list = sorted(os.listdir(label_path))
            self.data_list = []
            self.label_list = []

            for data_raw in data_raw_list:
                self.data_list.append(data_path + data_raw)
            for label_raw in label_raw_list:
                self.label_list.append(label_path + label_raw)

        else:
            print("There is no dataset partition named " + self.dataset_name + ", pal!")
            exit()
        
        # Define dataset length 
        self.data_len = len(self.data_list)
        

    def __len__(self):
        # For SNN, dataset size is calculated according to image load stride (imitating time steps)
        return (self.data_len - self.num_steps) // self.stride + 1
        

    
    def aug(self, img, targets, mode=None, flip=None):
        img = Image.fromarray(img.astype(np.uint8))

        if mode == "train" or mode == "val":
            # Gaussian noise and contrast adjustment
            img = gaussian_noise(img)
            img = contrast(img)

            # Flip with %20 probability
            if flip:
                img, targets = horizontal_flip(img, targets)
        
        # Resizing and padding 
        img, targets = resize_and_pad(img, targets, 240, 426, 432)

        img = np.array(img)
        return img, targets

    def __getitem__(self, index):
        # 10 back-to-back frames are fed to model
        concatenated_img = None
        index *= self.stride
        flip=False

        p = random.uniform(0.0, 1.0)
        if(p < 0.2):
            flip=True

        for iter in range(self.num_steps):
            img_path = self.data_list[index+iter]
            label_path = self.label_list[index+iter]

            img = cv2.imread(img_path)

            targets = []
            label_file = open(label_path, "r")
            labels_list = label_file.readlines()
            for label in labels_list:
                split_labels = label.split((" "))
                targets.append((int(split_labels[0]), float(split_labels[1]), float(split_labels[2]), float(split_labels[3]), float(split_labels[4])))
            
            if self.dataset_name == 'train':
                img, targets = self.aug(img, targets, "train", flip=flip)
            else:
                img, targets = self.aug(img, targets, "val", flip=flip)
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.device).float() / 255.0

            if concatenated_img is None:
                concatenated_img = img 
            else:
                concatenated_img = torch.cat((concatenated_img, img), dim=0)
            
            if iter == 0:
                targets = torch.tensor(targets)
                target_length_final = len(targets)

                max_target_len = 256
                padded_targets = []
                for target in targets:
                    padded_targets.append(tuple(target.tolist()))

                for idx in range(max_target_len - len(targets)):
                    padded_targets.append((0,0,0,0,0))
                
                targets_final = torch.tensor(padded_targets)
        return concatenated_img, targets_final, target_length_final
        
