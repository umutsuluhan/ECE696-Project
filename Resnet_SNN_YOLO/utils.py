import cv2
import torch
import random
import numpy as np
import torchvision.transforms.functional as FF

from PIL import Image

# Gaussian noise augmentation
def gaussian_noise(img):  
    return FF.gaussian_blur(img, (3,3), 0.7)

# Contrast augmentation
def contrast(img):  
    factor = random.uniform(0.9, 1.1)
    img = FF.adjust_contrast(img, factor)
    return img

# Horizontal flip augmentation
def horizontal_flip(img, targets):
    img = FF.hflip(img)
    new_targets =[]
    for target in targets:
        new_targets.append((target[0], 1 - target[1], target[2], target[3], target[4]))
    return img, new_targets

# Resize and padding augmentation
def resize_and_pad(img, targets, new_height, new_width, pad_width):
    original_width, original_height = img.size
    # Maintaining the aspect ratio
    scale_height = new_height / original_height
    
    new_targets = []
    for target in targets:
        class_id, x_center, y_center, width, height = target
        
        x_center = x_center * original_width
        y_center = y_center * original_height
        width = width * original_width
        height = height * original_height
        
        x_center_new = x_center * scale_height
        y_center_new = y_center * scale_height
        width_new = width * scale_height
        height_new = height * scale_height
        
        x_center_new /= new_width
        y_center_new /= new_height
        width_new /= new_width
        height_new /= new_height

        new_targets.append((class_id, x_center_new, y_center_new, width_new, height_new))

    img = img.resize((new_width, new_height))
    
    targets = new_targets
    padding_left = pad_width - new_width
    img = np.array(img)
    img = np.pad(img, ((0, 0), (padding_left, 0), (0, 0)), mode='constant', constant_values=0) 
    img = Image.fromarray(img.astype(np.uint8))

    padding_left_normalized = (padding_left / new_width) / 2
    new_targets = []
    for target in targets:
        class_id, x_center, y_center, width, height = target
        x_center += padding_left_normalized
        new_targets.append((class_id, x_center, y_center, width, height))
    return img, new_targets

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
    return result * 255
