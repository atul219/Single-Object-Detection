'''
Custom transformation to image and bounding box

'''

import torch
import torchvision
from torchvision.transforms import transforms
import math
import numpy as np
import PIL.Image



class ResizeIB(object):
    def __init__(self, size_tup):
        self.size = size_tup
        
    def __call__(self, sample):
        #bbox format (top left corner, bottom right corner) [y1, x1, y2, x2]
        img = sample['image'] 
        bbox_arr = sample['bbox']
        
        w, h = img.size
        w_r, h_r = self.size
        bbox = bbox_arr.copy()
        
        bbox[0] = int(bbox_arr[0]*(h_r/h))
        bbox[2] = int(bbox_arr[2]*(h_r/h))
        
        bbox[1] = int(bbox_arr[1]*(w_r/w))
        bbox[3] = int(bbox_arr[3]*(w_r/w))
        
        return { 'image' : img.resize(self.size), 'bbox' : bbox }


class RandomFlipIB(object):    
    def __call__(self, sample):
        img = sample['image']
        bbox = sample['bbox']
        
        if np.random.random() > 0.5:
           # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            #bbox format (top left corner, bottom right corner) [y1, x1, y2, x2]
            copy = int(bbox[1])
            
            bbox[1] = w - bbox[3]
            bbox[3] = w - copy
        return { 'image' : img, 'bbox' : bbox }


class ConvertTensor(object):        
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
        
    def __call__(self, sample):        
        return { 'image' : self.ToTensor(sample['image']), 'bbox' : torch.tensor(sample['bbox'], dtype=torch.float)} 
 
