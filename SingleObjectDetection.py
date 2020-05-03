import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import numpy as np
import os
import PIL.Image
import pandas as pd
from pathlib import Path
import json
import collections
import cv2

import helper
from custom_dataset import *
from custom_transform import *
from loss import *
from train import *


PATH = Path('../pascal')


train_json = json.load((PATH/'pascal_train2007.json').open())


IMAGE_PATH = PATH/'VOCdevkit/Voc2007/JPEGImages'


IMAGE = 'images'
ANNOTATIONS = 'annotations'
CATEGORIES = 'categories'
FILE_NAME = 'file_name'
ID = 'id'
IMAGE_ID = 'image_id'
BBOX = 'bbox'
CATEGORY_ID = 'category_id'
CATEGORY_NAME = 'name'

CATEGORY_DICT = {o[ID]: o[CATEGORY_NAME] for o in train_json[CATEGORIES]}
TRAIN_IMAGE = {o[ID]: o[FILE_NAME] for o in train_json[IMAGE]}
TRAIN_ID = {o[ID] for o in train_json[IMAGE]}

TRAIN_ANNOTAIONS = collections.defaultdict(lambda: [])

for i in train_json[ANNOTATIONS]:
    if not i['ignore']:
        b = i[BBOX]
        b = helper.convert_hw2bb(b)
        TRAIN_ANNOTAIONS[i[IMAGE_ID]].append((b, i[CATEGORY_ID]))
 
"""
to draw image with bounding box and category name

"""

def draw_idx(i):
	im_a = TRAIN_ANNOTAIONS[i]
	img = helper.open_image(IMAGE_PATH/TRAIN_IMAGE[i])
	helper.draw_img(img, im_a, CATEGORY_DICT)

"""
to get the largest annotations

"""

def get_largest_ann(b):
	b = sorted(b, key = lambda x: np.product(
		x[0][-2:] - x[0][:2]), reverse = True)
	return b[0]

TRAIN_LARGEST_ANNOTATIONS = {a: get_largest_ann(b) for a,b in TRAIN_ANNOTAIONS.items()}


""" 
to create csv files for bounding box and image category 

"""

(PATH/'csv_folder').mkdir(exist_ok = True)
CSV_BB = PATH/'csv_folder/largest_bb.csv'

CSV_CAT = PATH/'csv_folder/cat.csv'


df_cat = pd.DataFrame({'file_name' : [TRAIN_IMAGE[o] for o in TRAIN_ID],
					'category': [CATEGORY_DICT[TRAIN_LARGEST_ANNOTATIONS[o][1]] for o in TRAIN_ID]},
					columns = ['file_name', 'category'])


df_cat.to_csv(CSV_CAT, index = False)

bb = np.array([TRAIN_LARGEST_ANNOTATIONS[o][0] for o in TRAIN_ID])

bb_str = [' '.join(str(p) for p in o) for o in bb]

df_bb = pd.DataFrame({'file_name': [TRAIN_IMAGE[o] for o in TRAIN_ID],
					'bbs': bb_str},
					columns = ['file_name', 'bbs'])

df_bb.to_csv(CSV_BB, index = False)


'''
Transformations
'''
image_train_tfms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor()])

bbox_train_tfms = transforms.Compose([ResizeIB((224, 224)),
                                     RandomFlipIB(),
                                     ConvertTensor()])


'''

LOADING DATA

'''

'''
Image Data
'''
image_data = PascalImage(csv_dir = 'csv_folder/cat.csv', img_dir= IMAGE_PATH, transforms= image_train_tfms)

'''
Bounding Box Data
'''

bb_data = BoundingBox(csv_dir= 'csv_folder/largest_bb.csv', img_dir= IMAGE_PATH, transforms= bbox_train_tfms)

'''
Combine Data
'''

data = CombineData(img_data= image_data, bb_data= bb_data)

data_dl = DataLoader(data, batch_size= 64, shuffle= True)


model = models.vgg16(pretrained = True)

'''
Add custom classifer to vgg pretrained model
'''

classifier = nn.Sequential(
	nn.Linear(in_features = 25088, out_features = 256),
	nn.ReLU(),
	nn.BatchNorm1d(256),
	nn.Dropout(0.5),
	nn.Linear(in_features = 256, out_features = 4 + 20))

model = model.classifier

# Optimizer

optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

# <-----------------TRAINING----------------->

train(epochs= 1,model= model ,optimizer= optimizer, train_dataloader= data_dl)