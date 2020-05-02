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
        

def draw_idx(i):
	im_a = TRAIN_ANNOTAIONS[i]
	img = helper.open_image(IMAGE_PATH/TRAIN_IMAGE[i])
	helper.draw_img(img, im_a, CATEGORY_DICT)