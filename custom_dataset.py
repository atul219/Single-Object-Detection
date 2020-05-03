from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import cv2
from PIL import Image
import PIL.Image



class PascalImage(Dataset):

	def __init__(self, csv_dir, img_dir, transforms):
		super(PascalImage, self).__init__()


		self.df = pd.read_csv(csv_dir)
		self.img_dir = img_dir
		self.tfms = transforms

		categories = self.df['category'].unique()

		self.cat2id = {}
		self.id2cat = {}

		for i, cat in enumerate(categories):
			self.cat2id[cat] = i
			self.id2cat[i] = cat


	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
		image = cv2.imread(img_name, cv2.COLOR_BGR2RGB)
		image = self.tfms(image)

		label = self.cat2id[self.df.iloc[idx, 1]]

		sample = (image, label)

		return sample

	def get_category_label(self, idx):
		return self.id2cat[idx]


class BoundingBox(Dataset):

	def __init__(self, csv_dir, img_dir, transforms):
		super(BoundingBox, self).__init__()

		self.df = pd.read_csv(csv_dir)
		self.img_dir = img_dir
		self.tfms = transforms


	def __len__(self):

		return len(self.df)


	def __getitem__(self, idx):
		image = PIL.Image.open(self.img_dir/self.df.loc[idx]['file_name'])
		bbox_str = self.df.loc[idx]['bbs']
		bbox = [int(i) for i in bbox_str.split(' ')]
		sample = self.tfms({'image': image, 'bbox': bbox})
		sample['bbox'] = torch.FloatTensor(sample['bbox'])

		return (sample['image'], sample['bbox'])


	def get_category_label(self, idx):
		return self.id2cat[idx]



class CombineData(Dataset):

	def __init__(self, img_data, bb_data):
		super(CombineData, self).__init__()

		self.img_data = img_data
		self.bb_data = bb_data

		self.get_category_label = self.img_data.get_category_label
		self.total_categories = len(self.img_data.id2cat.items())


	def __len__(self):
		return len(self.img_data)

	def __getitem__(self, idx):

		_, cat_id = self.img_data[idx]
		img, bbox = self.bb_data[idx]

		return (img, (bbox, cat_id))

		