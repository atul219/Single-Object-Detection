'''
Training
'''

import torch
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import models

from loss import *

def train(epochs, model, optimizer, train_dataloader, val_dataloader = None):

	total_step = 0

	for i in range(epochs):

		model.train()

		for image, label in train_dataloader:

			optimizer.zero_grad()

			pred = model(image)
			l = loss(pred, label)
			l.backward()
			optimizer.step()

			cat_loss = get_cat_loss(pred, label)
			reg_loss = get_reg_loss(pred, label)
			acc = get_accuracy(pred, label)

			total_step += 1

			log_info = {'Detection overall loss': l.item(),
						'L1 Loss': reg_loss,
						'Cat Loss': cat_loss,
						'Accuracy': acc}

			print(log_info)


