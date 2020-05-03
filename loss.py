'''
Loss Functions 
'''

import torch
import torch.nn.functional as F
import numpy as np

# for combine loss of bounding box and classification


def loss(pred, label):
	bb_l, c_l = label
	bb_p, c_p = pred[:,:4], pred[:,4:]

	return F.l1_loss(bb_p, bb_l) + F.cross_entropy(c_p, c_l)*20


def get_reg_loss(pred, label):
	bb_l, c_l = label
	bb_p, c_p = pred[:,:4], pred[:,4:]

	return F.l1_loss(bb_p, bb_l).data


def get_cat_loss(pred, label):
	bb_l, c_l = label
	bb_p, c_p = pred[:,:4], pred[:,4:]

	return F.cross_entropy(c_p, c_l)


def get_accuracy(pred, label):
	_, c_l = label
	c_p = pred[:,4:]

	pred_labels = c_p.argmax(dim=1)
	acc = (pred_labels == c_l).float().mean()

	return acc