""" 
In this directory there 
are some helper fucntions 
"""

import  numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects



def convert_hw2bb(bb):
    return np.array([bb[1], bb[0], bb[1] + bb[3] - 1, bb[2] + bb[0] - 1])



def convert_bb2hw(bb):

	return np.array([bb[1], bb[0], bb[3] - bb[1] + 1, bb[2] - bb[0] + 1])


"""
Open Image

"""

def open_image(path):
	img = cv2.imread(str(path)).astype(np.float32)/255
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


"""
display image

"""

def show_image(img, figsize = None, axis = None):
	if not axis:
		fig, axis = plt.subplots(figsize = figsize)
	axis.imshow(img)
	axis.get_xaxis().set_visible(False)
	axis.get_yaxis().set_visible(False)
	return axis


def draw_outline(i, lw):
	i.set_path_effects([patheffects.Stroke(linewidth = lw, foreground = 'black'), patheffects.Normal()])


def draw_rect(axis , bb):

	patch = axis.add_patch(patches.Rectangle(bb[:2], *bb[-2:], fill = False, edgecolor = 'white', lw = 2))
	draw_outline(patch, 4)

def draw_text(axis, xy, text, size = 14):
	text = axis.text(*xy, text, verticalalignment = 'top', color = 'white', fontsize = size, weight = 'bold')

def draw_img(img, ann, cat_dict):
	ax = show_image(img, figsize = (16,8))
	for b,c in ann:
		b = convert_bb2hw(b)
		draw_rect(ax, b)
		draw_text(ax, b[:2], cat_dict[c], size = 16)

