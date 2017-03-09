import numpy as np


def define_loss(start_img, final_img):
	loss = ((start_img - final_img) ** 2).mean(axis = (1, 2, 3))
	return loss
