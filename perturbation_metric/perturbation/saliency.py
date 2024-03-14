import os
import shutil
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import pickle

from perturbation_metric.attributions.baselines.ViT_new import vit_base_patch16_224
from pytorch_grad_cam.utils.image import show_cam_on_image
from perturbation_metric.utils import utils,model

class Saliency:
	def __init__(self, model_wrapper):
		self.model_wrapper = model_wrapper

	def softmax(self, model_output):
		return nn.Softmax(dim=1)(model_output)

	def find_rectangle(self, mask):
		rows, cols = np.where(mask == 1)
		if not len(rows) or not len(cols):
			raise ValueError("No 1s found in the mask.")
		
		top_left = (min(rows), min(cols))
		bottom_right = (max(rows), max(cols))
		
		return top_left, bottom_right

	def crop_image_to_rectangle(self, image, top_left, bottom_right):
		crop_box = (top_left[1], top_left[0], bottom_right[1]+1, bottom_right[0]+1)
		cropped_image = image.crop(crop_box)
		return cropped_image

	def calculate_rectangle_size(self, top_left, bottom_right):
		width = bottom_right[1] - top_left[1] + 1
		height = bottom_right[0] - top_left[0] + 1
		return width, height

	def fix_attribution(self, attribution, step):
		batch_size = 1

		relevance = attribution.clone()
		if (relevance.dim() == 3):
			relevance = torch.unsqueeze(relevance, dim=0)
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		k = num_pixels * step

		top = torch.topk(relevance_flattened,k=int(k), dim=-1, largest=True).indices
		
		mask = torch.zeros_like(relevance_flattened)
		for b in range(batch_size):
			mask[b, top[b]] = 1
		mask = mask.reshape(relevance.shape)
		return mask.cuda()

	def get_sal_stats(self, image_file, attributions_directory_path, output_path):
		manipulation_methods_names = ['sal']
		steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		# Init stats containers
		acc1_conf_rec = {}

		# process input image
		float_images = transforms.ToTensor()(Image.open(image_file))
		float_images = float_images.type(torch.cuda.FloatTensor)
		float_images.cuda()
		float_images = torch.unsqueeze(float_images, dim=0)
		normalized_images = self.model_wrapper.model_normalization_method(float_images).cuda()
		
		# get predictions
		initial_output = self.model_wrapper.model.forward(normalized_images)
		classification_tensor = np.argmax(initial_output.cpu().detach(), axis=1).cuda()
		
		# Compute accuracy
		for method_id, method in enumerate(self.model_wrapper.attribution_methods):
			acc1_conf_rec[method] = []
			attribution = torch.load(attributions_directory_path + method + "_attr.pt")
			if (attribution.dim() == 3):
				attribution = torch.unsqueeze(attribution, dim=0)
			for s in range(len(steps)):
				mask_image = transforms.ToPILImage()(self.fix_attribution(attribution, steps[s])[0])
				top_left, bottom_right = self.find_rectangle(transforms.ToTensor()(mask_image)[0])
				cropped_image = self.crop_image_to_rectangle(transforms.ToPILImage()(float_images[0]), top_left, bottom_right)
				cropped_image = cropped_image.resize((224,224))
				normalized_cropped = self.model_wrapper.model_normalization_method(transforms.ToTensor()(cropped_image)).cuda()
				normalized_cropped = torch.unsqueeze(normalized_cropped, dim=0)
				output = self.model_wrapper.model(normalized_cropped)
				conf = self.softmax(output)
				rec = self.calculate_rectangle_size(top_left, bottom_right)
				conf_val = conf[0][classification_tensor[0].item()].item()
				acc1_conf_rec[method].append((rec, conf_val))
	
		# Store data (serialize)
		with open(output_path, 'wb') as handle:
			pickle.dump(acc1_conf_rec, handle, protocol=pickle.HIGHEST_PROTOCOL)

