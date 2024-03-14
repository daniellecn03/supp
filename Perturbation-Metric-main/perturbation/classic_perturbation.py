import os
import shutil
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn

from perturbation_metric.attributions.baselines.ViT_new import vit_base_patch16_224
from pytorch_grad_cam.utils.image import show_cam_on_image
from perturbation_metric.utils import utils,model

class ClassicPerturbation:
	def __init__(self, model_wrapper):
		self.model_wrapper = model_wrapper

	def delete_most_relevant(self, input_image, attribution, step):
		batch_size = input_image.shape[0]

		relevance = attribution.clone()
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		k = num_pixels * step
		
		# Replace the values of the pixels with the highest attribution, least relevant are kept
		replace_pixels = torch.topk(relevance_flattened,k=int(k), dim=-1, largest=True).indices
		
		mask = torch.ones_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, replace_pixels[b]] = 0
		mask = mask.reshape(relevance.shape)
		masked_images = input_image * mask
		return masked_images.cuda()


	def delete_least_relevant(self, input_image, attribution, step):
		batch_size = input_image.shape[0]

		relevance = attribution.clone()
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		k = num_pixels * (1-step)
		
		# keep the values of the pixels with the highest attribution, least relevant are deleted
		keep_pixels = torch.topk(relevance_flattened,k=int(k), dim=-1, largest=True).indices
		
		mask = torch.zeros_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 1
		mask = mask.reshape(relevance.shape)
		masked_images = input_image * mask
		return masked_images.cuda()

	def blur_most_relevant(self, input_images, attribution, step, blur_image):
		batch_size = input_images.shape[0]

		relevance = attribution.clone()
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		keep = num_pixels * (1-step)
		replace = num_pixels - keep
		
		if replace == 0:
			return input_images
		
		keep_pixels = torch.topk(relevance_flattened,k=int(keep), dim=-1, largest=False).indices

		mask = torch.zeros_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 1
		mask = mask.reshape(relevance.shape)
		replace_images = input_images * mask
			
		mask = torch.ones_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 0
		mask = mask.reshape(relevance.shape)
		keep_images = blur_image * mask
		
		return (keep_images + replace_images).cuda()


	def blur_least_relevant(self, input_images, attribution, step, blur_image):
		batch_size = input_images.shape[0]

		relevance = attribution.clone()
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		keep = num_pixels * (1-step)
		replace = num_pixels - keep
		
		if replace == 0:
			return input_images
		
		keep_pixels = torch.topk(relevance_flattened,k=int(keep), dim=-1, largest=True).indices

		mask = torch.zeros_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 1
		mask = mask.reshape(relevance.shape)
		keep_images = input_images * mask
			
		mask = torch.ones_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 0
		mask = mask.reshape(relevance.shape)
		replace_images = blur_image * mask
		
		return (keep_images + replace_images).cuda()


	def mean_most_relevant(self, input_images, attribution, step, mean_image):
		batch_size = input_images.shape[0]

		relevance = attribution.clone()
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		keep = num_pixels * (1-step)
		replace = num_pixels - keep
		
		if replace == 0:
			return input_images
		
		keep_pixels = torch.topk(relevance_flattened,k=int(keep), dim=-1, largest=False).indices

		mask = torch.zeros_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 1
		mask = mask.reshape(relevance.shape)
		replace_images = input_images * mask
			
		mask = torch.ones_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 0
		mask = mask.reshape(relevance.shape)
		keep_images = mean_image * mask
		
		return (keep_images + replace_images).cuda()


	def mean_least_relevant(self, input_images, attribution, step, mean_image):
		batch_size = input_images.shape[0]

		relevance = attribution.clone()
		relevance_flattened = relevance.clone().reshape(batch_size, -1)
		num_pixels = relevance_flattened.shape[-1]
		keep = num_pixels * (1-step)
		replace = num_pixels - keep
		
		if replace == 0:
			return input_images
		
		keep_pixels = torch.topk(relevance_flattened,k=int(keep), dim=-1, largest=True).indices

		mask = torch.zeros_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 1
		mask = mask.reshape(relevance.shape)
		keep_images = input_images * mask
			
		mask = torch.ones_like(relevance_flattened).cuda()
		for b in range(batch_size):
			mask[b, keep_pixels[b]] = 0
		mask = mask.reshape(relevance.shape)
		replace_images = mean_image * mask
		
		return (keep_images + replace_images).cuda()

	def accuracy(self, output, target, topk=(1,)):
		"""Computes the accuracy over the k top predictions for the specified values of k"""
		with torch.no_grad():
			maxk = max(topk)
			batch_size = target.size(0)

			_, pred = output.topk(maxk, 1, True, True)
			pred = pred.t()
			correct = pred.eq(target.view(1, -1).expand_as(pred))

			res = []
			for k in topk:
				correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
				res.append(correct_k.mul_(100.0 / batch_size))
			return res

	def get_prediction_full(self, normalized_image, i, j, k10):
		model_output = self.model_wrapper.model.forward(normalized_image).cpu().detach()
		top1 = torch.topk(model_output[0],2).indices[0].item()
		top2 = torch.topk(model_output[0],2).indices[1].item()
		top10 = torch.topk(model_output[0],10).indices[9].item()
		
		conf = utils.softmax(model_output)
		return top1, utils.get_conf(conf[0][top1]), top2, utils.get_conf(conf[0][top2]), top10, utils.get_conf(
			conf[0][top10]), i, utils.get_conf(conf[0][i]), j, utils.get_conf(
			conf[0][j]), k10, utils.get_conf(conf[0][k10])


	def get_stats(self, image_path, blur_path, attributions_directory_path, output_path, full_pred_path="", patchify = False):
		manipulation_methods = [self.delete_least_relevant,
								self.delete_most_relevant, 
								self.blur_least_relevant, 
								self.blur_most_relevant,
								self.mean_least_relevant,
								self.mean_most_relevant,
							   ]
		manipulation_methods_names = ['dlr', 'dmr', 'blr', 'bmr', 'mlr', 'mmr']

		steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
   
		# Init stats containers {d,b,m}lr = Delete/Blur/Mean Least Relevant, {d,b,m}mr = Delete/Blur/Mean Most Relevant
		acc1_stats = {'dlr':{}, 'dmr':{}, 'blr':{}, 'bmr':{}, 'mlr':{}, 'mmr':{}}

		# process input image
		float_images = transforms.ToTensor()(Image.open(image_path))
		float_images = float_images.type(torch.cuda.FloatTensor)
		float_images.cuda()
		float_images = torch.unsqueeze(float_images, dim=0)
		normalized_images = self.model_wrapper.model_normalization_method(float_images).cuda()
		
		# process splice image
		blur_images = transforms.ToTensor()(Image.open(blur_path))
		blur_images = blur_images.type(torch.cuda.FloatTensor)
		blur_images.cuda()
		blur_images = torch.unsqueeze(blur_images, dim=0)
		
		# process mean image
		mean_images = utils.get_mean_image(image_path)
		mean_images = torch.unsqueeze(mean_images, dim=0).cuda()
		
		# get predictions
		initial_output = self.model_wrapper.model.forward(normalized_images)
		classification_tensor = np.argmax(initial_output.cpu().detach(), axis=1).cuda()

		# Compute accuracy
		for method_id, method in enumerate(self.model_wrapper.attribution_methods):
			attribution = torch.load(attributions_directory_path +  method + "_attr.pt")
			if (attribution.dim() == 3):
				attribution = torch.unsqueeze(attribution, dim=0)
			
			for m_idx, m in enumerate(manipulation_methods):
				acc1_stats[manipulation_methods_names[m_idx]][self.model_wrapper.attribution_methods[method_id]] = [0] * len(steps)
				for s in range(len(steps)):
					if (m.__name__ == 'blur_most_relevant' or m.__name__ == 'blur_least_relevant'):
						manipulated_images = m(float_images, attribution, steps[s], blur_images)
					elif (m.__name__ == 'mean_most_relevant' or m.__name__ == 'mean_least_relevant'):
						manipulated_images = m(float_images, attribution, steps[s], mean_images)
					else:
						manipulated_images = m(float_images, attribution, steps[s])
					normalized_manipulated_images = self.model_wrapper.model_normalization_method(manipulated_images).cuda()
					normalized_manipulated_images = normalized_manipulated_images.type(torch.cuda.FloatTensor)
					output = self.model_wrapper.model(normalized_manipulated_images)
					
					# Acc compared to original prediction
					acc1, acc5 = self.accuracy(output, classification_tensor, topk=(1, 2))
					acc1_stats[manipulation_methods_names[m_idx]][self.model_wrapper.attribution_methods[method_id]][s] = acc1[0].item()
		
		# Store data (serialize)
		with open(output_path, 'wb') as handle:
			pickle.dump(acc1_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)