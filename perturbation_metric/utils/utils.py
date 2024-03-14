import os
import shutil
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pickle
from torchvision.utils import save_image
import torch.nn as nn
from scipy.stats import kendalltau
from typing import Tuple

def interpolate_down(res):
		return torch.nn.functional.interpolate(res, scale_factor=1/16, mode='bilinear').cuda()

def interpolate_up(res):
		return torch.nn.functional.interpolate(res, scale_factor=16, mode='bilinear').cuda()

def softmax(model_output):
	return nn.Softmax(dim=1)(model_output)

def transform_normalize_resnet(image_tensor):
	return transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image_tensor)

def transform_normalize_vgg(image_tensor):
	return transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image_tensor)

def transform_normalize_vit(image_tensor):
	return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)

def transform_normalize_alexnet(image_tensor):
	return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)

def preprocess_image(image_path):
	float_images = transforms.ToTensor()(Image.open(image_path))
	float_images = float_images.type(torch.cuda.FloatTensor)
	float_images.cuda()
	float_images = torch.unsqueeze(float_images, dim=0)
	return transforms.Compose[
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor()](float_images)

def get_conf(t):
	return round(t.item(),4)

def gen_attribution_mask(attribution, step, patchify = True):
	batch_size = 1

	relevance = attribution.clone()
	if (relevance.dim() == 3):
		relevance = torch.unsqueeze(relevance, dim=0)

	if patchify:
		relevance = interpolate_down(relevance)
		relevance = interpolate_up(relevance)

	relevance_flattened = relevance.clone().reshape(batch_size, -1)
	num_pixels = relevance_flattened.shape[-1]
	k = num_pixels * step

	top = torch.topk(relevance_flattened,k=int(k), dim=-1, largest=True).indices
	
	mask = torch.zeros_like(relevance_flattened)
	for b in range(batch_size):
		mask[b, top[b]] = 1
	mask = mask.reshape(relevance.shape)
	return mask.cuda()

def generate_images(pipeline, image_path, loaded_attribution, step, prompt, mask_output_path):
	image = Image.open(image_path).resize((512, 512))
	print('prompt: %s' % prompt)
	results = []

	mask = gen_attribution_mask(loaded_attribution, step)
	save_image(mask[0], mask_output_path)

	mask_image = Image.open(mask_output_path).resize((512, 512))
	num_samples = 3
	generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

	images = pipeline(
		prompt=prompt,
		image=image,
		mask_image=mask_image,
		guidance_scale=7.5,
		generator=generator,
		num_images_per_prompt=num_samples,
	).images

	return images

def ToHuman(idx):
	# Load names for ImageNet classes
	object_categories = {}
	with open("perturbation_metric/utils/imagenet1000_clsidx_to_labels.txt", "r") as f:
		for line in f:
			key, val = line.strip().split(":")
			object_categories[key] = val
	# reconstructing the data as a dictionary
	return object_categories[idx]

def ToHumanBinary(idx):
	if (idx == '0'):
		return "Dog"
	if (idx == '1'):
		return "Cat"
	return "ERROR"

def kandall_distance_matrix(orderings):
	n = len(orderings)
	mean_distances = np.zeros(n)

	# Convert the ordering elements to numerical representations
	numerical_orderings = [[ordering.index(item) for item in ordering] for ordering in orderings]
	
	# Create a reference of the mapping
	reference_ordering = orderings[0]
	mapping = {item: index for index, item in enumerate(reference_ordering)}

	# Convert to numerical representations using the mapping
	numerical_orderings = [[mapping[item] for item in ordering] for ordering in orderings]

	dist_mat = np.zeros((4,4))
	# Calculate pairwise Kendall distance
	for i in range(n):
		for j in range(n):
			tau, _ = kendalltau(numerical_orderings[i], numerical_orderings[j])
			dist_mat[i][j] = tau
	return dist_mat

def auc(y):
	z = y.copy()
	if len(z) == 10:
		z.insert(0, 0.0)
	auc = np.trapz(y=z, x=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	return round(auc,3)

def calculate_mean_per_channel(image_path: str) -> Tuple[float, float, float]:
	img = Image.open(image_path)
	img_tensor = transforms.ToTensor()(img)
	mean_per_channel = torch.mean(img_tensor, dim=(1, 2))
	return tuple(mean_per_channel.numpy())

def get_mean_image(image_path):
	mean_rgb_values = calculate_mean_per_channel(image_path)
	print(mean_rgb_values)
	mean_rgb_tensor = torch.tensor(mean_rgb_values, dtype=torch.float32)
	image_tensor = mean_rgb_tensor.view(3, 1, 1).repeat(1, 224, 224)
	return image_tensor

def blur_image(image_path, output_path):
	float_images = Image.open(image_path)
	blurred_image = float_images.filter(ImageFilter.GaussianBlur(radius = 10))
	blurred_image.save(output_path)

def init_weights_corr(directory, examples, attribution_methods, dict_to_fill):
	i = 0
	for ex in examples:
		print(i)
		i+=1
		for attr in attribution_methods:
			for step in  ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
				for idx in range(3):       
					key = "%s_%s_%s_%s" % (ex, attr, step, idx)
					top2_path = directory + ex + '/inpaint_attr/new_%s_%s_%s.pt' % (attr, step, idx)
					orig1_path = directory + ex + '/attr/%s_attr.pt' % (attr)
					orig2_path = directory + ex + '/orig2_attr/orig2_%s.pt' % (attr)
					
					top2_inpaint_attr = torch.load(top2_path).cuda()
					top2_tensor = fix_attribution(top2_inpaint_attr, float(step), should_interpolate=0)[0][0]

					orig1_attr = torch.load(orig1_path).cuda()
					orig1_tensor = fix_attribution(orig1_attr, float(step), should_interpolate=1)[0][0]

					orig2_attr = torch.load(orig2_path).cuda()
					orig2_tensor = fix_attribution(orig2_attr, float(step), should_interpolate=0)[0][0]

					plus = orig1_tensor + orig2_tensor
					mul = orig1_tensor * orig2_tensor
					sub = (plus - mul).cuda()
					cross = sub * top2_tensor
					dict_to_fill[key] = cross.sum().item()
