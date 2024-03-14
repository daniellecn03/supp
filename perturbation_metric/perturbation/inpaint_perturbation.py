import os
import shutil
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
from torchvision.utils import save_image
from diffusers import StableDiffusionInpaintPipeline
from perturbation_metric.attributions.baselines.ViT_new import vit_base_patch16_224
from pytorch_grad_cam.utils.image import show_cam_on_image
from perturbation_metric.utils import utils

class InpaintPerturbation:
	def __init__(self, model_wrapper):
		self.model_wrapper = model_wrapper
		self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
			"runwayml/stable-diffusion-inpainting",
			torch_dtype=torch.float16,
		)
		self.pipeline = self.pipeline.to("cuda")
		self.to_human = utils.ToHuman
		if self.model_wrapper.is_binary():
			self.to_human = utils.ToHumanBinary

	def get_prediction(self, image_path):  
		image = transforms.ToTensor()(Image.open(image_path))
		image = self.model_wrapper.model_normalization_method(image)
		image = torch.unsqueeze(image, dim=0).cuda()
					  
		model_output = self.model_wrapper.model.forward(image).cpu().detach()
		top1 = torch.topk(model_output[0],2).indices[0].item()
		top2 = torch.topk(model_output[0],2).indices[1].item()
		
		conf = utils.softmax(model_output)
		return top1, self.to_human(str(top1)), utils.get_conf(conf[0][top1]), top2, self.to_human(str(top2)), utils.get_conf(conf[0][top2])

	def get_stats(self, image_path, attribution_path, output_path, masks_path, patchify = True, prompt_is_top = 2):
		steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		output = []

		float_images = transforms.ToTensor()(Image.open(image_path))
		float_images = float_images.type(torch.cuda.FloatTensor)
		float_images.cuda()
		float_images = torch.unsqueeze(float_images, dim=0)
		normalized_images = self.model_wrapper.model_normalization_method(float_images).cuda()

		# get predictions
		model_output = self.model_wrapper.model.forward(normalized_images)
		classification_tensor = np.argmax(model_output.cpu().detach(), axis=1).cuda()
		prompt_class = torch.topk(model_output[0],prompt_is_top).indices[prompt_is_top-1].item()
		prompt_text = self.to_human(str(prompt_class))

		os.makedirs(output_path, exist_ok=True)
		os.makedirs(masks_path, exist_ok=True)

		for attr in self.model_wrapper.attribution_methods:
			# Load attribution
			attribution = torch.load(os.path.join(attribution_path, attr + "_attr.pt"))

			if (attribution.dim() == 3):
				attribution = torch.unsqueeze(attribution, dim=0)

			# Use patches instead of pixels if needed
			if patchify:
				attribution = utils.interpolate_down(attribution)
				attribution = utils.interpolate_up(attribution)

			for step in steps:
				mask_output_path = os.path.join(masks_path, "mask_%s_%s.png" % (attr, step))
				inpainted_images = utils.generate_images(self.pipeline, image_path, attribution, step, prompt_text, mask_output_path)
				for im_idx, im in enumerate(inpainted_images):
					res_path = os.path.join(output_path, "inpainted_%s_%s_%s.png" % (attr, step, im_idx))
					im.resize((224,224)).save(res_path)
					pred = self.get_prediction(res_path)
					output.append([
						image_path,
						attr,
						step,
						im_idx,
						pred[0],
						pred[2],
						pred[3],
						pred[5],
						prompt_class
					])
		with open(output_path + 'inpaint_stats.pkl' , 'wb') as f:
			pickle.dump(output, f)
		return output

