import os
import shutil
import timm
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from torchvision import models, transforms
from perturbation_metric.attributions.baselines.grad import InputGradient
from perturbation_metric.attributions.baselines.smoothgrad import SmoothGrad
from perturbation_metric.attributions.baselines.fullgrad import FullGrad
from captum.attr import GradientShap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam.utils.image import show_cam_on_image

class VGGAttributions:
	def __init__(self):

		# Model
		self.model = models.vgg19(pretrained=True)
		self.model.eval()
		self.model.cuda()

		# Attribution methods
		self.inputgradient = InputGradient(self.model)
		self.smoothgrad = SmoothGrad(self.model)
		self.fullgrad = FullGrad(self.model)
		self.gc = GradCAM(model=self.model, target_layers=[self.model.features[-1]], use_cuda=True)
		self.gradshap = GradientShap(self.model)

	def norm(self, A):
		A -= A.min(1, keepdim=True)[0]
		A /= A.max(1, keepdim=True)[0]

	def normalize_attribution(self, res):
		return (res - res.min()) / (res.max() - res.min())

	def get_sum_attribution(self, attribution):
		attribution = attribution.abs()
		attribution = attribution - attribution.mean()
		return torch.sum(attribution, dim=1)

	def gen_vgg_attributions(self, image_file, output_path):
		transform_normalize_vgg = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

		float_images = transforms.ToTensor()(Image.open(image_file))
		float_images = float_images.type(torch.cuda.FloatTensor)
		float_images.cuda()
		float_images = torch.unsqueeze(float_images, dim=0)

		normalized_images = transform_normalize_vgg(float_images).cuda()
		classification_tensor = np.argmax(self.model.forward(normalized_images).cpu().detach(), axis=1).cuda()

		# Random Baseline
		rand_attr = torch.rand(1, 1, 224, 224)
		rand_attr = self.normalize_attribution(rand_attr).cuda()
		torch.save(rand_attr, output_path + '/rand_attr.pt')

		# Input Gradients
		inputgradient_attr = self.get_sum_attribution(
			self.inputgradient.saliency(normalized_images, target_class=classification_tensor).type(torch.cuda.FloatTensor))    
		self.norm(inputgradient_attr)
		torch.save(inputgradient_attr, output_path + '/inputgradient_attr.pt')


		# Smooth Grad
		smoothgrad_attr = self.get_sum_attribution(
			self.smoothgrad.saliency(normalized_images, target_class=classification_tensor).type(torch.cuda.FloatTensor))     
		self.norm(smoothgrad_attr)
		torch.save(smoothgrad_attr, output_path + '/smoothgrad_attr.pt')


		# Full Grad
		fullgrad_attr = self.get_sum_attribution(
			self.fullgrad.saliency(normalized_images, target_class=classification_tensor).type(torch.cuda.FloatTensor))       
		self.norm(fullgrad_attr)
		torch.save(fullgrad_attr, output_path + '/fullgrad_attr.pt')


		# Grad Shap
		rand_img_dist = torch.cat([normalized_images * 0, normalized_images * 1]).cuda()
		gradshap_attr = self.get_sum_attribution(
			self.gradshap.attribute(normalized_images,
						  n_samples=50,
						  stdevs=0.0001,
						  baselines=rand_img_dist,
						  target=classification_tensor).type(torch.cuda.FloatTensor))
		self.norm(gradshap_attr)
		torch.save(gradshap_attr, output_path + '/gradshap_attr.pt')


		# Grad Cam
		targets = [ClassifierOutputTarget(classification_tensor[0])]
		gc_attr = torch.tensor(self.gc(input_tensor=normalized_images, targets=targets)).cuda()
		self.norm(gc_attr)
		torch.save(gc_attr, output_path + '/gc_attr.pt')

	

