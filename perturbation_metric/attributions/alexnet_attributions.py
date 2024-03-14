import os
import torch
import numpy as np
import perturbation_metric.utils.cnn as cnn
from PIL import Image
from torchvision import transforms

from captum.attr import GradientShap, InputXGradient, IntegratedGradients, LayerGradCam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pathlib import Path

from torchvision.utils import save_image


class AlexnetAttributions:
	def __init__(self):

		self.model = cnn.CNN()
		self.model.load_state_dict(torch.load("perturbation_metric/utils/cnn_weights.pt"))
		self.model.eval()

		self.ig = IntegratedGradients(self.model)
		self.ixg = InputXGradient(self.model)
		self.gs = GradientShap(self.model)
		self.gc = GradCAM(model=self.model, target_layers=[self.model.conv3], use_cuda=True)
		self.lgc = LayerGradCam(self.model, self.model.conv3)

	def normalize_attribution(self, res):
		return (res - res.min()) / (res.max() - res.min())

	def get_sum_attribution(self, attribution):
		attribution = attribution.abs()
		attribution = attribution - attribution.mean()
		return torch.sum(attribution, dim=1)

	def norm(self, A):
		A -= A.min(1, keepdim=True)[0]
		A /= A.max(1, keepdim=True)[0]

	def interpolate(self, res):
		return torch.nn.functional.interpolate(res, scale_factor=16, mode='bilinear').cuda()

	def gen_alexnet_attributions(self, image_file, output_path):
		transform_normalize_alexnet = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		float_images = transforms.ToTensor()(Image.open(image_file))
		float_images = float_images.type(torch.cuda.FloatTensor)
		float_images.cuda()
		float_images = torch.unsqueeze(float_images, dim=0)

		normalized_images = transform_normalize_alexnet(float_images).cuda()
		classification = np.argmax(self.model.forward(normalized_images).cpu().detach()).item()

		# IG
		attributions, delta = self.ig.attribute(normalized_images, target=classification, return_convergence_delta=True)
		attributions = self.get_sum_attribution(attributions)
		self.norm(attributions)
		torch.save(attributions, output_path + '/integratedgrad_attr.pt')
	
		# IXG
		attributions = self.ixg.attribute(normalized_images, target=classification)
		attributions = self.get_sum_attribution(attributions)
		self.norm(attributions)
		torch.save(attributions, output_path + '/inputxgrad_attr.pt')
	
		# GS
		baseline = torch.zeros_like(float_images)
		attributions = self.gs.attribute(normalized_images, baselines=baseline, target=classification)
		attributions = self.get_sum_attribution(attributions)
		self.norm(attributions)
		torch.save(attributions, output_path + '/gradshap_attr.pt')
		
		# GC  
		targets = [ClassifierOutputTarget(classification)]
		attributions = torch.tensor(self.gc(input_tensor=normalized_images, targets=targets)).cuda()
		self.norm(attributions)
		torch.save(attributions, output_path + '/gc_attr.pt')
		
		# LGC
		attributions = self.lgc.attribute(normalized_images, target=classification)
		attributions = LayerGradCam.interpolate(attributions, normalized_images.shape[2:])
		attributions = self.get_sum_attribution(attributions)
		self.norm(attributions)
		torch.save(attributions, output_path + '/layered_gc_attr.pt')
		
		# rand
		attributions = torch.rand(1, 224, 224)
		self.norm(attributions)
		torch.save(attributions, output_path + '/rand_attr.pt')    