import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from perturbation_metric.attributions.baselines.ViT_explanation_generator import Baselines, LRP
from perturbation_metric.attributions.baselines.ViT_new import vit_base_patch16_224
from perturbation_metric.attributions.baselines.ViT_LRP import vit_base_patch16_224 as vit_LRP
from perturbation_metric.attributions.baselines.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

from torchvision.utils import save_image


class ViTAttributions:
	def __init__(self):
		# Model
		self.model = vit_base_patch16_224(pretrained=True).cuda()
		self.model.eval()
		self.baselines = Baselines(self.model)

		# LRP
		self.model_LRP = vit_LRP(pretrained=True).cuda()
		self.model_LRP.eval()
		self.lrp = LRP(self.model_LRP)

		# orig LRP
		self.model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
		self.model_orig_LRP.eval()
		self.orig_lrp = LRP(self.model_orig_LRP)

	def normalize_attribution(self, res):
		return (res - res.min()) / (res.max() - res.min())

	def interpolate(self, res):
		return torch.nn.functional.interpolate(res, scale_factor=16, mode='bilinear').cuda()

	def gen_vit_attributions(self, image_file, output_path):
		transform_normalize_vit = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		float_images = transforms.ToTensor()(Image.open(image_file))
		float_images = float_images.type(torch.cuda.FloatTensor)
		float_images.cuda()
		float_images = torch.unsqueeze(float_images, dim=0)

		normalized_images = transform_normalize_vit(float_images).cuda()
		classification = np.argmax(self.model.forward(normalized_images).cpu().detach()).item()


		# Random Baseline
		rand_attr = torch.rand(1, 1, 224, 224)
		rand_attr = self.normalize_attribution(rand_attr).cuda()
		torch.save(rand_attr, output_path + '/rand_attr.pt')


		# Rollout
		rollout_attr = self.interpolate(self.baselines.generate_rollout(normalized_images, start_layer=0).reshape(1, 1, 14, 14))
		rollout_attr = self.normalize_attribution(rollout_attr).cuda()
		torch.save(rollout_attr, output_path + '/rollout_attr.pt')


		# Transformer
		transformer_attr = self.interpolate(self.lrp.generate_LRP(normalized_images, start_layer=1, method="transformer_attribution", index=classification).reshape(1, 1, 14, 14))
		transformer_attr = self.normalize_attribution(transformer_attr)
		torch.save(transformer_attr, output_path + '/transformer_attr.pt')


		# LRP last layer
		lrp_last_layer_attr = self.interpolate(self.orig_lrp.generate_LRP(normalized_images, method="last_layer", is_ablation=False, index=classification) \
						.reshape(1, 1, 14, 14))
		lrp_last_layer_attr = self.normalize_attribution(lrp_last_layer_attr)
		torch.save(lrp_last_layer_attr, output_path + '/lrp_last_layer_attr.pt')


		# Attention last layer 
		attn_last_layer_attr = self.interpolate(self.orig_lrp.generate_LRP(normalized_images, method="last_layer_attn", is_ablation=False) \
						.reshape(1, 1, 14, 14))
		attn_last_layer_attr = self.normalize_attribution(attn_last_layer_attr)
		torch.save(attn_last_layer_attr, output_path + '/attn_last_layer_attr.pt')


		# Grad Cam
		gc_attr = self.interpolate(self.baselines.generate_cam_attn(normalized_images, index=classification).reshape(1, 1, 14, 14))
		gc_attr = self.normalize_attribution(gc_attr)
		torch.save(gc_attr, output_path + '/gc_attr.pt')
