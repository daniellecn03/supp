import os
from enum import Enum
from torchvision import transforms, models
from diffusers import StableDiffusionInpaintPipeline
from perturbation_metric.attributions.baselines.ViT_new import vit_base_patch16_224
from perturbation_metric.utils import utils
import perturbation_metric.utils.cnn as cnn
import torch

class ModelOptions(Enum):
    RESNET = "resnet"
    VIT = "vit"
    VGG = "vgg"
    ALEXNET = "alexnet"


class Model:
    def __init__(self, model_option, weights_path=""):
        if model_option not in ModelOptions:
            raise ValueError(f"Invalid value for param: {model_option}. Must be one of {list(ModelOptions)}")
        self.model_option = model_option
        if (model_option == ModelOptions.RESNET):
            # Model
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.model.cuda()
            self.model_normalization_method = utils.transform_normalize_resnet
            self.attribution_methods = ['rand', 'inputgradient', 'smoothgrad', 'fullgrad', 'gradshap', 'gc']

        elif (model_option == ModelOptions.VGG):
            # Model
            self.model = models.vgg19(pretrained=True)
            self.model.eval()
            self.model.cuda()
            self.model_normalization_method = utils.transform_normalize_vgg
            self.attribution_methods = ['rand', 'inputgradient', 'smoothgrad', 'fullgrad', 'gradshap', 'gc']

        elif (model_option == ModelOptions.VIT):
            self.model = vit_base_patch16_224(pretrained=True).cuda()
            self.model.eval()
            self.model_normalization_method = utils.transform_normalize_vit
            self.attribution_methods = ['rand', 'gc', 'transformer', 'attn_last_layer', 'lrp_last_layer', 'rollout']

        elif (model_option == ModelOptions.ALEXNET):
            self.model = cnn.CNN().cuda()
            self.model.load_state_dict(torch.load(weights_path))
            self.model.eval()
            self.model_normalization_method = utils.transform_normalize_vit
            self.attribution_methods = ['rand', 'integratedgrad', 'inputxgrad', 'gradshap', 'gc', 'layered_gc']

    def is_binary(self):
        return self.model_option == ModelOptions.ALEXNET
