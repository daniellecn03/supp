## Dataset

We assume a directory, that contains sub-directory for each image. 
The top directory will be used to aggreagte the results over the entire dataset.

## Preprocess images

Apply preprocessing to get images in size (224,224) - at this point the images should not be normalized for a specific model.

## Attribution generation

In order to generate the attributions for an image initiate the relevant class:
vit_generator = vit_attributions.ViTAttributions() or 
resnet_generator = resnet_attributions.ResnetAttributions()

Use vit_generator.gen_vit_attributions(image_path, path_to_output_directory) or resnet_generator.gen_resnet_attributions(image_path, path_to_output_directory).

This will generate a set of attribution to the selected images.
vit attributions : ['rand', 'gc', 'transformer', 'attn_last_layer', 'lrp_last_layer', 'rollout']
resnet attributions : ['rand', 'inputgradient', 'smoothgrad', 'fullgrad', 'gradshap', 'gc']
vgg attributions : ['rand', 'inputgradient', 'smoothgrad', 'fullgrad', 'gradshap', 'gc']
alexnet attributions : ['rand', 'integratedgrad', 'inputxgrad', 'gradshap', 'gc', 'layered_gc]

## Apply classic perturbation - e.g. delete most relevant pixels.

Create a model wrapper:
model = model.Model(model.ModelOptions.VIT) use model.ModelOptions.{VIT,RESNET,ALEXNET,VGG} as needed.

Create a perturbator:
classic = classic_perturbation.ClassicPerturbation(model)

In order to get blur stats, please create a blurred version of your input image, by calling
utils.blur_image(input_image_path, output_image_path)
for example:
utils.blur_image("dataset/a.JPEG", "dataset/blur_a.JPEG")


Apply it on the selected image:
classic.get_stats(image_path, blur_image_path, attributions_directory_path, output_stats_path)

A pickle file will be created at the specified output_stats_path
Conatining the deletion/blur/mean results (0-is prediction has changes, 100- is prediction is the same).

For example: 
classic.get_stats("perturbation_metric/data/1/1.JPEG",
	   "perturbation_metric/data/1/1_blur.JPEG",
           "perturbation_metric/data/1/attr/",
           "perturbation_metric/data/classic_stats.pickle",

## Apply our perturbation - inpaint most relevant pixels.

Create a model wrapper:
model = model.Model(model.ModelOptions.VIT) or model = model.Model(model.ModelOptions.RESNET)

Create a perturbator:
ours = inpaint_perturbation.InpaintPerturbation(model)

Apply it on the selected image:
ours.get_stats(image_path, attributions_directory_path, inpainted_images_path, masks_path)

The masks per attribution and steps will be saved in the masks_path.
The inpainted images will be saved in the inpainted_images_path.

Make sure to rerun the attribution methods on top of the inpainted images to generate the weights, aggregation logic is available in utils.

