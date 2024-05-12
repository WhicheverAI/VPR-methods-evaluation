
#%%
# pip install grad-cam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
from vpr_models.utils import SimpleImageDataset
simple_ds = SimpleImageDataset(['./logs/default/2024-05-01_03-39-01/preds/002.jpg'], transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
from PIL import Image
rgb_img = Image.open(simple_ds.image_paths[0])
input_tensor = torch.unsqueeze(simple_ds[0], 0) # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, 
              target_layers=target_layers
              )
cam
# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   ...

#%%
import numpy as np
rgb_img = np.array(rgb_img)
rgb_img = (rgb_img - rgb_img.min())/ (rgb_img.max()-rgb_img.min())
rgb_img.min(), rgb_img.max()

#%%

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.


# If targets is None, the highest scoring category will be used for every image in the batch. 
# 这是重点

# 所以我们VPR可以有自己的grad_cam，这方面工作是缺乏的。

targets = [ClassifierOutputTarget(281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# You can also get the model outputs without having to re-inference
model_outputs = cam.outputs
# %%
from matplotlib import pyplot as plt
# plt.show()
plt.imshow(visualization)

# %%


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]
    
    
# 实际上是把 哪一类的那个分数找出来，然后方便求导