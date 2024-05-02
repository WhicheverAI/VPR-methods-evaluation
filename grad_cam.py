# 工作流程
# 1. 寻找想要可视化的模型。模型可以加载自某一次Evaluation的logs中，通过读取config.json，我们就知道模型怎么加载，知道评估数据集在哪。
#  TODO 目前这个是耦合的，只能支持写了我的 ViTPermuteAsCNN的模型，这个后面再泛化。
#  其他对比模型的 grad cam也应该支持。 
# 2. 寻找想要可视化的图片。可以是Evaluation中发现的错误或者正确图片，或者对比实验中。
# TODO 目前是自对比策略。其实应该有一张 query 图片，对应不同模型检索的database图片。
# 不同模型的注意力不一样。 
# 3. 运行脚本，得到可视化结果（一张新的同样大小的图片）。
# 


#%%

import io
import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import vpr_models
import parser
import commons
import visualizations
from test_dataset import TestDataset
# %%
from vpr_models.utils import SimpleImageDataset
data_transform = test_ds.base_transform
# batch_dir = "./logs/default/2024-05-01_03-39-01/preds"
batch_dir = "../VPR-datasets-downloader/datasets/pitts30k/images/images/test/queries"
single_image = f"{batch_dir}/@0584334.39@4476910.88@17@T@040.43857@-080.00562@004561@00@@@@@@pitch1_yaw1@.jpg"

# grad_cam_single(img_path = ,
#                 model=model,
#                 target_layers=target_layers)
simple_ds = SimpleImageDataset([single_image], 
                               test_ds.base_transform)
simple_loader = DataLoader(simple_ds, batch_size=1, shuffle=False)
for batch in simple_loader: pass
batch.shape
#%%
# 看来这样加载不了
# from vpr_models.utils import DeployedModel

# import sys
# sys.path.append("../deep-visual-geo-localization-benchmark")

# resume_folder = "./logs/default/2024-04-29_20-33-43"
# deployed_model:DeployedModel = torch.load(f"{resume_folder}/deployed_model.pth")
# %%
import json
resume_folder = "./logs/default/2024-04-29_20-33-43"
with open(f"{resume_folder}/config.json") as f:
    resumed_args = json.load(f)
import argparse

# 将字典转换为 Namespace 对象
args = argparse.Namespace(**resumed_args)
args
#%%
# jupyer中简单尝试，先用CPU推理
args.device = 'cpu'
# %%
model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension, args,
                             device = args.device)
model = model.eval().to(args.device)


test_ds = TestDataset(args.database_folder, args.queries_folder,
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Testing on {test_ds}")
from bigmodelvis import Visualization
Visualization(model).structure_graph()
# Visualization(model.model.backbone[0]).structure_graph()
model(torch.randn((1, 3, 224, 224))).shape
# %%
# pip install grad-cam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
# target_layers = [model.model.backbone[0].vit_model.encoder.layer[-1]]
# target_layers = [model.model.backbone[0].vit_model]
# target_layers = [model.model.aggregation]
# target_layers = [model.model.backbone[0]] 
target_layers = [model.model.backbone] # 这个应该输出的是 PermuteAsCNN 之后的，所以很对。
# cam = GradCAM(
cam = GradCAMPlusPlus(
    model=model, 
              target_layers=target_layers, 
                # reshape_transform=Debug()
              )
cam

#%%
query_descripor = model(batch)
from xai.grad_cam_utils import VPRTarget
grayscale_cam = cam(input_tensor=batch,
                    targets=[VPRTarget(query_descripor, 
                                       positive=True
                                       )], 
                    )
grayscale_cam.shape # (1, 480, 640)
#%%
# from torchvision import transforms
from PIL import Image
# rgb_image = np.array(transforms.ToTensor()(Image.open(single_image)))
rgb_image = np.array(Image.open(single_image)) / 255
# from cv2 import imread
# rgb_image = imread(single_image)/255
rgb_image.shape
#%%
from matplotlib import pyplot as plt
# plt.show()
# grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_image, 
                                  grayscale_cam[0, :], 
                                  use_rgb=True)
plt.imshow(visualization)


# %%
