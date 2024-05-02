#%%
# 我们需要改进一下算法， 写一个针对这个任务的 VPRTarget
# KNN 欧几里得距离= L2距离 = d(a, b) = l2_norm(a-b)
# l2_norm(v) = modulus(v) = sqrt(sum(v**2)) # 勾股定理
# 向量距离 = a dot b
# 余弦距离 = cos<a, b> = a dot b / |a| |b| 

# 余弦定理，如果知道 L2距离，|a| |b|, 可以求 cos<a, b>
# cos 0 = 1， cos 90 = 0, cos 180 = -1

# cos<a, b> = (|a|**2+|b|**2-l2_norm(a-b)**2)/2|a| |b|
# cos<a,b> 对 l2_norm(a-b) (>0) 单递递减， 如果 |a| |b| 不变的话

# VPR 实际上很落后，比起REID那边有metric learning，我们对比学习的目标和KNN的目标真的一致吗?
# 对比学习 的目标  l2norm(q, pos) < l2norm(q, neg)
# 这个目标倒也是对的。


# 我们原本的排序函数就是根据 l2 dis
# 我们没有二分类，没有softmax，没有sigmoid，纯粹是距离小概率大

# https://github.com/facebookresearch/faiss/wiki
# https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class VPRTarget:
    def __init__(self, query_descriptor:torch.Tensor, positive:bool=True):
        self.query_descriptor:torch.Tensor = query_descriptor
        self.distance_fun = nn.PairwiseDistance(2)
        self.sign = 1 if positive else -1 # 如果是True，展示模型最关注的东西，否则展示模型最不关注的东西

    def __call__(self, database_descriptor:torch.Tensor):
        l2_dis = self.distance_fun(database_descriptor, self.query_descriptor)
        return l2_dis * -1 * self.sign # 让两个图片距离近的特征很重要
        # grad_cam 是求导，所以能求导就行了，不需要满足 0-1之间

class VitPermuteAsCNN(nn.Module):
    """Some Information about VitPermuteAsCNN"""
    def __init__(self):
        super(VitPermuteAsCNN, self).__init__()

    def forward(self, x):
        # logging.info(f"x.shape: {x.shape}") # batch, patch数量, embed dim
        batch, patches, embed_dim = x.shape
        patch_side = int(patches ** 0.5)
        assert patch_side * patch_side == patches, f"Patch数量{patches}不是平方数"
        x = x.view(batch, patch_side, patch_side, embed_dim)
        x = x.permute(0, 3, 1, 2)
        return x

class VitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation
        if 'config' in dir(vit_model):
            self.config = vit_model.config # hf风格
        if 'dummy_inputs' in dir(vit_model):
            self.dummy_inputs = vit_model.dummy_inputs
        # .to('cuda')
        # b, c, h, w = 1, 3, 224, 224
        # self.dummy_inputs = torch.randn(b, c, h, w)
    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            res = self.vit_model(x).last_hidden_state[:, 1:, :]
        else:
            res = self.vit_model(x).last_hidden_state[:, 0, :]
        return res
    
class FacebookVitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation
    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            res = self.vit_model(x)['x_norm_patchtokens']
        else:
            res = self.vit_model(x)['x_norm_clstoken']
        return res

#%%
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image