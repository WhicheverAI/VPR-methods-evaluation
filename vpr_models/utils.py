
class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

import torch
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from typing import List
from PIL import Image

from torchvision import transforms
default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class SimpleImageDataset(data.Dataset):
    def __init__(self, image_paths:List[str], base_transform=default_transform):
        super().__init__()
        self.image_paths = image_paths
        self.base_transform = base_transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.base_transform(pil_img)
        return normalized_img

    def __len__(self):
        return len(self.image_paths)

from tqdm import tqdm
class DeployedModel(nn.Module):
    """Some Information about DeployedModel"""
    def __init__(self, model, 
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                 base_transform=None):
        super(DeployedModel, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.batch_size = 1
        self.num_workers = 1
        self.base_transform = base_transform

    def forward(self, image_paths:List[str]):
        dataset = SimpleImageDataset(image_paths, self.base_transform)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, 
                                     shuffle=False)
        with torch.inference_mode():
            all_descriptors = []
            for images, indices in tqdm(dataloader, ncols=100):
                descriptors = self.model(images.to(self.device))
                descriptors = descriptors
                descriptors.append(descriptors)
            all_descriptors = torch.vstack(all_descriptors)
            return all_descriptors


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Debug(nn.Module):
    """Some Information about Debug"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # print(f"activation with shape {x.shape} is {x}")
        # x = x[0]
        # print(f"activation with shape {x.shape}. ")
        print(f"activation with x={x}")
        return x
    
#%%
