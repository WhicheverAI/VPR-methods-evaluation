# Code from AnyLoc https://arxiv.org/abs/2308.00688

import torch
import torchvision.transforms as transforms


first_time_run = True
class AnyLocWrapper(torch.nn.Module):
    def __init__(self, model=None, resize=None):
        super().__init__()
        self.model = model or torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device="cuda")
        self.resize = resize
    def forward(self, images):
        if self.resize:
            images = transforms.functional.resize(images, self.resize, antialias=True)
        else:
            b, c, h, w = images.shape
            if first_time_run: 
                first_time_run = False
                print(f"AnyLocWrapper: input shape: {images.shape}")
            # DINO wants height and width as multiple of 14, therefore resize them
            # to the nearest multiple of 14
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            images = transforms.functional.resize(images, [h, w], antialias=True)
        return self.model(images)

