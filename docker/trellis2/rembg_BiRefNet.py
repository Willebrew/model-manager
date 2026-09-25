from typing import *

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BiRefNet:
    """Background-removal wrapper used by TRELLIS.2 preprocess_image.

    ZhengPeng7/BiRefNet (and BRIA RMBG-2.0) checkpoints are typically fp16.
    transforms.ToTensor() is fp32. Feeding float32 into a half-precision
    conv on this PyTorch/CUDA stack raises:

        RuntimeError: Input type (float) and bias type (c10::Half) should be the same

    Cast the input to the model's device and dtype instead of hardcoding cuda/fp32.
    """

    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        param = next(self.model.parameters())
        input_images = (
            self.transform_image(image)
            .unsqueeze(0)
            .to(device=param.device, dtype=param.dtype)
        )
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().float().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
