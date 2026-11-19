from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def swin_stage2_reshape(tensor):
    """
    Convert Swin Stage 2 NHWC features into NCHW format.
    Expected output: [B, C, H, W].
    """
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()

    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        side = int(tokens ** 0.5)

        if side * side != tokens:
            raise ValueError(
                f"Cannot reshape {tokens} tokens into a square grid."
            )

        return tensor.transpose(1, 2).reshape(
            batch, channels, side, side
        )

    raise ValueError(
        f"Unexpected Swin feature shape: {tuple(tensor.shape)}"
    )


class NetraScanXAI:

    def __init__(self, inference_engine):
        self.engine = inference_engine

        self.model = inference_engine.models["Swin-Tiny"]
        self.device = inference_engine.device

        self.target_layer = (
            self.model.layers[2].blocks[-1].norm1
        )

        print("PASS — Swin-Tiny Stage 2 Grad-CAM target configured.")

    def generate(self, image, class_index=None):

        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(
                "Expected PIL image or image file path."
            )

        image_size = 224

        resized_image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR
        )

        rgb_float = (
            np.asarray(resized_image, dtype=np.float32) / 255.0
        )

        input_tensor = self.engine.transforms["Swin-Tiny"](
            image
        ).unsqueeze(0).to(self.device)

        if class_index is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                class_index = int(logits.argmax(dim=1).item())

        from pytorch_grad_cam.utils.model_targets import (
            ClassifierOutputTarget
        )

        targets = [ClassifierOutputTarget(class_index)]

        with torch.enable_grad():
            with GradCAM(
                model=self.model,
                target_layers=[self.target_layer],
                reshape_transform=swin_stage2_reshape
            ) as cam:

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets
                )[0]

        overlay = show_cam_on_image(
            rgb_float,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.55
        )

        import cv2

        heatmap_bgr = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam),
            cv2.COLORMAP_JET
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

        return {
            "overlay": Image.fromarray(overlay),
            "heatmap": Image.fromarray(heatmap_rgb),
            "class_index": class_index,
            "cam": grayscale_cam
        }
