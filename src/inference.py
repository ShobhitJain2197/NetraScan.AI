from pathlib import Path
from collections import Counter
from time import perf_counter

import torch
import timm
from PIL import Image
from torchvision import transforms


APP_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["normal", "immature", "mature"]

MODEL_CONFIGS = {
    "ViT-B/16": {
        "architecture": "vit_base_patch16_224",
        "checkpoint": "vit_v4/vit_v4_best.pth",
        "size": 224,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "Swin-Tiny": {
        "architecture": "swin_tiny_patch4_window7_224",
        "checkpoint": "swin_v4/swin_v4_best.pth",
        "size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "MobileViT-S": {
        "architecture": "mobilevit_s",
        "checkpoint": "mobilevit_v4/mobilevit_v4_best.pth",
        "size": 256,
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    },
}


class NetraScanInference:

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.models = {}
        self.transforms = {}

        print(f"NetraScan.AI device: {self.device}")

        if self.device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

        self.load_models()

    def load_models(self):

        for name, config in MODEL_CONFIGS.items():

            print(f"Loading {name}...")

            model = timm.create_model(
                config["architecture"],
                pretrained=False,
                num_classes=len(CLASS_NAMES),
            )

            checkpoint_path = (
                APP_ROOT / "models" / config["checkpoint"]
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

            model.load_state_dict(checkpoint, strict=True)

            model.to(self.device)
            model.eval()

            self.models[name] = model

            self.transforms[name] = transforms.Compose([
                transforms.Resize(
                    (config["size"], config["size"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["mean"],
                    std=config["std"],
                ),
            ])

            print(f"PASS — {name} loaded")

    @torch.inference_mode()
    def predict(self, image):

        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(
                "Expected a PIL image or image file path."
            )

        predictions = {}
        votes = []

        for name, model in self.models.items():

            tensor = self.transforms[name](image)
            tensor = tensor.unsqueeze(0).to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = perf_counter()

            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (perf_counter() - start) * 1000

            index = int(probabilities.argmax().item())
            predicted_class = CLASS_NAMES[index]

            confidence = float(probabilities[index].item())

            predictions[name] = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    class_name: float(probabilities[i].item())
                    for i, class_name in enumerate(CLASS_NAMES)
                },
                "inference_ms": round(elapsed_ms, 2),
            }

            votes.append(predicted_class)

        counts = Counter(votes)
        ensemble_prediction, vote_count = counts.most_common(1)[0]

        if vote_count < 2:
            ensemble_prediction = "inconclusive"

        return {
            "ensemble_prediction": ensemble_prediction,
            "agreement": f"{vote_count}/3",
            "predictions": predictions,
        }
