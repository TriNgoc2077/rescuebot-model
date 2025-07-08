import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel

class ViTFeatureExtractorModule(nn.Module):
    # Module take in batch RGB images (PIL images or numpy arrays) 
    # and return feature vector (B x hidden_dim) from Vision transformer
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: torch.device = None
    ):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)  
        self.vit = ViTModel.from_pretrained(model_name) if pretrained \
                   else ViTModel(ViTFeatureExtractor.from_pretrained(model_name).config)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.hidden_dim = self.vit.config.hidden_size

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, images):
        # images: list of PIL.Image or numpy.ndarray HxWx3 (uint8) or torch.Tensor Bx3xHxW
        # Return : tensor size (B, hidden_dim)
        # If input is torch.Tensor then convert to numpy for extractor
        needs_numpy = isinstance(images, torch.Tensor)
        if needs_numpy:
            imgs = [img.permute(1,2,0).cpu().numpy() for img in images]
        else:
            imgs = images

        inputs = self.feature_extractor(images=imgs, return_tensors="pt").to(self.device)

        outputs = self.vit(**inputs)
        feat = outputs.pooler_output  

        return feat
