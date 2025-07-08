import torch
from PIL import Image
import numpy as np
from vit_extractor import ViTFeatureExtractorModule

# In this test, we create 2 sample images and send it to extractor
vit_ext = ViTFeatureExtractorModule(freeze_backbone=True)

dummy = [np.ones((224,224,3), dtype=np.uint8)*255 for _ in range(2)]
feats = vit_ext(dummy)  

print("Output shape:", feats.shape)
