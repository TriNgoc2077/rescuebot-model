import torch
print("CUDA available:", torch.cuda.is_available())
print("DirectML device:", torch.device("dml") if torch.backends.directml.is_available() else "dml not available")
