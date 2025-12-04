fast-ann-training

Lightweight PyTorch implementation of SPELA(O) — Solo Pass Embedded Learning Algorithm — that trains neural networks with local, per‑layer losses in a single forward sweep (no global backprop). The trainer works with arbitrary `nn.Module` layers (MLPs and CNNs) by optimizing at each layer a cosine‑similarity loss to fixed, per‑layer class embeddings ("symmetric vectors").

What is here
- `src/spela_train.py`: `SpelaTrainer`, `SpelaConfig`, and utilities.
- `docs/spela-paper.txt`: Text version of the paper for reference.
- `docs/2402.09769v2.pdf`: Original PDF referenced by the text.

Requirements
- Python 3.9+
- PyTorch (see requirements.txt)

Quick start
1) Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate 
```

2) Install dependencies
```
pip install -r requirements.txt
# Optional for vision datasets and models
pip install torchvision
```

3) Train a simple MLP on flattened images
```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.spela_train import SpelaTrainer, SpelaConfig

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)

model = MLP()
layers = [model.fc1, model.fc2]
activations = [nn.ReLU(), nn.Identity()]

cfg = SpelaConfig(num_classes=10, lr=1e-2)
trainer = SpelaTrainer(model, layers, activations, cfg)

# train_loader should yield (x, y) where x is (B, 1, 28, 28); the trainer auto‑flattens as needed
trainer.fit(train_loader, epochs=10)
acc = trainer.evaluate(test_loader)
print("test acc:", acc)
```

4) Using CNN blocks
Ensure the output used by the local loss is a vector per sample. A common pattern is to add global average pooling and flatten inside each block.
```
import torch.nn as nn
from src.spela_train import SpelaTrainer, SpelaConfig

block1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),  # -> (B, 32)
)
block2 = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(inplace=True),
)
head = nn.Linear(64, 10)

layers = [block1, block2, head]
activations = [None, None, nn.Identity()]
cfg = SpelaConfig(num_classes=10, lr=5e-3)
trainer = SpelaTrainer(nn.Sequential(), layers, activations, cfg)
```

How it works (brief)
- For each provided layer, the trainer computes the layer output `h` and immediately minimizes a local loss: `-cosine(h, class_vector[y])`.
- Class vectors (one per class per layer) live on the unit sphere and are fixed; two generation methods are provided:
  - `random`: unit‑normalized Gaussian vectors.
  - `farthest` (default): greedy farthest‑point sampling on the sphere for better separation.
- Each layer has its own optimizer and is updated right after its local loss is computed. The activation passed to the next layer is detached to keep learning local (SPELA(O) single forward pass).

Configuration highlights
- `num_classes`: number of classes for the task.
- `lr`, `momentum`, `weight_decay`: SGD hyperparameters (per layer).
- `device`, `dtype`: target device/dtype for the model and computations.
- `auto_flatten=True`: flatten non‑vector outputs to `(B, D)`. For CNNs, prefer global pooling + flatten to keep `D` stable.
- `normalize_layer_inputs=True`: L2‑normalize incoming activation before each layer (per paper’s algorithm).
- `amp`: enable automatic mixed precision (CUDA).
- `grad_clip_norm`: optional gradient clipping per layer.
- `embeddings_method`: `"farthest"` or `"random"` for class vector initialization.
- `seed`: reproducible embeddings.

CNN notes
- Ensure each layer output used for the local loss is a vector per sample.
- Prefer `nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()` inside blocks to stabilize dimensionality across image sizes.
- If you rely on `auto_flatten=True` without pooling, keep input spatial size fixed during training so the class-vector dimension stays constant.

Citation and credits
- This project is inspired by and builds on the ideas from the paper:
  - Aditya Somasundaram, Pushkal Mishra, and Ayon Borthakur. "Learning Using a Single Forward Pass." arXiv:2402.09769. https://arxiv.org/abs/2402.09769

Released under the open source MIT license  
For research and educational purposes.