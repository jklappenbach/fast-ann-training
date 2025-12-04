"""
SPELA(O) Trainer for PyTorch nn.Module networks.

Implements the Solo Pass Embedded Learning Algorithm using local, per-layer
cosine-similarity losses with symmetric class vectors. This trainer performs a
single forward sweep across a provided, ordered list of layers and updates each
layer immediately using its local loss, without backpropagating through other
layers.

References: docs/spela-paper.txt (SPELA / SPELA(O))

Usage (MLP example):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.spela_train import SpelaTrainer, SpelaConfig

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 1024)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 10)

    model = MLP()
    layers = [model.fc1, model.fc2]                 # ordered list of trainable layers
    activations = [nn.ReLU(), nn.Identity()]        # activation after each layer

    cfg = SpelaConfig(num_classes=10, lr=1e-2)
    trainer = SpelaTrainer(model, layers, activations, cfg)

    # x: (B, 1, 28, 28) -> flatten to (B, 784) before passing if using this MLP
    # dataloader should yield (inputs, labels)
    trainer.fit(train_loader, epochs=10)

Notes:
- This trainer operates layer-wise with the provided list of layers in forward order.
- For non-MLP networks (e.g., ConvNets), ensure that each provided layer's output
  is a 1D feature vector per sample before loss (or provide a suitable activation
  that reduces to a vector, e.g., AdaptiveAvgPool and flatten). If not, set
  auto_flatten=True in the config to flatten to vectors automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpelaConfig:
    num_classes: int
    lr: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    auto_flatten: bool = True
    normalize_layer_inputs: bool = True
    grad_clip_norm: Optional[float] = None
    amp: bool = False
    embeddings_method: str = "farthest"  # "farthest" or "random"
    embeddings_candidates: int = 2048     # for farthest method
    seed: Optional[int] = None


def _resolve_device_dtype(device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Tuple[torch.device, Optional[torch.dtype]]:
    dev = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return dev, dtype


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 2:
        return x
    return x.flatten(start_dim=1)


def _normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


@torch.no_grad()
def generate_symmetric_vectors(num_classes: int, dim: int, method: str = "farthest", num_candidates: int = 2048,
                               device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
                               seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate N unit vectors in R^dim to serve as class "symmetric vectors".

    Methods:
    - "random": random gaussian vectors normalized to unit norm.
    - "farthest": greedy farthest-point sampling on the unit sphere from a candidate pool.

    Returns tensor of shape (num_classes, dim).
    """
    assert num_classes > 0 and dim > 0
    dev = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(seed)

    if method == "random" or num_classes >= num_candidates:
        m = torch.randn((num_classes, dim), device=dev, dtype=dtype, generator=gen)
        return _normalize(m, dim=1)

    # candidate pool
    cand = torch.randn((num_candidates, dim), device=dev, dtype=dtype, generator=gen)
    cand = _normalize(cand, dim=1)

    # pick first randomly
    idx0 = torch.randint(0, num_candidates, (1,), device=dev, generator=gen).item()
    chosen = [idx0]
    chosen_vecs = [cand[idx0]]

    # iterative farthest-point selection based on cosine distance (1 - cosine sim)
    sims = cand @ cand[idx0].unsqueeze(1)  # (C,1)
    min_sim = sims.squeeze(1)  # running max of similarity closeness; we keep minimum similarity across chosen

    for _ in range(1, num_classes):
        # pick candidate with smallest max similarity to chosen (i.e., farthest on sphere)
        # we maintain min_sim as the maximum similarity to the chosen set's closest angle; to be robust recompute
        # current farthest index
        # Since we track min over chosen similarities, pick argmin(min_sim)
        far_idx = torch.argmin(min_sim).item()
        chosen.append(far_idx)
        v = cand[far_idx]
        chosen_vecs.append(v)
        # update min_sim with new chosen vector
        sims = cand @ v.unsqueeze(1)
        min_sim = torch.minimum(min_sim, sims.squeeze(1))

    vecs = torch.stack(chosen_vecs, dim=0)
    return _normalize(vecs, dim=1)


class SpelaTrainer:
    """
    SPELA(O) trainer operating on an ordered list of layers from an nn.Module.

    Requirements:
    - layers: a list of modules that can be called sequentially to transform the input.
      Their outputs must be feature vectors per sample (B, D). If not, set cfg.auto_flatten=True
      or provide activations that reduce to (B, D).
    - activations: optional list of callables or nn.Modules applied after each layer output.
    - The remaining parts of the model (if any) are ignored during training with SPELA.
    """

    def __init__(self,
                 model: nn.Module,
                 layers: Sequence[nn.Module],
                 activations: Optional[Sequence[Callable[[torch.Tensor], torch.Tensor]]] = None,
                 cfg: Optional[SpelaConfig] = None):
        if cfg is None:
            raise ValueError("SpelaConfig must be provided")
        if len(layers) == 0:
            raise ValueError("At least one layer must be provided for SPELA training.")
        if activations is not None and len(activations) != len(layers):
            raise ValueError("activations, if provided, must have the same length as layers.")

        self.model = model
        self.layers: List[nn.Module] = list(layers)
        self.acts: List[Optional[Callable[[torch.Tensor], torch.Tensor]]] = list(activations) if activations is not None else [None] * len(layers)
        self.cfg = cfg

        self.device, self.dtype = _resolve_device_dtype(cfg.device, cfg.dtype)
        self.model.to(self.device)
        if self.dtype is not None:
            self.model.to(self.dtype)

        # Per-layer class vectors
        self.class_vectors: List[torch.Tensor] = []  # list of (num_classes, dim)
        # Per-layer optimizers (SGD by default)
        self.optimizers: List[torch.optim.Optimizer] = []

        # Initialize embeddings and optimizers
        self._init_layer_state()

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

    def _layer_output_dim(self, layer: nn.Module, sample_shape: Tuple[int, ...]) -> int:
        """Infer output feature dimension D for a layer from a dummy input with batch size 1."""
        with torch.no_grad():
            dummy = torch.zeros((1, *sample_shape), device=self.device, dtype=self.dtype or torch.float32)
            out = layer(dummy)
            if self.cfg.auto_flatten and out.dim() > 2:
                out = _flatten_features(out)
            if out.dim() != 2:
                raise RuntimeError("Layer output is not 2D (B, D). Provide an activation to reduce or enable auto_flatten.")
            return out.shape[1]

    def _init_layer_state(self):
        # Try to infer an input sample shape for dimension probing from first layer's in_features if available
        inferred_in_dim: Optional[int] = getattr(self.layers[0], 'in_features', None)
        sample_shape: Optional[Tuple[int, ...]]
        if inferred_in_dim is not None:
            sample_shape = (inferred_in_dim,)
        else:
            # Fallback: cannot infer; will defer embedding creation until first batch
            sample_shape = None

        # Create placeholder; embeddings may be delayed until first batch if shape unknown
        self._dims: List[Optional[int]] = []
        for _ in self.layers:
            self._dims.append(None)
            self.class_vectors.append(torch.empty(0))
            # create optimizer for that layer's parameters
            self.optimizers.append(torch.optim.SGD(
                filter(lambda p: p.requires_grad, _ .parameters()),
                lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay
            ))

        # If sample shape is known, initialize now
        if sample_shape is not None:
            self._ensure_embeddings(sample_shape)

    def _ensure_embeddings(self, input_sample_shape: Tuple[int, ...]):
        # Determine per-layer output dims and create vectors if not yet created
        for i, layer in enumerate(self.layers):
            if self._dims[i] is None or self.class_vectors[i].numel() == 0:
                D = self._layer_output_dim(layer, input_sample_shape)
                self._dims[i] = D
                vecs = generate_symmetric_vectors(
                    self.cfg.num_classes, D,
                    method=self.cfg.embeddings_method,
                    num_candidates=self.cfg.embeddings_candidates,
                    device=self.device, dtype=self.dtype, seed=self.cfg.seed,
                )
                self.class_vectors[i] = vecs

    def _layer_forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        y = self.layers[layer_idx](x)
        act = self.acts[layer_idx]
        if act is not None:
            y = act(y)
        if self.cfg.auto_flatten:
            y = _flatten_features(y)
        return y

    def _local_loss(self, layer_idx: int, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # h: (B, D) possibly unnormalized; compute -cosine similarity to class vectors
        # select target vectors per sample
        vecs = self.class_vectors[layer_idx]  # (C, D)
        y_vec = vecs[labels]                  # (B, D)
        # normalize both
        h_n = _normalize(h, dim=1)
        y_n = _normalize(y_vec, dim=1)
        cos = (h_n * y_n).sum(dim=1)  # (B,)
        loss = -cos.mean()
        return loss

    def train_epoch(self, dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            if self.dtype is not None:
                x = x.to(self.dtype)
            y = y.to(self.device)

            # Initialize embeddings lazily if needed (infer shapes from batch)
            if self._dims[0] is None:
                in_shape = tuple(x.shape[1:])
                self._ensure_embeddings(in_shape)

            # propagate through layers sequentially, updating each with local loss
            h = x
            if self.cfg.auto_flatten and h.dim() > 2 and getattr(self.layers[0], 'in_features', None) is not None:
                # If first layer is Linear expecting (B, D), flatten inputs
                h = _flatten_features(h)

            for i, layer in enumerate(self.layers):
                # normalize previous activation if requested
                h_in = _normalize(h, dim=1) if self.cfg.normalize_layer_inputs and h.dim() >= 2 else h
                # Detach so gradients do not flow to previous layers
                h_in = h_in.detach()

                # zero grads for this layer only
                opt = self.optimizers[i]
                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.cfg.amp, dtype=(self.dtype or torch.float32)):
                    h_out = self._layer_forward(i, h_in)
                    loss = self._local_loss(i, h_out, y)

                # backward and step for this layer only
                if self.cfg.amp:
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip_norm is not None:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.layers[i].parameters(), self.cfg.grad_clip_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.layers[i].parameters(), self.cfg.grad_clip_norm)
                    opt.step()

                # pass detached activation to next layer
                h = h_out.detach()

                total_loss += loss.detach().item()

            n_batches += 1

        return total_loss / max(1, n_batches * len(self.layers))

    @torch.no_grad()
    def predict_layer(self, x: torch.Tensor, from_layer: int = -1) -> torch.Tensor:
        """
        Predict class indices using the nearest symmetric vector at the specified layer.
        from_layer: index in [0, L-1] or -1 for last layer.
        Returns: (B,) int64 tensor of predicted class indices.
        """
        self.model.eval()
        dev = self.device
        x = x.to(dev)
        if self.dtype is not None:
            x = x.to(self.dtype)

        h = x
        if self.cfg.auto_flatten and h.dim() > 2 and getattr(self.layers[0], 'in_features', None) is not None:
            h = _flatten_features(h)

        L = len(self.layers)
        target_layer = (L - 1) if from_layer == -1 else from_layer
        for i in range(target_layer + 1):
            h = self._layer_forward(i, _normalize(h, dim=1) if self.cfg.normalize_layer_inputs else h)

        h_n = _normalize(_flatten_features(h), dim=1)
        vecs = self.class_vectors[target_layer]  # (C, D)
        vecs_n = _normalize(vecs, dim=1)
        # cosine sims: (B, D) x (D, C) -> (B, C)
        sims = h_n @ vecs_n.t()
        preds = sims.argmax(dim=1)
        return preds

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], from_layer: int = -1) -> float:
        correct = 0
        total = 0
        for x, y in dataloader:
            y = y.to(self.device)
            preds = self.predict_layer(x, from_layer)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / max(1, total)

    def fit(self, dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], epochs: int = 1,
            val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
            log_fn: Optional[Callable[[int, float, Optional[float]], None]] = None):
        for e in range(1, epochs + 1):
            loss = self.train_epoch(dataloader)
            acc = None
            if val_loader is not None:
                acc = self.evaluate(val_loader, from_layer=-1)
            if log_fn is not None:
                log_fn(e, loss, acc)


__all__ = [
    "SpelaConfig",
    "SpelaTrainer",
    "generate_symmetric_vectors",
]
