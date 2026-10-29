"""LSTM autoencoder over 30-day behaviour sequences (PyTorch).

encoder LSTM (layers x hidden) -> last hidden state -> linear bottleneck -> repeated over
the sequence -> decoder LSTM -> linear head reconstructing every channel. Trained on
pooled training-window sequences (mostly "normal" behaviour); a large reconstruction
error means unusual behaviour.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from smd.config import LstmAECfg


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, hidden: int = 64, layers: int = 2, bottleneck: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True)
        self.to_latent = nn.Linear(hidden, bottleneck)
        self.from_latent = nn.Linear(bottleneck, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return self.to_latent(h[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, x.shape[1], 1)
        out, _ = self.decoder(dec_in)
        return self.head(out)


@dataclass
class TrainResult:
    model: LSTMAutoencoder
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val: float = float("inf")


def _loss_fn(name: str) -> nn.Module:
    return nn.HuberLoss() if name == "huber" else nn.MSELoss()


def train_autoencoder(x_train: np.ndarray, x_val: np.ndarray, cfg: LstmAECfg, seed: int) -> TrainResult:
    """Train with Adam and early stopping on the validation slice; fully seeded."""
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.set_num_threads(cfg.num_threads)
    n_ch, L = x_train.shape[2], x_train.shape[1]
    model = LSTMAutoencoder(n_ch, L, cfg.hidden, cfg.layers, cfg.bottleneck)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = _loss_fn(cfg.loss)
    gen = torch.Generator().manual_seed(seed)
    xt = torch.from_numpy(x_train)
    xv = torch.from_numpy(x_val) if len(x_val) else xt[: min(len(xt), 2048)]
    res = TrainResult(model)
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        tot = 0.0
        for i in range(0, len(xt), cfg.batch_size):
            xb = xt[perm[i: i + cfg.batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        val = float(reconstruction_errors(model, xv.numpy(), last_k=L, batch=4096).mean())
        res.history.append({"epoch": epoch, "train_loss": tot / len(xt), "val_mse": val})
        if val < res.best_val - 1e-6:
            res.best_val, res.best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return res


@torch.no_grad()
def reconstruction_errors(model: LSTMAutoencoder, x: np.ndarray, last_k: int, batch: int = 4096) -> np.ndarray:
    """Mean squared reconstruction error over the last `last_k` days and all channels."""
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i: i + batch])
        err = (model(xb) - xb) ** 2
        out[i: i + batch] = err[:, -last_k:, :].mean(dim=(1, 2)).numpy()
    return out
