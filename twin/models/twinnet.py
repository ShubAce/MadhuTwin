"""TwinNet: physics-gated multimodal fusion network.

    dynamic stream (6 h x 21 channels) --causal dilated TCN--+--FiLM(EHR)--> temporal summary
    static EHR vector -------------------MLP-----------------+                       |
    mechanistic twin forecast + synced state --MLP------------------------------------+
                                                                                      v
                                     fusion MLP --> median delta, P10/P90 spreads, spike & hypo logits
                                                --> per-horizon gate g in [0,1]

    median(h) = g(h) * twin_forecast(h) + correction(h)

The gate is a learned, per-horizon trust in the mechanistic twin: the network leans on
physics where it helps (typically longer horizons) and on data where it does not.
Training uses modality dropout (wearables, events, EHR, physics randomly removed) so the
same model degrades gracefully when a sensor or the EHR is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from twin.data.windows import CH, DYN_CHANNELS, HISTORY, MODALITIES, STATIC_NAMES, H, PatientArrays

N_UKF = 5
STATIC_DIM = len(STATIC_NAMES)
N_STATIC_NUMERIC = sum(1 for n in STATIC_NAMES if n.endswith("_missing"))


class CausalBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.norm = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.gelu(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        h = self.drop(self.conv2(nn.functional.pad(h, (self.pad, 0))))
        return self.norm(x + h)


@dataclass
class TwinNetConfig:
    hidden: int = 64
    static_hidden: int = 64
    fusion: int = 192
    dropout: float = 0.1
    use_physics: bool = True
    use_static: bool = True
    use_wearable: bool = True
    use_events: bool = True


class TwinNet(nn.Module):
    def __init__(self, cfg: TwinNetConfig | None = None):
        super().__init__()
        cfg = cfg or TwinNetConfig()
        self.cfg = cfg
        c, hs = cfg.hidden, cfg.static_hidden
        self.inp = nn.Linear(len(DYN_CHANNELS), c)
        self.blocks = nn.ModuleList([CausalBlock(c, d, cfg.dropout) for d in (1, 2, 4, 8, 16)])
        self.static = nn.Sequential(nn.Linear(STATIC_DIM + 1, hs), nn.GELU(), nn.Linear(hs, hs), nn.GELU())
        self.film = nn.Linear(hs, 2 * c)
        self.attn = nn.Linear(c, 1)
        self.phys = nn.Sequential(nn.Linear(H + N_UKF + 1, hs), nn.GELU(), nn.Linear(hs, hs), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2 * c + 2 * hs, cfg.fusion), nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.fusion, cfg.fusion), nn.GELU())
        self.median = nn.Linear(cfg.fusion, H)
        self.spread = nn.Linear(cfg.fusion, 2 * H)
        self.gate = nn.Linear(cfg.fusion, H)
        self.events = nn.Linear(cfg.fusion, 2)

    def forward(self, dyn, static, physics, ukf, phys_avail, static_avail):
        # dyn (B, T, C); static (B, S); physics (B, H); ukf (B, 5); *_avail (B, 1)
        x = self.inp(dyn).transpose(1, 2)
        s = self.static(torch.cat([static * static_avail, static_avail], dim=1))
        gamma, beta = self.film(s).chunk(2, dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == 1:
                x = x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        last = x[:, :, -1]
        w = torch.softmax(self.attn(x.transpose(1, 2)).squeeze(-1), dim=1)
        pooled = (x * w.unsqueeze(1)).sum(-1)
        p = self.phys(torch.cat([physics * phys_avail, ukf * phys_avail, phys_avail], dim=1))
        z = self.fuse(torch.cat([last, pooled, s, p], dim=1))
        gate = torch.sigmoid(self.gate(z)) * phys_avail
        med = gate * physics + self.median(z)
        lo_hi = nn.functional.softplus(self.spread(z)).view(-1, 2, H) + 1e-3
        q = torch.stack([med - lo_hi[:, 0], med, med + lo_hi[:, 1]], dim=-1)  # (B, H, 3)
        return q, self.events(z), gate


# ----------------------------------------------------------------------------------------------- data
class Windows:
    """All anchors of a set of patients as flat arrays; windows are gathered in vectorised batches."""

    FIELDS = ("physics", "ukf", "y", "spike", "spike_ok", "hypo", "hypo_ok", "now", "anchor")

    def __init__(self, arrays: list[PatientArrays], after_calibration: bool = False, physics: str = "physics",
                 before_calibration: bool = False, max_bin: int | None = None):
        dyn, gbin, pat = [], [], []
        cols: dict[str, list] = {k: [] for k in self.FIELDS}
        off = 0
        pad = np.zeros((HISTORY, len(DYN_CHANNELS)), dtype=np.float32)  # "no data" before each record
        for i, a in enumerate(arrays):
            if after_calibration:
                sel = np.flatnonzero(a.anchors >= a.calib_bins)
            elif before_calibration:  # the patient's own calibration period (for personal fine-tuning)
                # targets look 2 h ahead, so stop early enough that no label crosses into the evaluation period
                limit = (a.calib_bins if max_bin is None else max_bin) - H
                sel = np.flatnonzero(a.anchors < limit)
            else:
                sel = np.arange(len(a.anchors))
            dyn.append(pad)
            off += HISTORY
            dyn.append(a.dyn)
            gbin.append(a.anchors[sel] + off)
            pat.append(np.full(len(sel), i))
            cols["physics"].append(getattr(a, physics)[sel])
            cols["ukf"].append(a.ukf[sel])
            for k in ("y", "spike", "spike_ok", "hypo", "hypo_ok"):
                cols[k].append(getattr(a, k)[sel])
            cols["now"].append(a.cgm[a.anchors[sel]])
            cols["anchor"].append(a.anchors[sel])
            off += len(a.dyn)
        self.arrays = arrays
        self.dyn = np.concatenate(dyn).astype(np.float32)
        self.gbin = np.concatenate(gbin)                   # global bin index of each anchor in self.dyn
        self.patient = np.concatenate(pat)
        self.static = np.stack([a.static for a in arrays])[self.patient].astype(np.float32)
        for k in self.FIELDS:
            v = np.concatenate(cols[k])
            if k == "ukf":
                v = np.nan_to_num(v)
            setattr(self, k, v if k in ("now", "anchor") else v.astype(np.float32))
        self.group = np.array([arrays[i].group for i in self.patient], dtype=object)

    def __len__(self) -> int:
        return len(self.gbin)

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        g = self.gbin[idx]
        win = self.dyn[g[:, None] + np.arange(-HISTORY + 1, 1)[None, :]]
        return {"dyn": win, "static": self.static[idx], "physics": np.nan_to_num(self.physics[idx]),
                "ukf": self.ukf[idx], "y": self.y[idx], "spike": self.spike[idx], "spike_ok": self.spike_ok[idx],
                "hypo": self.hypo[idx], "hypo_ok": self.hypo_ok[idx]}


WEARABLE_IDX = [CH[c] for c in MODALITIES["wearable"]]
EVENT_IDX = [CH[c] for c in MODALITIES["events"]]


def apply_modalities(b: dict, cfg: TwinNetConfig, rng: np.random.Generator | None = None, p_drop: float = 0.15) -> dict:
    """Remove modalities the config excludes and, during training, drop others at random."""
    n = len(b["dyn"])
    dyn = b["dyn"].copy()

    def drop(p: float, enabled: bool) -> np.ndarray:
        """Rows from which a modality is removed: all rows if disabled, a random subset in training."""
        if not enabled:
            return np.ones(n, bool)
        return rng.random(n) < p if rng is not None else np.zeros(n, bool)

    w = drop(p_drop, cfg.use_wearable)
    dyn[np.ix_(w, np.arange(HISTORY), WEARABLE_IDX)] = 0
    e = drop(p_drop, cfg.use_events)
    dyn[np.ix_(e, np.arange(HISTORY), EVENT_IDX)] = 0
    s = drop(0.1, cfg.use_static)
    ph = drop(p_drop, cfg.use_physics)
    static = b["static"].copy()
    static[s] = 0
    out = dict(b, dyn=dyn, static=static)
    out["static_avail"] = (~s).astype(np.float32)[:, None]
    out["phys_avail"] = (~ph).astype(np.float32)[:, None]
    return out


# ------------------------------------------------------------------------------------------- training
QS = torch.tensor([0.1, 0.5, 0.9])


def loss_fn(q, logits, b, event_weight: float = 0.3):
    y = b["y"]
    mask = torch.isfinite(y)
    y0 = torch.nan_to_num(y)
    err = y0.unsqueeze(-1) - q
    pin = torch.maximum(QS * err, (QS - 1) * err).mean(-1)
    reg = (pin * mask).sum() / mask.sum().clamp(min=1)
    ev = 0.0
    for k, name in enumerate(("spike", "hypo")):
        ok = b[f"{name}_ok"]
        if ok.sum() > 0:
            pos = (b[name] * ok).sum().clamp(min=1)
            pw = ((ok.sum() - pos) / pos).clamp(1, 20)
            bce = nn.functional.binary_cross_entropy_with_logits(logits[:, k], b[name], pos_weight=pw, reduction="none")
            ev = ev + (bce * ok).sum() / ok.sum()
    return reg + event_weight * ev


def _to_t(b: dict, device: str) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in b.items()}


def predict(model: TwinNet, data: Windows, idx: np.ndarray | None = None, batch: int = 4096, device: str = "cpu",
            modality_override: TwinNetConfig | None = None) -> dict[str, np.ndarray]:
    model.eval()
    cfg = modality_override or model.cfg
    idx = np.arange(len(data)) if idx is None else idx
    qs, ev, gates = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = _to_t(apply_modalities(data.batch(idx[s : s + batch]), cfg), device)
            q, logits, gate = model(b["dyn"], b["static"], b["physics"], b["ukf"], b["phys_avail"], b["static_avail"])
            qs.append(q.cpu().numpy())
            ev.append(torch.sigmoid(logits).cpu().numpy())
            gates.append(gate.cpu().numpy())
    q = np.concatenate(qs)
    e = np.concatenate(ev)
    return {"q": q, "spike": e[:, 0], "hypo": e[:, 1], "gate": np.concatenate(gates)}


def train(model: TwinNet, train_data: Windows, val_data: Windows | None = None, epochs: int = 8,
          samples_per_epoch: int = 300_000, batch: int = 512, lr: float = 2e-3, weight_decay: float = 1e-4,
          seed: int = 0, device: str = "cpu", log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(min(samples_per_epoch, len(train_data)) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    history, best, best_state = [], float("inf"), None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_data))[: samples_per_epoch]
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch):
            b = _to_t(apply_modalities(train_data.batch(order[s : s + batch]), model.cfg, rng), device)
            q, logits, _ = model(b["dyn"], b["static"], b["physics"], b["ukf"], b["phys_avail"], b["static_avail"])
            loss = loss_fn(q, logits, b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.detach().item()
            nb += 1
        rec = {"epoch": ep + 1, "train_loss": tot / nb}
        if val_data is not None:
            vidx = np.random.default_rng(1).permutation(len(val_data))[:60_000]
            pr = predict(model, val_data, vidx, device=device)
            y = val_data.y[vidx]
            m = np.isfinite(y)
            rec["val_rmse_mgdl"] = float(np.sqrt(np.mean(((pr["q"][..., 1] - y) * 50)[m] ** 2)))
            if rec["val_rmse_mgdl"] < best:
                best = rec["val_rmse_mgdl"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        log(f"    epoch {rec['epoch']}: " + ", ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"))
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_rmse": best}


def personalise(base: TwinNet, arr: PatientArrays, epochs: int = 3, lr: float = 3e-4, max_bin: int | None = None,
                min_windows: int = 40) -> TwinNet:
    """'The twin learns you': fine-tune a copy of the population model on one patient's own
    calibration-period data (no labels from the evaluation period are used)."""
    import copy

    w = Windows([arr], before_calibration=True, max_bin=max_bin)
    if len(w) < min_windows:
        return base
    net = copy.deepcopy(base)
    train(net, w, None, epochs=epochs, samples_per_epoch=len(w), batch=128, lr=lr, log=lambda m: None)
    return net
