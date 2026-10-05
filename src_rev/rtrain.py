"""
Fast, device-resident training loop (no DataLoader overhead) for the revision runs.

Loss options
  ce    : standard cross-entropy
  cw    : cross-entropy weighted by inverse class frequency, w_c = N / (K n_c)
  focal : focal loss FL = (1 - p_t)^gamma * CE, no alpha weighting (alpha_c = 1 for all c)
  os    : standard cross-entropy with class-balanced random oversampling of the training
          set (sampling probability of a beat proportional to 1 / n_class, with replacement,
          epoch size = training-set size)
"""
import copy
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import rconfig as C
import rdata
import rmetrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss_fn(kind, y_train, device, gamma=2.0):
    if kind == "cw":
        w = torch.tensor(rdata.inverse_frequency_weights(y_train), dtype=torch.float32,
                         device=device)
        return lambda out, y: F.cross_entropy(out, y, weight=w)
    if kind == "focal":
        def focal(out, y):
            ce = F.cross_entropy(out, y, reduction="none")
            pt = torch.exp(-ce)
            return ((1.0 - pt) ** gamma * ce).mean()
        return focal
    return lambda out, y: F.cross_entropy(out, y)          # ce and os


@torch.no_grad()
def predict_proba(model, X, rr, device, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        xb = torch.as_tensor(X[s:s + bs], device=device)
        rb = torch.as_tensor(rr[s:s + bs], device=device) if (rr is not None and model.n_rr) else None
        out.append(torch.softmax(model(xb, rb), dim=1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, C.K), np.float32)


def train_model(model, X, y, rr, device, seed, lr, bs, wd, epochs, kind="ce", gamma=2.0,
                Xval=None, yval=None, rrval=None, patience=None):
    """Train for up to `epochs`. If validation data are given, macro-F1 on the validation
    set is recorded each epoch and early stopping with `patience` is applied (tuning only).
    Returns (model, curve, best_epoch, best_val_macro_f1)."""
    set_seed(seed)
    model = model.to(device)
    Xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(y, device=device)
    Rt = torch.as_tensor(rr, device=device) if (rr is not None and model.n_rr) else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = _loss_fn(kind, y, device, gamma)
    rng = np.random.default_rng(seed)
    n = len(y)
    probs = None
    if kind == "os":
        cnt = np.maximum(rdata.class_counts(y), 1)
        probs = 1.0 / cnt[y]
        probs = probs / probs.sum()
    curve, best, best_ep, bad, best_state = [], -1.0, 0, 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        idx = rng.choice(n, size=n, replace=True, p=probs) if probs is not None else rng.permutation(n)
        idx_t = torch.as_tensor(idx, device=device)
        tot = torch.zeros((), device=device)
        for s in range(0, n, bs):
            b = idx_t[s:s + bs]
            if len(b) < 2:                       # BatchNorm needs > 1 sample
                continue
            out = model(Xt[b], Rt[b] if Rt is not None else None)
            loss = loss_fn(out, yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot = tot + loss.detach() * len(b)
        rec = {"epoch": ep, "train_loss": float(tot.item() / n), "sec": round(time.time() - t0, 3)}
        if Xval is not None:
            pv = predict_proba(model, Xval, rrval, device).argmax(1)
            f1 = rmetrics.summarize(yval, pv)["macro_f1"]
            rec["val_macro_f1"] = f1
            if f1 > best + 1e-9:
                best, best_ep, bad = f1, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        curve.append(rec)
        if Xval is not None and patience and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve, best_ep, best
