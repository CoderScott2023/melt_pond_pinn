import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from nnlayer import DiffConvCubicBSpline

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def load_taf_strip(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    depth = torch.tensor(np.nan_to_num(d["depth"], nan=0.0), dtype=DTYPE)
    cell_m = float(d["cell_m"])
    cumulative_days = torch.cumsum(torch.tensor(d["dt_days"], dtype=DTYPE), dim=0)
    return depth, cell_m, cumulative_days


def day_to_frame_index(cumulative_days, query_day):
    hi = torch.searchsorted(cumulative_days, query_day).clamp(1, len(cumulative_days) - 1)
    lo = hi - 1
    day_lo, day_hi = cumulative_days[lo], cumulative_days[hi]
    frac = (query_day - day_lo) / (day_hi - day_lo)
    return lo.to(DTYPE) + frac


def build_stencils(field, cell_m, query_tyx):
    T, H, W = field.shape
    days, rows, cols = torch.arange(T, dtype=DTYPE), torch.arange(H, dtype=DTYPE), torch.arange(W, dtype=DTYPE)

    nearest_t = DiffConvCubicBSpline.get_nearest_index(days, query_tyx[:, 0]).clamp(1, T - 2)
    nearest_r = DiffConvCubicBSpline.get_nearest_index(rows, query_tyx[:, 1]).clamp(1, H - 2)
    nearest_c = DiffConvCubicBSpline.get_nearest_index(cols, query_tyx[:, 2]).clamp(1, W - 2)

    y = torch.stack([
        field[t - 1:t + 2, r - 1:r + 2, c - 1:c + 2]
        for t, r, c in zip(nearest_t.tolist(), nearest_r.tolist(), nearest_c.tolist())
    ])
    x = torch.stack([nearest_t, nearest_r * cell_m, nearest_c * cell_m], dim=-1)
    x_eval = torch.stack([query_tyx[:, 0], query_tyx[:, 1] * cell_m, query_tyx[:, 2] * cell_m], dim=-1)
    return x, y, x_eval


def build_stencils_2d(field2d, cell_m, query_rc):
    H, W = field2d.shape
    rows, cols = torch.arange(H, dtype=DTYPE), torch.arange(W, dtype=DTYPE)

    nearest_r = DiffConvCubicBSpline.get_nearest_index(rows, query_rc[:, 0]).clamp(1, H - 2)
    nearest_c = DiffConvCubicBSpline.get_nearest_index(cols, query_rc[:, 1]).clamp(1, W - 2)

    y = torch.stack([
        field2d[r - 1:r + 2, c - 1:c + 2]
        for r, c in zip(nearest_r.tolist(), nearest_c.tolist())
    ])
    x = torch.stack([nearest_r * cell_m, nearest_c * cell_m], dim=-1)
    x_eval = torch.stack([query_rc[:, 0] * cell_m, query_rc[:, 1] * cell_m], dim=-1)
    return x, y, x_eval


class meltpond_PINN(nn.Module):
    def __init__(self, dx, dx_topo=None, n_copies=8):
        super(meltpond_PINN, self).__init__()

        self.n_copies = n_copies
        self.conv = DiffConvCubicBSpline(dx)
        self.conv_topo = DiffConvCubicBSpline(dx_topo) if dx_topo is not None else None

        n_features = 2 if self.conv_topo is not None else 1
        self.hidden_layer1 = nn.Linear(n_copies * n_features + 1, 32)  # +1 for raw dt
        self.hidden_layer2 = nn.Linear(32, 32)
        self.hidden_layer3 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = nn.Tanh()

    def forward(self, x, y, t, context, dt, topo_context=None):
        if x.ndim != 1:
            raise ValueError
        if y.ndim != 1:
            raise ValueError
        if t.ndim != 1:
            raise ValueError
        if dt.ndim != 1:
            raise ValueError

        batch_size = x.shape[0]
        if y.shape[0] != batch_size:
            raise ValueError('Received incorrect batch size for y!')
        if t.shape[0] != batch_size:
            raise ValueError('Received incorrect batch size for t!')
        if context.shape[0] != batch_size:
            raise ValueError('Received incorrect batch size for context!')
        if dt.shape[0] != batch_size:
            raise ValueError('Received incorrect batch size for dt!')

        x_eval = torch.stack((t, x, y), dim=-1)
        x_grid = torch.round(x_eval / self.conv.dx) * self.conv.dx
        y_stencil = context.reshape(batch_size, 3, 3, 3)

        depth_feat = self.conv(x_grid, y_stencil, x_eval)  # (batch,)
        feats = [torch.tile(depth_feat.unsqueeze(-1), (1, self.n_copies))]

        if self.conv_topo is not None:
            if topo_context is None:
                raise ValueError('Model has terrain enabled but no topo_context was given!')
            if topo_context.shape[0] != batch_size:
                raise ValueError('Received incorrect batch size for topo_context!')

            xy_eval = torch.stack((x, y), dim=-1)
            xy_grid = torch.round(xy_eval / self.conv_topo.dx) * self.conv_topo.dx
            topo_stencil = topo_context.reshape(batch_size, 3, 3)

            topo_feat = self.conv_topo(xy_grid, topo_stencil, xy_eval)  # (batch,)
            feats.append(torch.tile(topo_feat.unsqueeze(-1), (1, self.n_copies)))

        feats.append(dt.unsqueeze(-1))  # raw, un-interpolated -- the only carrier of "how far ahead"

        out = torch.cat(feats, dim=-1)
        out = self.activation(self.hidden_layer1(out))
        out = self.activation(self.hidden_layer2(out))
        out = self.activation(self.hidden_layer3(out))
        predicted_depth = self.output_layer(out)
        return predicted_depth


def sample_points(n, cumulative_days, H, W, margin=2):
    day = torch.empty(n).uniform_(float(cumulative_days[2]), float(cumulative_days[-2]))
    row = torch.empty(n).uniform_(margin, H - 1 - margin)
    col = torch.empty(n).uniform_(margin, W - 1 - margin)
    return day, row, col


def train_on_strip_causal(model, opt, npz_path, n_steps=100, batch_size=256, log_every=None,
                           loader=load_taf_strip, max_dt=None):
    depth, cell_m, cumulative_days = loader(npz_path)
    T, H, W = depth.shape
    max_dt = max_dt if max_dt is not None else float(cumulative_days[-1])

    last_loss = None
    for step in range(n_steps):
        anchor_day, row, col = sample_points(batch_size, cumulative_days, H, W)
        anchor_frame_idx = day_to_frame_index(cumulative_days, anchor_day)

        dt = torch.empty(batch_size).uniform_(0.0, max_dt)
        target_day = (anchor_day + dt).clamp(max=float(cumulative_days[-2]))
        dt = target_day - anchor_day  # re-derive after the clamp so it stays exact
        target_frame_idx = day_to_frame_index(cumulative_days, target_day)

        _, y_a, _ = build_stencils(depth, cell_m, torch.stack([anchor_frame_idx, row, col], dim=-1))
        context = y_a.reshape(y_a.shape[0], -1)

        x_t, y_t, x_eval_t = build_stencils(depth, cell_m, torch.stack([target_frame_idx, row, col], dim=-1))
        with torch.no_grad():
            z_true = model.conv(x_t, y_t, x_eval_t)

        z_pred = model(row * cell_m, col * cell_m, anchor_frame_idx, context, dt).squeeze(-1)
        loss = ((z_pred - z_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        last_loss = loss.item()
        if log_every and step % log_every == 0:
            print(f"    step {step:4d}  loss {last_loss:.6f}")
    return last_loss


def train_on_strip_causal_with_topo(model, opt, npz_path, loader, n_steps=100, batch_size=256,
                                     log_every=None, max_dt=None):
    depth, topo, cell_m, cumulative_days = loader(npz_path)
    T, H, W = depth.shape
    max_dt = max_dt if max_dt is not None else float(cumulative_days[-1])

    last_loss = None
    for step in range(n_steps):
        anchor_day, row, col = sample_points(batch_size, cumulative_days, H, W)
        anchor_frame_idx = day_to_frame_index(cumulative_days, anchor_day)

        dt = torch.empty(batch_size).uniform_(0.0, max_dt)
        target_day = (anchor_day + dt).clamp(max=float(cumulative_days[-2]))
        dt = target_day - anchor_day
        target_frame_idx = day_to_frame_index(cumulative_days, target_day)

        _, y_a, _ = build_stencils(depth, cell_m, torch.stack([anchor_frame_idx, row, col], dim=-1))
        context = y_a.reshape(y_a.shape[0], -1)

        query_rc = torch.stack([row, col], dim=-1)
        _, y_topo, _ = build_stencils_2d(topo, cell_m, query_rc)
        topo_context = y_topo.reshape(y_topo.shape[0], -1)

        x_t, y_t, x_eval_t = build_stencils(depth, cell_m, torch.stack([target_frame_idx, row, col], dim=-1))
        with torch.no_grad():
            z_true = model.conv(x_t, y_t, x_eval_t)

        z_pred = model(row * cell_m, col * cell_m, anchor_frame_idx, context, dt, topo_context=topo_context).squeeze(-1)
        loss = ((z_pred - z_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        last_loss = loss.item()
        if log_every and step % log_every == 0:
            print(f"    step {step:4d}  loss {last_loss:.6f}")
    return last_loss


def predict_full_frame(model, depth_source, cell_m, t_anchor, dt_scalar, topo_source=None, margin=2):
    H, W = depth_source.shape[-2], depth_source.shape[-1]
    rows = torch.arange(margin, H - margin, dtype=DTYPE)
    cols = torch.arange(margin, W - margin, dtype=DTYPE)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    row, col = grid_r.reshape(-1), grid_c.reshape(-1)
    n = row.shape[0]

    anchor_frame_idx = torch.full((n,), float(t_anchor), dtype=DTYPE)
    _, y, _ = build_stencils(depth_source, cell_m, torch.stack([anchor_frame_idx, row, col], dim=-1))
    context = y.reshape(y.shape[0], -1)
    dt = torch.full((n,), float(dt_scalar), dtype=DTYPE)

    topo_context = None
    if topo_source is not None:
        query_rc = torch.stack([row, col], dim=-1)
        _, y_topo, _ = build_stencils_2d(topo_source, cell_m, query_rc)
        topo_context = y_topo.reshape(y_topo.shape[0], -1)

    with torch.no_grad():
        pred = model(row * cell_m, col * cell_m, anchor_frame_idx, context, dt, topo_context=topo_context).squeeze(-1)

    full = torch.full((H, W), float("nan"), dtype=DTYPE)
    full[margin:H - margin, margin:W - margin] = pred.reshape(H - 2 * margin, W - 2 * margin)
    return full


def predict_full_frame_from_2d(model, field2d, cell_m, dt_scalar, topo_source=None, margin=2):
    fake_stack = field2d.unsqueeze(0).repeat(3, 1, 1)
    return predict_full_frame(model, fake_stack, cell_m, t_anchor=1.0, dt_scalar=dt_scalar,
                               topo_source=topo_source, margin=margin)


def rollout_chained(model, depth, cell_m, cumulative_days, topo=None, margin=2):
    field = depth[0]
    preds = []
    for k in range(1, depth.shape[0]):
        dt_step = float(cumulative_days[k] - cumulative_days[k - 1])
        field = predict_full_frame_from_2d(model, field, cell_m, dt_step, topo_source=topo, margin=margin)
        preds.append(field)
    return preds


def rollout_direct(model, depth, cell_m, cumulative_days, topo=None, margin=2):
    preds = []
    for k in range(1, depth.shape[0]):
        dt_direct = float(cumulative_days[k] - cumulative_days[0])
        preds.append(predict_full_frame(model, depth, cell_m, t_anchor=0.0, dt_scalar=dt_direct,
                                         topo_source=topo, margin=margin))
    return preds


def rmse_vs_real(pred, real, crop):
    p = pred[crop:-crop, crop:-crop]
    r = real[crop:-crop, crop:-crop]
    valid = ~torch.isnan(p)
    return torch.sqrt(((p[valid] - r[valid]) ** 2).mean()).item()


if __name__ == "__main__":

    all_strips = glob.glob(os.path.join(HERE, "pipeline", "data", "taf_strips", "tafstrip_*.npz"))
    random.Random(0).shuffle(all_strips)
    n_strips = 200

    dx = torch.tensor([1.0, 30.0, 30.0], dtype=DTYPE)
    model = meltpond_PINN(dx).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i, npz_path in enumerate(all_strips[:n_strips]):
        final_loss = train_on_strip_causal(model, opt, npz_path, n_steps=100, batch_size=256)
        print(f"strip {i:4d}/{n_strips}  {os.path.basename(npz_path):45s}  final loss {final_loss:.6f}")
