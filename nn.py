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


def build_interpolant(cell_m):
    return DiffConvCubicBSpline(torch.tensor([1.0, cell_m, cell_m], dtype=DTYPE))


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


def spatial_context(field3d, query_tyx):
    T, H, W = field3d.shape
    days, rows, cols = torch.arange(T, dtype=DTYPE), torch.arange(H, dtype=DTYPE), torch.arange(W, dtype=DTYPE)
    nearest_t = DiffConvCubicBSpline.get_nearest_index(days, query_tyx[:, 0]).clamp(0, T - 1)
    nearest_r = DiffConvCubicBSpline.get_nearest_index(rows, query_tyx[:, 1]).clamp(1, H - 2)
    nearest_c = DiffConvCubicBSpline.get_nearest_index(cols, query_tyx[:, 2]).clamp(1, W - 2)
    return torch.stack([
        field3d[t, r - 1:r + 2, c - 1:c + 2].reshape(-1)
        for t, r, c in zip(nearest_t.tolist(), nearest_r.tolist(), nearest_c.tolist())
    ])


class meltpond_PINN(nn.Module):
    def __init__(self, context_size=9):
        super(meltpond_PINN, self).__init__()

        self.input_layer = nn.Linear(3 + context_size, 32)
        self.hidden_layer1 = nn.Linear(32, 32)
        self.hidden_layer2 = nn.Linear(32, 32)
        self.hidden_layer3 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = nn.Tanh()

    def forward(self, x, y, t, context):
        inputs = torch.cat((x, y, t, context), dim=1)
        out = self.activation(self.input_layer(inputs))
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


def train_on_strip(model, opt, npz_path, n_steps=100, batch_size=256, log_every=None):
    depth, cell_m, cumulative_days = load_taf_strip(npz_path)
    T, H, W = depth.shape
    conv = build_interpolant(cell_m)

    last_loss = None
    for step in range(n_steps):
        day, row, col = sample_points(batch_size, cumulative_days, H, W)
        frame_idx = day_to_frame_index(cumulative_days, day)
        query = torch.stack([frame_idx, row, col], dim=-1)
        query_prev = torch.stack([frame_idx - 1, row, col], dim=-1)

        with torch.no_grad():
            x, y, x_eval = build_stencils(depth, cell_m, query)
            z_true = conv(x, y, x_eval)
            context = spatial_context(depth, query_prev)

        z_pred = model(row.unsqueeze(1), col.unsqueeze(1), day.unsqueeze(1), context).squeeze(1)
        loss = ((z_pred - z_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        last_loss = loss.item()
        if log_every and step % log_every == 0:
            print(f"    step {step:4d}  loss {last_loss:.6f}")
    return last_loss


if __name__ == "__main__":

    all_strips = glob.glob(os.path.join(HERE, "pipeline", "data", "taf_strips", "tafstrip_*.npz"))
    random.Random(0).shuffle(all_strips)
    n_strips = 200

    model = meltpond_PINN().double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i, npz_path in enumerate(all_strips[:n_strips]):
        final_loss = train_on_strip(model, opt, npz_path, n_steps=100, batch_size=256)
        print(f"strip {i:4d}/{n_strips}  {os.path.basename(npz_path):45s}  final loss {final_loss:.6f}")
