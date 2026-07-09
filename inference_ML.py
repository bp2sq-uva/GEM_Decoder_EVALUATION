import os
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage
from tqdm.auto import tqdm

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "/volatile/halla/sbs/bhasitha/Tracking_ML/GEM_Decoder_EVALUATION/data_for_inference_ML_withoutROIcut.txt"

OUTPUT_FILE = "scratch/ML/hit_centers_ML.txt"
PRED_MASK_DIR = "scratch/ML/npz"

# CKPT_PATH = "../checkpoints_replayed/nosinglestrips_UNet_asymmfocalloss_withsinglestrips_withstripgaps/latest_model.pt"
# CKPT_PATH = "../checkpoints_replayed/nosinglestrips_UNet_asymmfocalloss_withsinglestrips_withstripgaps_withscheduler/best_candidate_finder.pt"
# CKPT_PATH = "../checkpoints_replayed//UNet_asymmfocalloss_withsinglestrips_cleaneddata_newrecallparams/latest_model.pt"
CKPT_PATH = "/work/halla/sbs/bhasitha/Tracking_ML/GEMDecoder_ML/model_data/model_training/checkpoints_Eval/nosinglestrips_UNet_asymmfocalloss_withsinglestrips_withstripgaps_withscheduler_cleaneddata_reducedgrace_morepositives_newdata/best_candidate_finder.pt"

APPLY_LOG = False
BASE_CHANNELS = 48
CEXTRA = 14

COMPACT_MIN_Q = 0.00001

GAP_MIN = 3
N_SPACER = 3

PRED_THR = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

COLS_INFER = [
    "event_id", "module_id", "strip_id",
    "adc0", "adc1", "adc2", "adc3", "adc4", "adc5"
]

ADC_COLS = ["adc0", "adc1", "adc2", "adc3", "adc4", "adc5"]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_df_inference(df: pd.DataFrame, global_scale=True, pctl=99.5, eps=1e-12):
    df = df.copy()

    for c in ["event_id", "module_id", "strip_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["event_id"]  = df["event_id"].fillna(-1).astype(int)
    df["module_id"] = df["module_id"].fillna(-1).astype(int)
    df["strip_id"]  = df["strip_id"].fillna(-1).astype(int)

    tot = np.nan_to_num(
        df[ADC_COLS].to_numpy(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if global_scale:
        scale = np.percentile(tot, pctl)
        scale = float(max(scale, eps))
        df[ADC_COLS] = tot / scale
    else:
        for ev, idx in df.groupby("event_id").groups.items():
            block_tot = np.nan_to_num(
                df.loc[idx, ADC_COLS].to_numpy(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            scale_e = float(max(np.percentile(block_tot, pctl), eps))
            df.loc[idx, ADC_COLS] = block_tot / scale_e

    df[ADC_COLS] = np.clip(df[ADC_COLS].to_numpy(np.float32), 0.0, 1.5)

    return df


df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    header=None,
    names=COLS_INFER,
    usecols=list(range(len(COLS_INFER)))
)

print("raw rows read:", len(df))
print(df.head())

df = normalize_df_inference(df, global_scale=True)

print("normalization finished")


# ============================================================
# MODEL
# ============================================================

class TemporalEncoder(nn.Module):
    def __init__(self, T=6, out_feats=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, out_feats, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att = nn.Linear(out_feats, 1, bias=False)

    def forward(self, x):
        # x: (B,2,6,H,W)
        B, C, T, H, W = x.shape

        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        f = self.conv(x)

        ft = f.transpose(1, 2)
        w = self.att(ft)
        w = torch.softmax(w.squeeze(-1), dim=1).unsqueeze(1)

        f_att = (f * w).sum(dim=-1)
        f_max = f.amax(dim=-1)

        f2 = torch.cat([f_att, f_max], dim=1)
        F2 = f2.shape[1]

        f2 = f2.view(B, H, W, F2).permute(0, 3, 1, 2).contiguous()
        return f2


class Conv2DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch, base=16):
        super().__init__()

        self.enc1 = Conv2DBlock(in_ch, base)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = Conv2DBlock(base, base * 2)
        self.enc3 = Conv2DBlock(base * 2, base * 4)
        self.bott = Conv2DBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = Conv2DBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = Conv2DBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = Conv2DBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _pad_to_factor(x, factor=8):
        B, C, H, W = x.shape

        padH = (factor - (H % factor)) % factor
        padW = (factor - (W % factor)) % factor

        if padH or padW:
            x = F.pad(x, (0, padW, 0, padH))

        return x, padH, padW, H, W

    def forward(self, x):
        x, pH, pW, H0, W0 = self._pad_to_factor(x, 8)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bott(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.out(d1)

        if pH or pW:
            logits = logits[..., :H0, :W0]

        return logits


class STUNet2p5D(nn.Module):
    def __init__(self, temp_feats=64, base=16, extra_ch=1):
        super().__init__()
        self.temporal = TemporalEncoder(T=6, out_feats=temp_feats)
        self.unet2d = UNet2D(in_ch=2 * temp_feats + extra_ch, base=base)

    def forward(self, x, extra, x_ids=None, y_ids=None):
        feat2d = self.temporal(x)
        feat2d = torch.cat([feat2d, extra], dim=1)
        return self.unet2d(feat2d)


# ============================================================
# STRIP-GAP INFERENCE BUILDER
# ============================================================

def ids_with_spacers(ids, gap_min=3, n_spacer=3):
    """
    Keep observed strip IDs, but insert fake zero-waveform spacer IDs
    if consecutive observed strip IDs are separated by gap_min or more.
    """
    ids = sorted(set(int(i) for i in ids))
    if len(ids) <= 1:
        return ids

    out = [ids[0]]
    fake = -10_000_000

    for a, b in zip(ids[:-1], ids[1:]):
        if (b - a) >= gap_min:
            for _ in range(n_spacer):
                out.append(fake)
                fake -= 1
        out.append(b)

    return out


def build_x3d_and_extra_inference_stripgap(
    event_df: pd.DataFrame,
    apply_log: bool = False,
    min_q: float = 0.00001,
    thr_width: float = 0.11,
    qthr_cluster: float = 0.15,
    shifts: tuple = (-1, 0, 1),
    CLIP_CLUST: int = 8,
    QRATIO_CLIP: float = 10.0,
    gap_min: int = 3,
    n_spacer: int = 3,
):
    """
    New inference builder matching strip-gap training.

    Returns:
      x3d:        (2, 6, H, W)
      extra:      (14, H, W)
      x_ids:      length W, includes fake negative spacer IDs
      y_ids:      length H, includes fake negative spacer IDs
      valid_mask: (H, W), 1 only where both x and y are real strips
    """

    T = len(ADC_COLS)
    Tm1 = max(1, T - 1)

    def with_charge(df_xy: pd.DataFrame) -> pd.DataFrame:
        out = df_xy.copy()
        if out.empty:
            out["qsum"] = np.array([], dtype=float)
            return out

        arr = out[ADC_COLS].to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out["qsum"] = arr.sum(axis=1)
        return out

    def strip_scalar_features(adc_mat: np.ndarray, thr: float):
        qsum  = adc_mat.sum(axis=1).astype(np.float32)
        peak  = adc_mat.max(axis=1).astype(np.float32)
        tpeak = adc_mat.argmax(axis=1).astype(np.int64)
        width = (adc_mat > thr).sum(axis=1).astype(np.int64)
        return qsum, peak, tpeak, width

    def neighbor_cluster_size(strip_ids: np.ndarray, qsum: np.ndarray, qthr: float):
        """
        Important: fake spacer IDs are ignored.
        """
        N = len(strip_ids)
        clust = np.zeros(N, dtype=np.float32)
        if N == 0:
            return clust

        strip_ids = np.asarray(strip_ids)
        real = strip_ids >= 0
        active = (qsum >= qthr) & real

        i = 0
        while i < N:
            if not active[i]:
                i += 1
                continue

            j = i
            while (
                j + 1 < N
                and active[j + 1]
                and real[j]
                and real[j + 1]
                and (strip_ids[j + 1] == strip_ids[j] + 1)
            ):
                j += 1

            size = float(j - i + 1)
            clust[i:j + 1] = size
            i = j + 1

        return clust

    def max_norm_xcorr_shifts(xpart: np.ndarray, ypart: np.ndarray, shifts_, eps: float = 1e-6):
        H_, W_, T_ = xpart.shape
        best = np.full((H_, W_), -1e9, dtype=np.float32)

        for s in shifts_:
            if s == 0:
                xa = xpart
                ya = ypart
            elif s > 0:
                xa = xpart[:, :, :T_ - s]
                ya = ypart[:, :, s:]
            else:
                ss = -s
                xa = xpart[:, :, ss:]
                ya = ypart[:, :, :T_ - ss]

            num = (xa * ya).sum(axis=2)
            den = (np.linalg.norm(xa, axis=2) * np.linalg.norm(ya, axis=2)) + eps
            score = (num / den).astype(np.float32)
            best = np.maximum(best, score)

        return np.clip(best, -1.0, 1.0).astype(np.float32)

    # --------------------------------------------------------
    # Split X and Y
    # --------------------------------------------------------
    x_df = with_charge(event_df[event_df["module_id"] == 0][["strip_id", *ADC_COLS]])
    y_df = with_charge(event_df[event_df["module_id"] == 1][["strip_id", *ADC_COLS]])

    if min_q > 0.0:
        x_df = x_df[x_df["qsum"] >= float(min_q)]
        y_df = y_df[y_df["qsum"] >= float(min_q)]

    if x_df.empty or y_df.empty:
        return None, None, None, None, None

    x_best = x_df.loc[x_df.groupby("strip_id")["qsum"].idxmax()]
    y_best = y_df.loc[y_df.groupby("strip_id")["qsum"].idxmax()]

    Ax = {
        int(r["strip_id"]): np.nan_to_num(
            r[ADC_COLS].to_numpy(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        for _, r in x_best.iterrows()
    }

    Ay = {
        int(r["strip_id"]): np.nan_to_num(
            r[ADC_COLS].to_numpy(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        for _, r in y_best.iterrows()
    }

    # --------------------------------------------------------
    # New part: insert fake spacer IDs exactly like training
    # --------------------------------------------------------
    x_ids = np.array(
        ids_with_spacers(Ax.keys(), gap_min=gap_min, n_spacer=n_spacer),
        dtype=np.int32
    )

    y_ids = np.array(
        ids_with_spacers(Ay.keys(), gap_min=gap_min, n_spacer=n_spacer),
        dtype=np.int32
    )

    W = len(x_ids)
    H = len(y_ids)

    if W == 0 or H == 0:
        return None, None, None, None, None

    if W > 160 or H > 160:
        print(f"[warn] event has H={H}, W={W}, larger than expected search region")

    x_real = x_ids >= 0
    y_real = y_ids >= 0
    valid_mask = (y_real[:, None] & x_real[None, :]).astype(np.uint8)

    # --------------------------------------------------------
    # Build input cube
    # --------------------------------------------------------
    zeroT = np.zeros(T, dtype=np.float32)

    x_adc = np.stack([Ax.get(int(sid), zeroT) for sid in x_ids], axis=0)
    y_adc = np.stack([Ay.get(int(sid), zeroT) for sid in y_ids], axis=0)

    cube = np.zeros((H, W, 2 * T), dtype=np.float32)
    cube[:, :, :T] = x_adc[None, :, :]
    cube[:, :, T:] = y_adc[:, None, :]

    if apply_log:
        cube = np.log1p(cube)

    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

    xpart = cube[:, :, :T]
    ypart = cube[:, :, T:]

    qx, peakx, tpx, widthx = strip_scalar_features(x_adc, thr=float(thr_width))
    qy, peaky, tpy, widthy = strip_scalar_features(y_adc, thr=float(thr_width))

    xn = np.linalg.norm(xpart, axis=2) + 1e-6
    yn = np.linalg.norm(ypart, axis=2) + 1e-6
    dot = (xpart * ypart).sum(axis=2)

    xy_sim = (dot / (xn * yn)).astype(np.float32)
    xcorr = max_norm_xcorr_shifts(xpart, ypart, shifts_=shifts)

    clx = neighbor_cluster_size(x_ids.astype(np.int64), qx, qthr=float(qthr_cluster))
    cly = neighbor_cluster_size(y_ids.astype(np.int64), qy, qthr=float(qthr_cluster))

    qsum_x  = np.broadcast_to(qx[None, :],     (H, W)).astype(np.float32)
    qsum_y  = np.broadcast_to(qy[:, None],     (H, W)).astype(np.float32)
    peak_x  = np.broadcast_to(peakx[None, :],  (H, W)).astype(np.float32)
    peak_y  = np.broadcast_to(peaky[:, None],  (H, W)).astype(np.float32)
    tpeak_x = np.broadcast_to(tpx[None, :],    (H, W)).astype(np.float32)
    tpeak_y = np.broadcast_to(tpy[:, None],    (H, W)).astype(np.float32)
    width_x = np.broadcast_to(widthx[None, :], (H, W)).astype(np.float32)
    width_y = np.broadcast_to(widthy[:, None], (H, W)).astype(np.float32)
    clust_x = np.broadcast_to(clx[None, :],    (H, W)).astype(np.float32)
    clust_y = np.broadcast_to(cly[:, None],    (H, W)).astype(np.float32)

    qsum_x_n  = (qsum_x / float(T)).astype(np.float32)
    qsum_y_n  = (qsum_y / float(T)).astype(np.float32)

    width_x_n = (width_x / float(T)).astype(np.float32)
    width_y_n = (width_y / float(T)).astype(np.float32)

    tpeak_x_n = (tpeak_x / float(Tm1)).astype(np.float32)
    tpeak_y_n = (tpeak_y / float(Tm1)).astype(np.float32)

    clust_x_n = (
        np.clip(clust_x, 0.0, float(CLIP_CLUST)) / float(CLIP_CLUST)
    ).astype(np.float32)

    clust_y_n = (
        np.clip(clust_y, 0.0, float(CLIP_CLUST)) / float(CLIP_CLUST)
    ).astype(np.float32)

    peak_x_n = peak_x.astype(np.float32)
    peak_y_n = peak_y.astype(np.float32)

    dtpeak_abs = (
        np.abs(tpeak_x - tpeak_y) / float(Tm1)
    ).astype(np.float32)

    qratio = (qsum_x / (qsum_y + 1e-6)).astype(np.float32)
    qratio = np.clip(qratio, 0.0, float(QRATIO_CLIP)).astype(np.float32)
    qratio = np.log1p(qratio).astype(np.float32)
    qratio = (qratio / np.log1p(float(QRATIO_CLIP))).astype(np.float32)

    # --------------------------------------------------------
    # New part: image-coordinate position channels
    # Do NOT use raw strip IDs because fake spacer IDs are negative.
    # --------------------------------------------------------
    x_pos = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    y_pos = np.linspace(-1.0, 1.0, H, dtype=np.float32)

    x_pos_ch = np.broadcast_to(x_pos[None, :], (H, W)).astype(np.float32)
    y_pos_ch = np.broadcast_to(y_pos[:, None], (H, W)).astype(np.float32)

    extra = np.stack(
        [
            xy_sim,
            xcorr,
            qsum_x_n,
            qsum_y_n,
            peak_x_n,
            peak_y_n,
            width_x_n,
            width_y_n,
            clust_x_n,
            clust_y_n,
            dtpeak_abs,
            qratio,
            x_pos_ch,
            y_pos_ch,
        ],
        axis=0,
    ).astype(np.float32)

    assert extra.shape[0] == CEXTRA, f"extra has {extra.shape[0]} channels, expected {CEXTRA}"

    x6 = np.moveaxis(xpart, -1, 0)
    y6 = np.moveaxis(ypart, -1, 0)

    x3d = np.stack([x6, y6], axis=0).astype(np.float32)

    return x3d, extra, x_ids, y_ids, valid_mask


# ============================================================
# LOAD CHECKPOINT
# ============================================================
# ============================================================
# LOAD MODEL
# ============================================================

ckpt = torch.load(CKPT_PATH, map_location=device)

print("Loaded checkpoint keys:", ckpt.keys())
print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))
print("Checkpoint best_thr:", ckpt.get("best_thr", "not saved"))

model = STUNet2p5D(
    temp_feats=80,
    base=BASE_CHANNELS,
    extra_ch=CEXTRA,
).to(device)

model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

if PRED_THR is None:
    PRED_THR = float(ckpt.get("best_thr", 0.40))

print("Using PRED_THR =", PRED_THR)

# ============================================================
# BLOB EXTRACTION
# ============================================================

def extract_predicted_blobs(
    pred_hw: np.ndarray,
    x_ids: np.ndarray,
    y_ids: np.ndarray,
    event_id: int,
    connectivity: int = 2,
):
    """
    Find connected predicted blobs and return one dictionary per blob.

    Important for strip-gap model:
      - fake spacer rows/cols are allowed to help connected components
      - but the reported centroid is snapped to the nearest REAL strip pixel
    """

    pred_hw = (pred_hw > 0).astype(np.uint8)

    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, n_blobs = ndimage.label(pred_hw, structure=structure)

    H, W = pred_hw.shape

    x_ids = np.asarray(x_ids)
    y_ids = np.asarray(y_ids)

    real_x = x_ids >= 0
    real_y = y_ids >= 0
    valid_real = real_y[:, None] & real_x[None, :]

    rows = []

    for lab in range(1, n_blobs + 1):
        ys, xs = np.where(labeled == lab)

        if ys.size == 0:
            continue

        # Prefer real pixels inside the blob when computing centroid.
        real_inside = valid_real[ys, xs]

        if np.any(real_inside):
            ys_use = ys[real_inside]
            xs_use = xs[real_inside]
        else:
            # Blob exists only on fake spacer rows/cols.
            # Ignore it because it cannot map to a physical strip pair.
            continue

        cy = float(ys_use.mean())
        cx = float(xs_use.mean())

        iy0 = int(np.clip(round(cy), 0, H - 1))
        ix0 = int(np.clip(round(cx), 0, W - 1))

        # Snap to nearest real y row
        if y_ids[iy0] < 0:
            real_y_idx = np.where(real_y)[0]
            iy = int(real_y_idx[np.argmin(np.abs(real_y_idx - iy0))])
        else:
            iy = iy0

        # Snap to nearest real x column
        if x_ids[ix0] < 0:
            real_x_idx = np.where(real_x)[0]
            ix = int(real_x_idx[np.argmin(np.abs(real_x_idx - ix0))])
        else:
            ix = ix0

        rows.append({
            "event_id": int(event_id),
            "blob_id": int(lab - 1),
            "cy": cy,
            "cx": cx,
            "iy": iy,
            "ix": ix,
            "y_strip": int(y_ids[iy]),
            "x_strip": int(x_ids[ix]),
            "area": int(len(ys)),
            "real_area": int(len(ys_use)),
        })

    return rows


# ============================================================
# RUN INFERENCE
# ============================================================

@torch.no_grad()
def run_inference_and_save_blobs(
    df: pd.DataFrame,
    model,
    output_file: str,
    pred_mask_dir: str,
    pred_thr: float,
    compact_min_q: float = 0.00001,
    apply_log: bool = False,
):
    grouped = {int(ev): sub.copy() for ev, sub in df.groupby("event_id")}
    event_ids = sorted(grouped.keys())

    # os.makedirs(pred_mask_dir, exist_ok=True)
    shutil.rmtree(pred_mask_dir, ignore_errors=True)
    os.makedirs(pred_mask_dir, exist_ok=True)

    all_rows = []

    n_skipped = 0
    n_no_blob = 0

    for ev in tqdm(event_ids, desc="Running inference"):
        sub = grouped[ev]

        built = build_x3d_and_extra_inference_stripgap(
            sub,
            apply_log=apply_log,
            min_q=compact_min_q,
            gap_min=GAP_MIN,
            n_spacer=N_SPACER,
        )

        x3d, extra, x_ids, y_ids, valid_mask = built

        if x3d is None:
            n_skipped += 1
            continue

        x_t = torch.from_numpy(x3d).float().unsqueeze(0).to(device)
        extra_t = torch.from_numpy(extra).float().unsqueeze(0).to(device)

        # logits = model(x_t, extra_t)
        logits = model(
            x_t,
            extra_t,
            x_ids=x_ids,
            y_ids=y_ids,
        )
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        # Do not allow fake spacer-only pixels to become physical predictions.
        # This makes blob extraction/evaluation cleaner.
        prob_valid = prob * valid_mask.astype(np.float32)

        pred = (prob_valid > float(pred_thr)).astype(np.uint8)

        np.savez_compressed(
            os.path.join(pred_mask_dir, f"event_{int(ev)}.npz"),
            event_id=np.array(int(ev), dtype=np.int64),
            pred_mask=pred.astype(np.uint8),
            prob=prob.astype(np.float32),
            prob_valid=prob_valid.astype(np.float32),
            valid_mask=valid_mask.astype(np.uint8),
            x_ids=np.asarray(x_ids, dtype=np.int32),
            y_ids=np.asarray(y_ids, dtype=np.int32),
        )

        rows_this_event = extract_predicted_blobs(
            pred_hw=pred,
            x_ids=x_ids,
            y_ids=y_ids,
            event_id=ev,
            connectivity=2,
        )

        if len(rows_this_event) == 0:
            n_no_blob += 1
        else:
            all_rows.extend(rows_this_event)

    out_cols = [
        "event_id",
        "blob_id",
        "cy",
        "cx",
        "iy",
        "ix",
        "y_strip",
        "x_strip",
        "area",
        "real_area",
    ]

    out_df = pd.DataFrame(all_rows, columns=out_cols)

    # out_dir = os.path.dirname(output_file)
    # if out_dir:
    #     os.makedirs(out_dir, exist_ok=True)

    out_dir = os.path.dirname(output_file)

    if out_dir:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(
        output_file,
        sep=" ",
        index=False,
        float_format="%.6f"
    )

    print()
    print("finished")
    print("total events:", len(event_ids))
    print("skipped events because X or Y region could not be built:", n_skipped)
    print("events with zero predicted blobs:", n_no_blob)
    print("predicted blobs saved:", len(out_df))
    print("blob text file saved to:", output_file)
    print("per-event predicted masks saved in:", pred_mask_dir)

    return out_df


out_df = run_inference_and_save_blobs(
    df=df,
    model=model,
    output_file=OUTPUT_FILE,
    pred_mask_dir=PRED_MASK_DIR,
    pred_thr=PRED_THR,
    compact_min_q=COMPACT_MIN_Q,
    apply_log=APPLY_LOG,
)

out_df.head()


