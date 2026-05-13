# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR     = os.path.join(DATASET_DIR, "by_person")
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"

# ── Multi-view config ──────────────────────────────────────────────────────
ALL_CAMERAS   = [0, 1, 2]
NUM_VIEWS     = len(ALL_CAMERAS)
NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17

TARGET_FRAMES = 100

# ── Training config ────────────────────────────────────────────────────────
EPOCHS        = 300
LR            = 1e-4
BATCH_SIZE    = 32
WEIGHT_DECAY  = 1e-3
OUT_DIR       = "/mvdlph/masa/GCN_MultiView_LateFusion_Results"

# ── Early Stopping ─────────────────────────────────────────────────────────
PATIENCE      = 80
MIN_DELTA     = 1e-4

# ── Single-view baseline ───────────────────────────────────────────────────
SV_TEST_MAE  = 0.4076
SV_TEST_RMSE = 0.5569
SV_TEST_R2   = 0.4859
SV_TEST_PCC  = 0.7443

# ── Early-fusion reference (from previous run) ─────────────────────────────
EF_TEST_MAE  = 0.3786
EF_TEST_RMSE = 0.4972
EF_TEST_R2   = 0.5895
EF_TEST_PCC  = 0.7720

print('✓ Configuration loaded')
print(f'  DATASET_DIR  : {DATASET_DIR}')
print(f'  ALL_CAMERAS  : {ALL_CAMERAS}  (NUM_VIEWS={NUM_VIEWS})')
print(f'  NUM_JOINTS   : {NUM_JOINTS}  (per camera, NOT fused)')
print(f'  LR           : {LR}  |  WEIGHT_DECAY: {WEIGHT_DECAY}')
print(f'  PATIENCE     : {PATIENCE} epochs')
print(f'\n  Single-View Baseline  : MAE={SV_TEST_MAE}  RMSE={SV_TEST_RMSE}  R²={SV_TEST_R2}  PCC={SV_TEST_PCC}')
print(f'  Early-Fusion Baseline : MAE={EF_TEST_MAE}  RMSE={EF_TEST_RMSE}  R²={EF_TEST_R2}  PCC={EF_TEST_PCC}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 2 — Imports & output folders
# ══════════════════════════════════════════════════════════════════════════

import re, glob, json, logging, datetime, copy, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

run_name  = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
RUN_DIR   = os.path.join(OUT_DIR, run_name)
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
LOGS_DIR  = os.path.join(RUN_DIR, "logs")

for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Libraries imported")
print("✓ Run directory:", RUN_DIR)


# ══════════════════════════════════════════════════════════════════════════
# Cell 3 — Explore dataset folder
# ══════════════════════════════════════════════════════════════════════════

all_npz = sorted(glob.glob(os.path.join(DATASET_DIR, '**/*.npz'), recursive=True))
print(f'Total NPZ files found : {len(all_npz)}')

if len(all_npz) == 0:
    print('\n❌ No NPZ files found!')
else:
    print(f'First file: {os.path.basename(all_npz[0])}')
    sample = np.load(all_npz[0])
    print(f'Keys      : {list(sample.keys())}')
    for k in sample.keys():
        print(f'  {k!r:20s} → shape {sample[k].shape}  dtype {sample[k].dtype}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 4 — Load CSV labels
# ══════════════════════════════════════════════════════════════════════════

df_csv = None

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'rb') as f:
        raw = f.read()
    for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'cp1252']:
        try:
            text = raw.decode(enc)
            tmp  = pd.read_csv(io.StringIO(text))
            tmp.columns = tmp.columns.str.strip()
            if 'exercise' in tmp.columns:
                df_csv = tmp
                print(f'✓ CSV loaded with encoding: {enc}')
                break
        except Exception as e:
            print(f'  {enc}: {e}')
else:
    print(f'⚠️  CSV not found at: {CSV_PATH}')

if df_csv is None:
    raise FileNotFoundError(f'\n❌ CSV not loaded from: {CSV_PATH}')

print(f'Columns : {df_csv.columns.tolist()}')
print(f'Shape   : {df_csv.shape}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 5 — Logging setup
# ══════════════════════════════════════════════════════════════════════════

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(LOGS_DIR, f"training_{timestamp}.log")


class Tee:
    def __init__(self, console, filepath):
        self.console  = console
        self._logfile = open(filepath, 'a', encoding='utf-8', buffering=1)

    def write(self, msg):
        self.console.write(msg)
        self._logfile.write(msg)

    def flush(self):
        self.console.flush()
        self._logfile.flush()

    def restore(self):
        sys.stdout = self.console
        self._logfile.close()


if not isinstance(sys.stdout, Tee):
    sys.stdout = Tee(sys.stdout, log_file)
print(f'✓ stdout → also writing to {log_file}')

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("GCN-MultiView-LateFusion")
log.info("=" * 70)
log.info(f"ST-GCN Multi-View LATE Fusion | Cameras={ALL_CAMERAS}")
log.info(f"Epochs={EPOCHS} | Patience={PATIENCE} | Batch={BATCH_SIZE} | LR={LR}")
log.info(f"Single-View Baseline : MAE={SV_TEST_MAE} RMSE={SV_TEST_RMSE} R²={SV_TEST_R2} PCC={SV_TEST_PCC}")
log.info(f"Early-Fusion Baseline: MAE={EF_TEST_MAE} RMSE={EF_TEST_RMSE} R²={EF_TEST_R2} PCC={EF_TEST_PCC}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader
# ══════════════════════════════════════════════════════════════════════════

SKELETON_EDGES = [
    (0, 1), (1, 2),  (2, 3),
    (0, 4), (4, 5),  (5, 6),
    (0, 7), (7, 8),  (8, 9),
    (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

NUM_EXERCISES = 10


def parse_filename(fpath):
    base = os.path.basename(fpath)
    m    = re.match(r"E(\d+)_P(\d+)_T(\d+)_C(\d+)_seg(\d+)", base)
    if m is None:
        return None
    return {
        "exercise" : int(m.group(1)),
        "person"   : f"P{m.group(2)}",
        "trial_num": int(m.group(3)),
        "trial_id" : f"T{m.group(3)}",
        "camera"   : int(m.group(4)),
        "segment"  : int(m.group(5)),
        "filepath" : fpath,
    }


def load_skeleton(fpath, key=NPZ_KEY):
    """Load NPZ skeleton → (T, 17, 3) float32 or None on failure."""
    try:
        data = np.load(fpath, allow_pickle=True)
        arr  = data[key] if key in data else data[list(data.keys())[0]]
        arr  = arr.astype(np.float32)
        if arr.ndim == 1:  return None
        if arr.ndim == 2:  arr = arr.reshape(arr.shape[0], 17, 3)
        elif arr.ndim == 4: arr = arr.squeeze(0)
        if arr.shape[1] != 17 or arr.shape[2] != 3:
            return None
        return arr
    except Exception:
        return None


print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build multi-view dataset index
# ══════════════════════════════════════════════════════════════════════════

def build_multiview_index(split_name, df_csv, cameras=ALL_CAMERAS):
    split_path = os.path.join(SPLIT_DIR, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    all_files = sorted(glob.glob(
        os.path.join(split_path, '**/*.npz'), recursive=True))
    print(f'\n[{split_name.upper()}] NPZ files found : {len(all_files)}')

    lookup = {}
    for fpath in all_files:
        meta = parse_filename(fpath)
        if meta is None:
            continue
        if meta['camera'] not in cameras:
            continue
        key = (meta['exercise'], meta['person'], meta['trial_id'], meta['segment'])
        lookup.setdefault(key, {})
        lookup[key][meta['camera']] = fpath

    df_csv_clean         = df_csv.copy()
    df_csv_clean.columns = df_csv_clean.columns.str.strip()

    records = []
    for (exercise, person, trial_id, segment), cam_paths in lookup.items():
        trial_num = int(trial_id[1:])

        row = df_csv_clean[
            (df_csv_clean['exercise'] == f"E{exercise}") &
            (df_csv_clean['person']   == person)          &
            (df_csv_clean['trial']    == trial_id)
        ]
        quality   = float(row.iloc[0]['mean']) if len(row) > 0 else np.nan
        trial_key = f"E{exercise}_{person}_{trial_id}"

        record = {
            'exercise' : exercise,
            'person'   : person,
            'trial_num': trial_num,
            'trial_id' : trial_id,
            'segment'  : segment,
            'quality'  : quality,
            'trial_key': trial_key,
            'split'    : split_name,
        }
        n_available = 0
        for cam in cameras:
            record[f'filepath_C{cam}'] = cam_paths.get(cam, None)
            if cam_paths.get(cam) is not None:
                n_available += 1
        record['n_views_available'] = n_available
        records.append(record)

    if not records:
        print(f'  ⚠️  No records found for split="{split_name}"')
        return pd.DataFrame()

    df = pd.DataFrame(records)

    correct_mean   = df.loc[df['trial_num'] <= 2, 'quality'].mean()
    erroneous_mean = df.loc[df['trial_num'] >= 3, 'quality'].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5
    df.loc[df['quality'].isna() & (df['trial_num'] <= 2), 'quality'] = correct_mean
    df.loc[df['quality'].isna() & (df['trial_num'] >= 3), 'quality'] = erroneous_mean

    print(f'  Unique segments      : {len(df)}')
    print(f'  Unique trial keys    : {df["trial_key"].nunique()}')
    print(f'  Quality mean±std     : {df["quality"].mean():.3f} ± {df["quality"].std():.3f}')
    print(f'  Views available dist :')
    print(df['n_views_available'].value_counts().sort_index().to_string(header=False))

    return df


def remove_all_corrupted(df, cameras=ALL_CAMERAS, label=''):
    drop_idx = []
    for i, row in df.iterrows():
        valid = False
        for cam in cameras:
            fp = row.get(f'filepath_C{cam}')
            if fp is not None and load_skeleton(fp) is not None:
                valid = True
                break
            elif fp is not None:
                df.at[i, f'filepath_C{cam}'] = None
        if not valid:
            drop_idx.append(i)
    if drop_idx:
        print(f'  [{label}] Dropping {len(drop_idx)} fully-corrupted rows')
        df = df.drop(index=drop_idx).reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


train_df = build_multiview_index('train', df_csv)
val_df   = build_multiview_index('valid', df_csv)
test_df  = build_multiview_index('test',  df_csv)

print('\nChecking for corrupted files...')
train_df = remove_all_corrupted(train_df, label='TRAIN')
val_df   = remove_all_corrupted(val_df,   label='VALID')
test_df  = remove_all_corrupted(test_df,  label='TEST')

tr_keys = set(train_df['trial_key'])
vl_keys = set(val_df['trial_key'])
te_keys = set(test_df['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

print(f'\n{"═"*72}')
print(f'  {"Split":<8} {"Segs":>6} {"Correct":>9} {"Erroneous":>11} '
      f'{"Q mean":>8} {"Q std":>7} {"Q min":>7} {"Q max":>7}')
print(f'  {"─"*70}')
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    cor = (d['trial_num'] <= 2).sum()
    err = (d['trial_num'] >= 3).sum()
    q   = d['quality']
    print(f'  {name:<8} {len(d):>6} {cor:>9} {err:>11} '
          f'{q.mean():>8.3f} {q.std():>7.3f} '
          f'{q.min():>7.3f} {q.max():>7.3f}')
print(f'{"═"*72}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Preprocessing helpers  (same as early fusion)
# ══════════════════════════════════════════════════════════════════════════

def _normalise_length_np(skel, target_frames):
    T = skel.shape[0]
    if T == target_frames:
        return skel
    old_idx  = np.linspace(0, 1, T)
    new_idx  = np.linspace(0, 1, target_frames)
    flat     = skel.reshape(T, -1)
    out_flat = np.zeros((target_frames, flat.shape[1]), dtype=np.float32)
    for k in range(flat.shape[1]):
        out_flat[:, k] = np.interp(new_idx, old_idx, flat[:, k])
    return out_flat.reshape(target_frames, skel.shape[1], skel.shape[2])


def _centre_scale_np(skel):
    hip      = (skel[:, 1:2, :] + skel[:, 4:5, :]) / 2.0
    skel     = skel - hip
    shoulder = (skel[:, 11:12, :] + skel[:, 14:15, :]) / 2.0
    torso_h  = np.abs(shoulder[:, :, 1:2]).mean(axis=0, keepdims=True).clip(min=1e-6)
    return skel / torso_h


def _add_velocity_np(skel):
    vel     = np.zeros_like(skel)
    vel[1:] = skel[1:] - skel[:-1]
    return np.concatenate([skel, vel], axis=-1)


print('✓ Preprocessing helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — BZULateFusionDataset
#
#  KEY DIFFERENCE from early fusion:
#    Returns a list of per-camera tensors: List[(T, J, 6)] × NUM_VIEWS
#    Each camera is processed independently — NOT concatenated here.
#    Missing cameras → zero tensor (T, J, 6), flagged in view_mask.
#
#  The model receives separate per-camera inputs and fuses AFTER encoding.
# ══════════════════════════════════════════════════════════════════════════

class BZULateFusionDataset(Dataset):
    def __init__(self, df, cameras=ALL_CAMERAS,
                 target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.cameras       = cameras
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        cam_skels  = []   # list of (T, J, 6) per camera
        view_mask  = []   # True = camera available

        for cam in self.cameras:
            fp   = row.get(f'filepath_C{cam}')
            skel = load_skeleton(fp) if fp is not None else None

            if skel is None:
                cam_skels.append(
                    torch.zeros(self.target_frames, NUM_JOINTS, 6,
                                dtype=torch.float32))
                view_mask.append(False)
                continue

            skel = _normalise_length_np(skel, self.target_frames)
            if self.augment:
                skel = self._augment(skel)
            skel = _centre_scale_np(skel)
            skel = _add_velocity_np(skel)                       # (T, J, 6)
            cam_skels.append(torch.tensor(skel, dtype=torch.float32))
            view_mask.append(True)

        # Stack into (V, T, J, 6) — model receives all views separately
        cam_skels  = torch.stack(cam_skels, dim=0)              # (V, T, J, 6)
        quality    = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)
        mask_tensor = torch.tensor(view_mask,       dtype=torch.bool)  # (V,)

        return cam_skels, quality, exercise_id, mask_tensor

    def _augment(self, skel):
        T     = skel.shape[0]
        speed = np.random.uniform(0.85, 1.15)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = _normalise_length_np(skel[idxs], self.target_frames)
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.003
        if np.random.rand() < 0.5:
            skel[:, :, 0] *= -1.0
        return skel


print('✓ BZULateFusionDataset defined (per-camera tensors, shape (V, T, J, 6))')

_ds = BZULateFusionDataset(train_df)
_s, _q, _e, _m = _ds[0]
print(f'  cam_skels shape : {_s.shape}  (expected V={NUM_VIEWS}, T={TARGET_FRAMES}, J={NUM_JOINTS}, C=6)')
assert _s.shape == (NUM_VIEWS, TARGET_FRAMES, NUM_JOINTS, 6), f"Shape mismatch! Got {_s.shape}"
del _ds, _s, _q, _e, _m
print('✓ Dataset shape check passed')


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — Shared GCN building blocks  (same as early fusion, reused)
# ══════════════════════════════════════════════════════════════════════════

def build_single_view_adj(num_joints, intra_edges):
    """Standard Kipf-normalised adjacency for a single-camera skeleton."""
    N = num_joints
    A = np.zeros((N, N), dtype=np.float32)
    for (i, j) in intra_edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A_tilde  = A + np.eye(N, dtype=np.float32)
    deg      = A_tilde.sum(axis=1)
    d_inv_sq = np.diag(np.power(deg, -0.5))
    A_hat    = d_inv_sq @ A_tilde @ d_inv_sq
    return torch.tensor(A_hat, dtype=torch.float32)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)
        if self.W.bias is not None:
            nn.init.zeros_(self.W.bias)

    def forward(self, H, A_hat):
        """H: (N, J, Cin) → (N, J, Cout)"""
        support = self.W(H)
        A_exp   = A_hat.unsqueeze(0).expand(H.size(0), -1, -1)
        return torch.bmm(A_exp, support)


class GCNBackbone(nn.Module):
    def __init__(self, in_features, hidden_dims, dropout=0.5):
        super().__init__()
        dims   = [in_features] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(GraphConvolution(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        self.layers  = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.n_gcn   = len(hidden_dims)

    def forward(self, x, A_hat):
        """x: (B, T, J, C) → (B, T, J, C_out)"""
        B, T, J, C = x.shape
        h = x.reshape(B * T, J, C)
        gcn_idx = 0
        for i in range(0, len(self.layers), 2):
            h = self.layers[i](h, A_hat)
            N, J2, Co = h.shape
            h = self.layers[i+1](h.reshape(N * J2, Co)).reshape(N, J2, Co)
            gcn_idx += 1
            if gcn_idx < self.n_gcn:
                h = F.relu(h)
                h = self.dropout(h)
        return h.reshape(B, T, J, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size    = feat_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.out_dim = hidden_dim * 2
        self.drop    = nn.Dropout(dropout)

    def forward(self, h):
        """h: (B, T, J, C) → (B, 2H)"""
        h = h.max(dim=2)[0]                      # (B, T, C) — max over joints
        _, hidden = self.gru(h)                   # (2*L, B, H)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, 2H)
        return self.drop(h)


print('✓ GCN building blocks defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — SingleViewEncoder
#
#  One per camera: data_bn → GCN → TemporalEncoder → embedding (B, D_enc)
#  All cameras share the SAME weights (weight sharing).
#  Optionally use separate weights per camera (set shared_weights=False).
# ══════════════════════════════════════════════════════════════════════════

class SingleViewEncoder(nn.Module):
    """
    Encodes one camera's skeleton (B, T, J, 6) → embedding (B, D_enc).
    Registers A_hat as a buffer so it moves to GPU automatically.
    """
    def __init__(self, in_features=6, hidden_dims=None, dropout=0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        A_hat = build_single_view_adj(NUM_JOINTS, SKELETON_EDGES)
        self.register_buffer('A_hat', A_hat)

        self.data_bn  = nn.BatchNorm1d(in_features)
        self.gcn      = GCNBackbone(in_features, hidden_dims, dropout=dropout)
        self.temporal = TemporalEncoder(
            feat_dim   = hidden_dims[-1],
            hidden_dim = 128,
            num_layers = 2,
            dropout    = 0.4,
        )
        self.out_dim = self.temporal.out_dim   # 256

    def forward(self, x):
        """x: (B, T, J, 6) → (B, 256)"""
        B, T, J, C = x.shape
        xbn = x.permute(0, 3, 1, 2).reshape(B, C, T * J)
        xbn = self.data_bn(xbn)
        x   = xbn.reshape(B, C, T, J).permute(0, 2, 3, 1)  # (B,T,J,C)
        h   = self.gcn(x, self.A_hat)                        # (B,T,J,256)
        h   = self.temporal(h)                               # (B,256)
        return h


print('✓ SingleViewEncoder defined  (out_dim=256 per camera)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — FusionModule
#
#  Four fusion strategies available via FUSION_TYPE:
#
#  "attention"   — Learnable attention over camera embeddings (recommended).
#                  Missing cameras are masked out of the softmax.
#                  This is the primary late-fusion strategy.
#
#  "mean"        — Simple mean of available camera embeddings (strong baseline).
#
#  "max"         — Element-wise max over camera embeddings.
#
#  "concat_proj" — Concatenate all V embeddings → linear projection.
#                  Missing cameras replaced with learnable missing-token.
#                  Fast, but fixed input size (requires all V slots filled).
# ══════════════════════════════════════════════════════════════════════════

FUSION_TYPE = "attention"   # ← change to "mean" / "max" / "concat_proj" to compare


class AttentionFusion(nn.Module):
    """
    Soft attention over V camera embeddings.
    Missing cameras are masked to -inf before softmax → weight = 0.
    """
    def __init__(self, embed_dim, num_views):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.num_views = num_views

    def forward(self, embeds, view_mask):
        """
        embeds    : (B, V, D)
        view_mask : (B, V) bool  — True = valid camera
        Returns   : (B, D)
        """
        scores = self.score(embeds).squeeze(-1)        # (B, V)
        # Mask missing cameras: set to -inf so softmax gives 0 weight
        mask_float = view_mask.float()
        scores     = scores * mask_float + (1.0 - mask_float) * (-1e9)
        weights    = torch.softmax(scores, dim=1)      # (B, V)
        fused      = (weights.unsqueeze(-1) * embeds).sum(dim=1)  # (B, D)
        return fused, weights


class MeanFusion(nn.Module):
    """Mean of available camera embeddings (ignores missing)."""
    def forward(self, embeds, view_mask):
        """
        embeds    : (B, V, D)
        view_mask : (B, V) bool
        Returns   : (B, D)
        """
        mask   = view_mask.float().unsqueeze(-1)         # (B, V, 1)
        n_valid = mask.sum(dim=1).clamp(min=1)           # (B, 1)
        fused  = (embeds * mask).sum(dim=1) / n_valid    # (B, D)
        return fused, mask.squeeze(-1)


class MaxFusion(nn.Module):
    """Element-wise max over available camera embeddings."""
    def forward(self, embeds, view_mask):
        """
        embeds    : (B, V, D)
        view_mask : (B, V) bool
        """
        mask  = view_mask.float().unsqueeze(-1)           # (B, V, 1)
        # Set missing cameras to -inf before max
        masked = embeds * mask + (1.0 - mask) * (-1e9)
        fused  = masked.max(dim=1)[0]                     # (B, D)
        return fused, mask.squeeze(-1)


class ConcatProjFusion(nn.Module):
    """
    Concatenate all V embeddings → linear projection.
    Missing cameras filled with a learnable missing-token embedding.
    """
    def __init__(self, embed_dim, num_views, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = embed_dim
        self.missing_token = nn.Parameter(torch.zeros(embed_dim))
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * num_views, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.num_views = num_views
        self.out_dim   = out_dim

    def forward(self, embeds, view_mask):
        """
        embeds    : (B, V, D)
        view_mask : (B, V) bool
        """
        B, V, D = embeds.shape
        token   = self.missing_token.view(1, 1, D).expand(B, V, D)
        mask    = view_mask.float().unsqueeze(-1)         # (B, V, 1)
        filled  = mask * embeds + (1.0 - mask) * token   # (B, V, D)
        fused   = self.proj(filled.reshape(B, V * D))     # (B, out_dim)
        return fused, mask.squeeze(-1)


def build_fusion_module(fusion_type, embed_dim, num_views):
    if fusion_type == "attention":
        return AttentionFusion(embed_dim, num_views)
    elif fusion_type == "mean":
        return MeanFusion()
    elif fusion_type == "max":
        return MaxFusion()
    elif fusion_type == "concat_proj":
        return ConcatProjFusion(embed_dim, num_views, out_dim=embed_dim)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")


print(f'✓ FusionModule defined  (using FUSION_TYPE="{FUSION_TYPE}")')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — GCN_LateFusion_Regression  (main model)
#
#  Architecture:
#    For each camera v:
#        x_v (B, T, J, 6) → SingleViewEncoder → e_v (B, 256)
#    Stack: [e_0, e_1, e_2] → (B, V, 256)
#    FusionModule(embeds, view_mask) → fused (B, 256)
#    Per-exercise regression head → scalar quality score
#
#  Weight sharing: all cameras share one SingleViewEncoder by default.
#  Set shared_weights=False for independent per-camera encoders.
# ══════════════════════════════════════════════════════════════════════════

class GCN_LateFusion_Regression(nn.Module):
    def __init__(self,
                 in_features   = 6,
                 hidden_dims   = None,
                 dropout       = 0.5,
                 fusion_type   = FUSION_TYPE,
                 shared_weights = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.num_views      = NUM_VIEWS
        self.shared_weights = shared_weights

        # ── Per-camera encoders ──────────────────────────────────────────
        if shared_weights:
            # One encoder, called once per camera (tied weights)
            encoder = SingleViewEncoder(in_features, hidden_dims, dropout)
            self.encoders = nn.ModuleList([encoder] * NUM_VIEWS)
            # Note: ModuleList with repeated modules only registers one set
            # of parameters — exactly what we want for weight sharing
            self._shared_encoder = encoder   # keep a reference
        else:
            # Independent encoder per camera (3× parameters)
            self.encoders = nn.ModuleList([
                SingleViewEncoder(in_features, hidden_dims, dropout)
                for _ in range(NUM_VIEWS)
            ])

        embed_dim = self.encoders[0].out_dim   # 256

        # ── Fusion ──────────────────────────────────────────────────────
        self.fusion = build_fusion_module(fusion_type, embed_dim, NUM_VIEWS)

        # Fused dim may differ for concat_proj (still embed_dim here)
        fused_dim = embed_dim

        # ── Per-exercise regression heads ────────────────────────────────
        self.heads = nn.ModuleDict({
            f'E{i}': nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            for i in range(NUM_EXERCISES)
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(param)
                    elif 'bias'  in name: nn.init.zeros_(param)

    def forward(self, cam_skels, exercise_id, view_mask):
        """
        cam_skels   : (B, V, T, J, 6)
        exercise_id : (B,) long
        view_mask   : (B, V) bool

        Returns     : (B,) quality scores in (0.6, 5.4)
        """
        B, V, T, J, C = cam_skels.shape

        # ── Encode each camera independently ────────────────────────────
        cam_embeds = []
        for v in range(self.num_views):
            x_v = cam_skels[:, v, :, :, :]          # (B, T, J, 6)
            e_v = self.encoders[v](x_v)              # (B, 256)
            cam_embeds.append(e_v)

        embeds = torch.stack(cam_embeds, dim=1)      # (B, V, 256)

        # ── Fuse embeddings ──────────────────────────────────────────────
        fused, _ = self.fusion(embeds, view_mask)    # (B, 256)

        # ── Per-exercise head ────────────────────────────────────────────
        out = torch.cat([
            self.heads[f'E{exercise_id[b].item()}'](fused[b].unsqueeze(0))
            for b in range(B)
        ], dim=0).squeeze(-1)                        # (B,)

        # Scale to (0.6, 5.4) — same as early fusion
        out = 3.0 + 2.4 * torch.tanh(out)
        return out


# ── Sanity check ──────────────────────────────────────────────────────────
_dummy_cams = torch.zeros(2, NUM_VIEWS, TARGET_FRAMES, NUM_JOINTS, 6)
_dummy_ex   = torch.zeros(2, dtype=torch.long)
_dummy_mask = torch.ones(2, NUM_VIEWS, dtype=torch.bool)
_model      = GCN_LateFusion_Regression(shared_weights=True)
_out        = _model(_dummy_cams, _dummy_ex, _dummy_mask)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
print(f'✓ GCN_LateFusion_Regression sanity check passed — output: {_out.shape}')
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters (shared weights): {total_params:,}')

_model_sep = GCN_LateFusion_Regression(shared_weights=False)
sep_params  = sum(p.numel() for p in _model_sep.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters (separate weights): {sep_params:,}')
del _dummy_cams, _dummy_ex, _model, _out, _model_sep


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Device & run_epoch
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def run_epoch(model, loader, reg_fn, optimiser=None, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for cam_skels, qualities, exercise_ids, view_mask in loader:
            cam_skels    = cam_skels.to(DEVICE)       # (B, V, T, J, 6)
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)
            view_mask    = view_mask.to(DEVICE)        # (B, V)

            preds = model(cam_skels, exercise_ids, view_mask)
            loss  = reg_fn(preds, qualities)

            if is_train and optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            total_loss += loss.item()
            q_true.extend(qualities.cpu().numpy())
            q_pred.extend(preds.detach().cpu().numpy())

    n  = max(1, len(loader))
    qt = np.array(q_true)
    qp = np.array(q_pred)
    pcc = float(pearsonr(qt, qp)[0]) if len(qt) > 1 else 0.0

    return {
        'loss': total_loss / n,
        'rmse': float(np.sqrt(np.mean((qt - qp) ** 2))),
        'mae' : float(np.mean(np.abs(qt - qp))),
        'r2'  : float(r2_score(qt, qp)) if len(qt) > 1 else 0.0,
        'pcc' : pcc,
    }


print('✓ run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Plotting helpers
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def _add_test_line(ax, val, label, color='green'):
    ax.axhline(val, color=color, linestyle='-.', linewidth=1.5,
               label=f'Test {label}={val:.4f}')


def _add_ref_line(ax, val, label, color, linestyle=':'):
    ax.axhline(val, color=color, linestyle=linestyle, linewidth=1.4,
               label=label)


def plot_loss_curves(history, save_dir, test_loss=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    if test_loss: _add_test_line(ax, test_loss, 'Loss')
    ax.set_title('Regression Loss (Huber) — Multi-View Late Fusion',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curve.png'))


def plot_rmse_mae(history, save_dir, test_rmse=None, test_mae=None):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('RMSE & MAE — Late Fusion vs EF vs SV Baselines',
                 fontsize=13, fontweight='bold')
    for ax, metric, title, test_val, ef_val, sv_val in [
        (axes[0], 'rmse', 'RMSE', test_rmse, EF_TEST_RMSE, SV_TEST_RMSE),
        (axes[1], 'mae',  'MAE',  test_mae,  EF_TEST_MAE,  SV_TEST_MAE),
    ]:
        ax.plot(epochs, history[f'train_{metric}'], label='Train',      color='steelblue')
        ax.plot(epochs, history[f'val_{metric}'],   label='Validation', color='darkorange')
        if test_val: _add_test_line(ax, test_val, title)
        _add_ref_line(ax, ef_val, f'EF {title}={ef_val:.4f}', 'purple')
        _add_ref_line(ax, sv_val, f'SV {title}={sv_val:.4f}', 'gray')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'rmse_mae.png'))


def plot_r2_pcc(history, save_dir, test_r2=None, test_pcc=None):
    epochs = range(1, len(history['val_r2']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('R² & PCC — Late Fusion vs EF vs SV Baselines',
                 fontsize=13, fontweight='bold')
    for ax, metric, title, test_val, ef_val, sv_val in [
        (axes[0], 'r2',  'R²',  test_r2,  EF_TEST_R2,  SV_TEST_R2),
        (axes[1], 'pcc', 'PCC', test_pcc, EF_TEST_PCC, SV_TEST_PCC),
    ]:
        ax.plot(epochs, history[f'train_{metric}'], label='Train',      color='steelblue')
        ax.plot(epochs, history[f'val_{metric}'],   label='Validation', color='darkorange')
        if test_val: _add_test_line(ax, test_val, title)
        _add_ref_line(ax, ef_val, f'EF {title}={ef_val:.4f}', 'purple')
        _add_ref_line(ax, sv_val, f'SV {title}={sv_val:.4f}', 'gray')
        ax.axhline(1.0, color='silver', linestyle=':', linewidth=1)
        ax.axhline(0.0, color='red',    linestyle=':', linewidth=1)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'r2_pcc.png'))


def plot_regression_scatter(q_true, q_pred, split_name='Test', save_dir=None):
    qt   = np.array(q_true)
    qp   = np.array(q_pred)
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp) ** 2)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(qt, qp, alpha=0.6, edgecolors='black', linewidths=0.4,
               color='mediumseagreen', s=60)
    lo = min(qt.min(), qp.min()) - 0.2
    hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True Quality Score',      fontsize=12)
    ax.set_ylabel('Predicted Quality Score', fontsize=12)
    ax.set_title(f'{split_name} — True vs Predicted (Multi-View Late Fusion)',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.97,
            f'Late Fusion\nR²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}\n'
            f'─────────────\nEarly Fusion\nR²   = {EF_TEST_R2:.4f}\nMAE  = {EF_TEST_MAE:.4f}\nRMSE = {EF_TEST_RMSE:.4f}\n'
            f'─────────────\nSingle-View\nR²   = {SV_TEST_R2:.4f}\nMAE  = {SV_TEST_MAE:.4f}\nRMSE = {SV_TEST_RMSE:.4f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    if save_dir:
        save_and_show(fig, os.path.join(save_dir, f'regression_scatter_{split_name.lower()}.png'))
    else:
        plt.close(fig)


def plot_early_stop(history, stopped_epoch, best_epoch, save_dir):
    epochs = range(1, len(history['val_mae']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_mae'], label='Train MAE', color='steelblue')
    ax.plot(epochs, history['val_mae'],   label='Val MAE',   color='darkorange')
    ax.axvline(best_epoch,    color='purple', linestyle=':',  linewidth=2,
               label=f'Best epoch ({best_epoch})')
    ax.axvline(stopped_epoch, color='red',    linestyle='--', linewidth=2,
               label=f'Early stop ({stopped_epoch})')
    _add_ref_line(ax, EF_TEST_MAE, f'EF MAE={EF_TEST_MAE:.4f}', 'purple')
    _add_ref_line(ax, SV_TEST_MAE, f'SV MAE={SV_TEST_MAE:.4f}', 'gray')
    ax.set_title('MAE + Early Stopping — Multi-View Late Fusion',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))


def plot_comparison_bar(final_te, save_dir):
    metrics  = ['MAE', 'RMSE', 'R²', 'PCC']
    sv_vals  = [SV_TEST_MAE,  SV_TEST_RMSE,  SV_TEST_R2,  SV_TEST_PCC]
    ef_vals  = [EF_TEST_MAE,  EF_TEST_RMSE,  EF_TEST_R2,  EF_TEST_PCC]
    lf_vals  = [final_te['mae'], final_te['rmse'], final_te['r2'], final_te['pcc']]

    x  = np.arange(len(metrics))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w,   sv_vals, w, label='Single-View',       color='steelblue',    edgecolor='black', alpha=0.85)
    b2 = ax.bar(x,       ef_vals, w, label='Early Fusion',       color='darkorange',   edgecolor='black', alpha=0.85)
    b3 = ax.bar(x + w,   lf_vals, w, label='Late Fusion (ours)', color='mediumseagreen', edgecolor='black', alpha=0.85)

    for bars, vals in [(b1, sv_vals), (b2, ef_vals), (b3, lf_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_title('Single-View vs Early Fusion vs Late Fusion — Test Set',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'comparison_sv_ef_lf.png'))


def plot_attention_weights(model, loader, save_dir, n_batches=5):
    """
    Visualise learned attention weights per camera (only for attention fusion).
    Shows per-exercise mean attention weights.
    """
    if not isinstance(model.fusion, AttentionFusion):
        print('  ℹ️  Attention weight plot skipped (fusion_type != attention)')
        return

    model.eval()
    all_weights   = []   # (N, V)
    all_exercises = []

    with torch.no_grad():
        for i, (cam_skels, qualities, exercise_ids, view_mask) in enumerate(loader):
            if i >= n_batches:
                break
            cam_skels    = cam_skels.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)
            view_mask    = view_mask.to(DEVICE)

            B, V, T, J, C = cam_skels.shape
            cam_embeds = []
            for v in range(NUM_VIEWS):
                e_v = model.encoders[v](cam_skels[:, v])
                cam_embeds.append(e_v)
            embeds  = torch.stack(cam_embeds, dim=1)

            scores  = model.fusion.score(embeds).squeeze(-1)
            mf      = view_mask.float()
            scores  = scores * mf + (1.0 - mf) * (-1e9)
            weights = torch.softmax(scores, dim=1).cpu().numpy()   # (B, V)

            all_weights.extend(weights)
            all_exercises.extend(exercise_ids.cpu().numpy())

    all_weights   = np.array(all_weights)    # (N, V)
    all_exercises = np.array(all_exercises)

    unique_ex = sorted(np.unique(all_exercises))
    mean_w    = np.array([
        all_weights[all_exercises == ex].mean(axis=0)
        for ex in unique_ex
    ])   # (E, V)

    fig, ax = plt.subplots(figsize=(10, 4))
    x       = np.arange(len(unique_ex))
    w_bar   = 0.25
    colors  = ['steelblue', 'darkorange', 'mediumseagreen']
    for v in range(NUM_VIEWS):
        ax.bar(x + (v - 1) * w_bar, mean_w[:, v], w_bar,
               label=f'Camera {v}', color=colors[v], edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'E{e}' for e in unique_ex])
    ax.set_title('Mean Attention Weights per Camera per Exercise (Val Set)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Exercise'); ax.set_ylabel('Mean Attention Weight')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'attention_weights_per_exercise.png'))


print('✓ Plotting helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Early Stopping
# ══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_mae   = float('inf')
        self.counter    = 0
        self.best_wts   = None
        self.best_epoch = 1

    def step(self, val_mae, model, epoch):
        if val_mae < self.best_mae - self.min_delta:
            self.best_mae   = val_mae
            self.counter    = 0
            self.best_wts   = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False, True
        else:
            self.counter += 1
            return self.counter >= self.patience, False


print('✓ EarlyStopping defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Training Loop
# ══════════════════════════════════════════════════════════════════════════

reg_fn      = nn.SmoothL1Loss(beta=1.0)
num_workers = min(4, os.cpu_count() or 1)
print(f'DataLoader num_workers = {num_workers}')

# Set shared_weights=True  → one encoder for all cameras (fewer params, faster)
# Set shared_weights=False → independent encoder per camera (may capture
#                            camera-specific biases at the cost of 3× params)
SHARED_WEIGHTS = True

train_loader = DataLoader(
    BZULateFusionDataset(train_df, augment=True),
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = num_workers,
    pin_memory  = (DEVICE == 'cuda'),
    persistent_workers = (num_workers > 0),
)
val_loader = DataLoader(
    BZULateFusionDataset(val_df, augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = num_workers,
    pin_memory  = (DEVICE == 'cuda'),
    persistent_workers = (num_workers > 0),
)
test_loader = DataLoader(
    BZULateFusionDataset(test_df, augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = num_workers,
    pin_memory  = (DEVICE == 'cuda'),
    persistent_workers = (num_workers > 0),
)

model = GCN_LateFusion_Regression(
    in_features    = 6,
    hidden_dims    = [64, 128, 256],
    dropout        = 0.5,
    fusion_type    = FUSION_TYPE,
    shared_weights = SHARED_WEIGHTS,
).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=20, min_lr=1e-6, verbose=True)

early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val']
METRICS = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info('=' * 70)
log.info('STARTING MULTI-VIEW LATE FUSION TRAINING')
log.info(f'Fusion={FUSION_TYPE}  SharedWeights={SHARED_WEIGHTS}  Params={total_params:,}')
log.info(f'Loss=SmoothL1(Huber)  LR={LR}  WD={WEIGHT_DECAY}  Batch={BATCH_SIZE}')
log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')
log.info('=' * 70)

print(f'\n{"═"*72}')
print(f'  Multi-View LATE Fusion  |  Cameras: {ALL_CAMERAS}  |  Fusion: {FUSION_TYPE}')
print(f'  SharedWeights={SHARED_WEIGHTS}  |  Params: {total_params:,}')
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Loss : SmoothL1 (Huber)  |  LR: {LR}  |  WD: {WEIGHT_DECAY}')
print(f'  Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}  |  Workers: {num_workers}')
print(f'  Single-View Baseline  : MAE={SV_TEST_MAE} RMSE={SV_TEST_RMSE} R²={SV_TEST_R2}')
print(f'  Early-Fusion Baseline : MAE={EF_TEST_MAE} RMSE={EF_TEST_RMSE} R²={EF_TEST_R2}')
print(f'{"═"*72}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, reg_fn, optimiser=optimiser, is_train=True)
    vl = run_epoch(model, val_loader,   reg_fn, optimiser=None,      is_train=False)
    scheduler.step(vl['mae'])

    for split, res in [('train', tr), ('val', vl)]:
        for m in METRICS:
            history[f'{split}_{m}'].append(res[m])

    stop, improved = early_stop.step(vl['mae'], model, epoch)

    star = ' ★' if improved else ''
    msg = (f'  Ep {epoch:3d}/{EPOCHS} | '
           f'Tr loss={tr["loss"]:.4f} mae={tr["mae"]:.3f} '
           f'r2={tr["r2"]:.3f} pcc={tr["pcc"]:.3f} | '
           f'Vl loss={vl["loss"]:.4f} mae={vl["mae"]:.3f} '
           f'r2={vl["r2"]:.3f} pcc={vl["pcc"]:.3f} | '
           f'ES {early_stop.counter}/{PATIENCE}{star}')
    print(msg)
    log.info(msg)

    if improved:
        print(f'    ★ val_mae={early_stop.best_mae:.4f}  '
              f'rmse={vl["rmse"]:.4f}  r2={vl["r2"]:.4f}  pcc={vl["pcc"]:.4f}')

    if stop:
        stopped_epoch = epoch
        print(f'\n  ⏹  Early stopping at epoch {epoch} '
              f'(best={early_stop.best_epoch})')
        log.info(f'Early stopping at epoch {epoch} best={early_stop.best_epoch}')
        break

print('\n✓ Training complete!')

# ── Restore best weights ──────────────────────────────────────────────────
model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch

# ── Final test evaluation ─────────────────────────────────────────────────
final_te = run_epoch(model, test_loader, reg_fn, optimiser=None, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  {"Metric":<8}  {"Late Fusion":>12}  {"Early Fusion":>13}  {"Single-View":>12}  {"ΔvsEF":>8}')
print(f'  {"─"*60}')
for metric, lf_val, ef_val, sv_val in [
    ('Loss',  final_te["loss"], None,         None),
    ('RMSE',  final_te["rmse"], EF_TEST_RMSE, SV_TEST_RMSE),
    ('MAE',   final_te["mae"],  EF_TEST_MAE,  SV_TEST_MAE),
    ('R²',    final_te["r2"],   EF_TEST_R2,   SV_TEST_R2),
    ('PCC',   final_te["pcc"],  EF_TEST_PCC,  SV_TEST_PCC),
]:
    if ef_val is not None:
        delta = lf_val - ef_val
        arrow = '↑' if (metric in ['R²', 'PCC'] and delta > 0) or \
                       (metric in ['RMSE', 'MAE'] and delta < 0) else '↓'
        print(f'  {metric:<8}  {lf_val:>12.4f}  {ef_val:>13.4f}  {sv_val:>12.4f}  {delta:>+7.4f} {arrow}')
    else:
        print(f'  {metric:<8}  {lf_val:>12.4f}')

# ── Collect predictions ───────────────────────────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for cam_skels, qualities, exercise_ids, view_mask in test_loader:
        cam_skels    = cam_skels.to(DEVICE)
        exercise_ids = exercise_ids.to(DEVICE)
        view_mask    = view_mask.to(DEVICE)
        preds        = model(cam_skels, exercise_ids, view_mask)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# ── Save plots ────────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR,    test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2_pcc(history, PLOTS_DIR,      test_r2=final_te['r2'],    test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)
plot_comparison_bar(final_te, PLOTS_DIR)
plot_attention_weights(model, val_loader, PLOTS_DIR)   # attention only

json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 18 — Per-exercise test metrics
# ══════════════════════════════════════════════════════════════════════════

all_true_q_arr  = np.array(all_true_q)
all_pred_q_arr  = np.array(all_pred_q)
all_ex_arr      = np.array(all_exercise_ids)

unique_exercises = sorted(np.unique(all_ex_arr))
per_ex_results   = {}

print('=' * 72)
print(f'  {"Exercise":<12} {"n":>5} {"MAE":>8} {"RMSE":>8} {"R²":>8} {"PCC":>8}  {"ΔvsEF MAE":>10}')
print('─' * 72)

# EF per-exercise results from the previous run (for comparison)
EF_PER_EX = {
    0: 0.3794, 1: 0.3866, 2: 0.2836, 3: 0.7190, 4: 0.4946,
    5: 0.2823, 6: 0.2759, 7: 0.3996, 8: 0.2861, 9: 0.1878,
}

for ex_id in unique_exercises:
    mask = all_ex_arr == ex_id
    qt   = all_true_q_arr[mask]
    qp   = all_pred_q_arr[mask]
    n    = mask.sum()
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp) ** 2)))
    r2   = float(r2_score(qt, qp))    if n > 1 else float('nan')
    pcc  = float(pearsonr(qt, qp)[0]) if n > 1 else float('nan')
    per_ex_results[ex_id] = dict(n=n, mae=mae, rmse=rmse, r2=r2, pcc=pcc)
    ef_mae = EF_PER_EX.get(ex_id, float('nan'))
    delta  = mae - ef_mae
    arrow  = '↑' if delta < 0 else '↓'
    print(f'  E{ex_id:<11} {n:>5} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pcc:>8.4f}  {delta:>+9.4f} {arrow}')

print('─' * 72)
print(f'  {"Overall":<12} {len(all_true_q_arr):>5} '
      f'{final_te["mae"]:>8.4f} {final_te["rmse"]:>8.4f} '
      f'{final_te["r2"]:>8.4f} {final_te["pcc"]:>8.4f}')
print('=' * 72)

per_ex_df  = pd.DataFrame([{'exercise': f'E{ex}', **vals}
                            for ex, vals in per_ex_results.items()])
per_ex_csv = os.path.join(LOGS_DIR, 'per_exercise_metrics.csv')
per_ex_df.to_csv(per_ex_csv, index=False)
print(f'\n  ✓ Per-exercise CSV → {per_ex_csv}')

# ── Per-exercise scatter grid ─────────────────────────────────────────────
n_ex   = len(unique_exercises)
n_cols = 3
n_rows = int(np.ceil(n_ex / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle('Per-Exercise: True vs Predicted — Multi-View Late Fusion (Test)',
             fontsize=14, fontweight='bold')

for i, ex_id in enumerate(unique_exercises):
    ax   = axes[i]
    mask = all_ex_arr == ex_id
    qt   = all_true_q_arr[mask]
    qp   = all_pred_q_arr[mask]
    res  = per_ex_results[ex_id]
    ax.scatter(qt, qp, alpha=0.65, edgecolors='black',
               linewidths=0.4, color='mediumseagreen', s=55)
    lo = min(qt.min(), qp.min()) - 0.2
    hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f'Exercise E{ex_id}  (n={res["n"]})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('True Quality', fontsize=9)
    ax.set_ylabel('Predicted Quality', fontsize=9)
    ax.grid(alpha=0.3)
    ef_mae = EF_PER_EX.get(ex_id, float('nan'))
    ax.text(0.05, 0.97,
            f'LF:  MAE={res["mae"]:.3f}  R²={res["r2"]:.3f}\n'
            f'EF:  MAE={ef_mae:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
scatter_grid_path = os.path.join(PLOTS_DIR, 'per_exercise_scatter.png')
plt.savefig(scatter_grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise scatter grid → {scatter_grid_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 19 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv           = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 70)
print('  TRAINING SUMMARY — Multi-View Late Fusion')
print(f'  Fusion: {FUSION_TYPE}  |  SharedWeights: {SHARED_WEIGHTS}  |  Cameras: {ALL_CAMERAS}')
print('=' * 70)
print(f'  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')
print(f'  Best Val PCC     : {best_val_pcc:.4f}')
print('─' * 70)
print(f'  {"Metric":<8}  {"Late Fusion":>12}  {"Early Fusion":>13}  {"Single-View":>12}  {"ΔvsEF":>8}')
print(f'  {"─"*60}')
for metric, lf_val, ef_val, sv_val in [
    ('MAE',  final_te["mae"],  EF_TEST_MAE,  SV_TEST_MAE),
    ('RMSE', final_te["rmse"], EF_TEST_RMSE, SV_TEST_RMSE),
    ('R²',   final_te["r2"],   EF_TEST_R2,   SV_TEST_R2),
    ('PCC',  final_te["pcc"],  EF_TEST_PCC,  SV_TEST_PCC),
]:
    delta = lf_val - ef_val
    arrow = '↑' if (metric in ['R²', 'PCC'] and delta > 0) or \
                   (metric in ['RMSE', 'MAE'] and delta < 0) else '↓'
    print(f'  {metric:<8}  {lf_val:>12.4f}  {ef_val:>13.4f}  {sv_val:>12.4f}  {delta:>+7.4f} {arrow}')
print('=' * 70)

log.info(f'Late Fusion Summary: Fusion={FUSION_TYPE} Best={best_epoch} stopped={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f} RMSE={final_te["rmse"]:.4f} '
         f'R²={final_te["r2"]:.4f} PCC={final_te["pcc"]:.4f}')
log.info(f'vs EF: ΔMAE={final_te["mae"]-EF_TEST_MAE:+.4f} '
         f'ΔR²={final_te["r2"]-EF_TEST_R2:+.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [
    {'model': 'single_view',
     'test_mae': SV_TEST_MAE, 'test_rmse': SV_TEST_RMSE,
     'test_r2':  SV_TEST_R2,  'test_pcc':  SV_TEST_PCC},
    {'model': 'multiview_early_fusion',
     'test_mae': EF_TEST_MAE, 'test_rmse': EF_TEST_RMSE,
     'test_r2':  EF_TEST_R2,  'test_pcc':  EF_TEST_PCC},
    {'model': f'multiview_late_fusion_{FUSION_TYPE}',
     'fusion':        FUSION_TYPE,
     'shared_weights': SHARED_WEIGHTS,
     'best_epoch':    best_epoch,
     'stopped_epoch': stopped_epoch,
     'val_mae':  best_val_mae,  'val_rmse': best_val_rmse,
     'val_r2':   best_val_r2,   'val_pcc':  best_val_pcc,
     'test_mae': final_te['mae'],  'test_rmse': final_te['rmse'],
     'test_r2':  final_te['r2'],   'test_pcc':  final_te['pcc']},
]
for ex, vals in per_ex_results.items():
    rows.append({'model': 'late_fusion_per_exercise',
                 'exercise': f'E{ex}',
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
