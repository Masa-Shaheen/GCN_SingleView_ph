# ══════════════════════════════════════════════════════════════════════════
# Multi-View Early Fusion ST-GCN  — Clean & Correct Version
# ══════════════════════════════════════════════════════════════════════════

# ── Cell 1 — Configuration ─────────────────────────────────────────────────
import os

DATASET_DIR  = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR    = os.path.join(DATASET_DIR, "by_person")
CSV_PATH     = "/mvdlph/label_events_20260129_155122_stats_short.csv"
OUT_DIR      = "/mvdlph/masa/GCN_MultiView_EarlyFusion_Results"

ALL_CAMERAS  = [0, 1, 2]
NUM_VIEWS    = len(ALL_CAMERAS)
NPZ_KEY      = "keypoints_3d"
NUM_JOINTS   = 17
FUSED_JOINTS = NUM_JOINTS * NUM_VIEWS   # 51

TARGET_FRAMES = 100
EPOCHS        = 300
LR            = 5e-5
BATCH_SIZE    = 32
WEIGHT_DECAY  = 1e-3
PATIENCE      = 50          # increased: model still improving at ep 42
MIN_DELTA     = 1e-4

# cross_view=False: proven better in experiments because cameras
# have different coordinate frames — connecting same joint across
# cameras adds noise, not signal.
CROSS_VIEW    = False

SV_TEST_MAE  = 0.4076
SV_TEST_RMSE = 0.5569
SV_TEST_R2   = 0.4859
SV_TEST_PCC  = 0.7443

# ── Cell 2 — Imports ───────────────────────────────────────────────────────
import re, glob, json, logging, datetime, copy, sys, io
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
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

# ── Cell 3 — Logging ───────────────────────────────────────────────────────
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger("MV-EF-GCN")
log.info(f"Run dir: {RUN_DIR}")
log.info(f"cross_view={CROSS_VIEW}  LR={LR}  WD={WEIGHT_DECAY}  Patience={PATIENCE}")

# ── Cell 4 — Filename parser & skeleton loader ────────────────────────────
def parse_filename(fpath):
    m = re.match(r"E(\d+)_P(\d+)_T(\d+)_C(\d+)_seg(\d+)",
                 os.path.basename(fpath))
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

def load_skeleton(fpath):
    """Load (T, 17, 3) float32 or return None on any failure."""
    if fpath is None:
        return None
    try:
        data = np.load(fpath, allow_pickle=True)
        arr  = data[NPZ_KEY] if NPZ_KEY in data else data[list(data.keys())[0]]
        arr  = arr.astype(np.float32)
        if arr.ndim == 1:   return None
        if arr.ndim == 2:   arr = arr.reshape(arr.shape[0], 17, 3)
        elif arr.ndim == 4: arr = arr.squeeze(0)
        if arr.shape[1] != 17 or arr.shape[2] != 3:
            return None
        return arr
    except Exception:
        return None

print("✓ parse_filename / load_skeleton defined")

# ── Cell 5 — Load CSV labels ───────────────────────────────────────────────
df_csv = None
with open(CSV_PATH, 'rb') as f:
    raw = f.read()
for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']:
    try:
        tmp = pd.read_csv(io.StringIO(raw.decode(enc)))
        tmp.columns = tmp.columns.str.strip()
        if 'exercise' in tmp.columns:
            df_csv = tmp
            print(f"✓ CSV loaded ({enc}), shape={df_csv.shape}")
            break
    except Exception:
        pass
assert df_csv is not None, "❌ CSV not loaded"

# ── Cell 6 — Build multi-view index ───────────────────────────────────────
def build_multiview_index(split_name, df_csv, cameras=ALL_CAMERAS):
    split_path = os.path.join(SPLIT_DIR, split_name)
    all_files  = sorted(glob.glob(os.path.join(split_path, '**/*.npz'), recursive=True))
    print(f"\n[{split_name.upper()}] NPZ files: {len(all_files)}")

    lookup = {}
    for fpath in all_files:
        meta = parse_filename(fpath)
        if meta is None or meta['camera'] not in cameras:
            continue
        key = (meta['exercise'], meta['person'], meta['trial_id'], meta['segment'])
        lookup.setdefault(key, {})[meta['camera']] = fpath

    df_clean = df_csv.copy()
    df_clean.columns = df_clean.columns.str.strip()
    records = []
    for (exercise, person, trial_id, segment), cam_paths in lookup.items():
        trial_num = int(trial_id[1:])
        row = df_clean[
            (df_clean['exercise'] == f"E{exercise}") &
            (df_clean['person']   == person)          &
            (df_clean['trial']    == trial_id)
        ]
        quality   = float(row.iloc[0]['mean']) if len(row) > 0 else np.nan
        trial_key = f"E{exercise}_{person}_{trial_id}"
        record = dict(exercise=exercise, person=person, trial_num=trial_num,
                      trial_id=trial_id, segment=segment, quality=quality,
                      trial_key=trial_key, split=split_name)
        n_avail = 0
        for cam in cameras:
            record[f'filepath_C{cam}'] = cam_paths.get(cam, None)
            if cam_paths.get(cam) is not None:
                n_avail += 1
        record['n_views_available'] = n_avail
        records.append(record)

    df = pd.DataFrame(records)
    # fill missing quality labels
    for trial_type, fill in [(df['trial_num'] <= 2, 4.0), (df['trial_num'] >= 3, 2.5)]:
        mean_q = df.loc[trial_type & df['quality'].notna(), 'quality'].mean()
        df.loc[trial_type & df['quality'].isna(), 'quality'] = mean_q if not np.isnan(mean_q) else fill

    print(f"  Segments: {len(df)}  |  trial keys: {df['trial_key'].nunique()}")
    print(f"  Quality: {df['quality'].mean():.3f} ± {df['quality'].std():.3f}")
    print(f"  Views dist: {df['n_views_available'].value_counts().sort_index().to_dict()}")
    return df

def remove_corrupted(df, cameras=ALL_CAMERAS, label=''):
    """Drop rows where every camera view is missing or unreadable."""
    drop_idx = []
    for i, row in df.iterrows():
        valid = False
        for cam in cameras:
            fp = row.get(f'filepath_C{cam}')
            if fp is not None:
                if load_skeleton(fp) is not None:
                    valid = True
                    break
                else:
                    df.at[i, f'filepath_C{cam}'] = None
        if not valid:
            drop_idx.append(i)
    if drop_idx:
        print(f"  [{label}] Dropping {len(drop_idx)} fully-corrupted rows")
        df = df.drop(index=drop_idx).reset_index(drop=True)
    print(f"  [{label}] Clean: {len(df)}")
    return df

train_df = build_multiview_index('train', df_csv)
val_df   = build_multiview_index('valid', df_csv)
test_df  = build_multiview_index('test',  df_csv)

print("\nChecking corrupted files...")
train_df = remove_corrupted(train_df, label='TRAIN')
val_df   = remove_corrupted(val_df,   label='VALID')
test_df  = remove_corrupted(test_df,  label='TEST')

# Sanity: no trial_key leak
tr_k = set(train_df['trial_key'])
vl_k = set(val_df['trial_key'])
te_k = set(test_df['trial_key'])
assert tr_k.isdisjoint(vl_k), "LEAK train∩val"
assert tr_k.isdisjoint(te_k), "LEAK train∩test"
assert vl_k.isdisjoint(te_k), "LEAK val∩test"
print("✓ No data-leakage across splits")

# ── Cell 7 — Preprocessing ────────────────────────────────────────────────

def normalise_length(skel, target_frames):
    """
    Resample (T, J, C) → (target_frames, J, C) using linear interpolation.
    Vectorised over all J*C channels at once.
    """
    T = skel.shape[0]
    if T == target_frames:
        return skel
    old_t    = np.linspace(0, 1, T)
    new_t    = np.linspace(0, 1, target_frames)
    flat     = skel.reshape(T, -1)
    out_flat = np.stack(
        [np.interp(new_t, old_t, flat[:, k]) for k in range(flat.shape[1])],
        axis=1
    ).astype(np.float32)
    return out_flat.reshape(target_frames, skel.shape[1], skel.shape[2])


def centre_and_scale(skel):
    """
    FIX vs old code: torso height = distance from hip midpoint to
    shoulder midpoint, computed per-frame then averaged.
    This is stable even when the person bends.

    skel: (T, 17, 3) → (T, 17, 3)
    """
    # Centre on mid-hip every frame
    mid_hip = (skel[:, 1:2, :] + skel[:, 4:5, :]) / 2.0   # (T, 1, 3)
    skel    = skel - mid_hip

    # Torso height: distance hip_mid → shoulder_mid, per frame
    mid_shoulder = (skel[:, 11:12, :] + skel[:, 14:15, :]) / 2.0  # (T, 1, 3)
    torso_h = np.linalg.norm(mid_shoulder, axis=-1).mean()          # scalar
    torso_h = max(torso_h, 1e-6)

    return (skel / torso_h).astype(np.float32)


def add_velocity(skel):
    """
    Append per-joint frame-difference as velocity channels.
    skel: (T, J, 3) → (T, J, 6)
    """
    vel     = np.zeros_like(skel)
    vel[1:] = skel[1:] - skel[:-1]
    return np.concatenate([skel, vel], axis=-1).astype(np.float32)


def augment_skeleton(skel, target_frames):
    """
    FIX vs old code: speed jitter is applied BEFORE resampling to
    target_frames, so the temporal distortion is actually preserved.

    skel: (T, J, 3) with arbitrary T  →  (target_frames, J, 3)
    """
    T     = skel.shape[0]
    speed = np.random.uniform(0.8, 1.2)
    new_T = max(10, int(T * speed))

    # Sample at new speed (stretch or compress time)
    old_t  = np.linspace(0, 1, T)
    new_t  = np.linspace(0, 1, new_T)
    flat   = skel.reshape(T, -1)
    warped = np.stack(
        [np.interp(new_t, old_t, flat[:, k]) for k in range(flat.shape[1])],
        axis=1
    ).reshape(new_T, NUM_JOINTS, 3).astype(np.float32)

    # Small random rotation around Y axis (±10°)
    angle  = np.random.uniform(-np.pi / 18, np.pi / 18)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, 0, sin_a],
                  [0,     1, 0    ],
                  [-sin_a,0, cos_a]], dtype=np.float32)
    warped = warped @ R.T

    # Tiny Gaussian noise
    warped += np.random.randn(*warped.shape).astype(np.float32) * 0.002

    # Resample back to target_frames
    return normalise_length(warped, target_frames)


print("✓ Preprocessing helpers defined (centre_and_scale / augment fixes)")


# ── Cell 8 — Dataset ──────────────────────────────────────────────────────

class MultiViewEarlyFusionDataset(Dataset):
    """
    Per sample pipeline:
      For each camera c in [0, 1, 2]:
        1. Load skeleton  → (T, 17, 3)
        2. Augment        → (target_frames, 17, 3)   [train only]
        3. Normalise length → (target_frames, 17, 3)
        4. Centre & scale → (target_frames, 17, 3)
        5. Add velocity   → (target_frames, 17, 6)
        [If camera missing → zero block]
      Concatenate along joint axis → (target_frames, 51, 6)
    """
    def __init__(self, df, cameras=ALL_CAMERAS,
                 target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.cameras       = cameras
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        views = []

        for cam in self.cameras:
            skel = load_skeleton(row.get(f'filepath_C{cam}'))

            if skel is None:
                # Zero-pad for missing/corrupted camera
                block = np.zeros((self.target_frames, NUM_JOINTS, 6), dtype=np.float32)
                views.append(block)
                continue

            # FIX: augment BEFORE normalise_length so speed jitter is real
            if self.augment:
                skel = augment_skeleton(skel, self.target_frames)
            else:
                skel = normalise_length(skel, self.target_frames)

            skel = centre_and_scale(skel)   # (T, 17, 3)
            skel = add_velocity(skel)        # (T, 17, 6)
            views.append(skel)

        # Early fusion: concatenate along joint axis → (T, 51, 6)
        fused = np.concatenate(views, axis=1)

        return (
            torch.tensor(fused,              dtype=torch.float32),
            torch.tensor(row['quality'],     dtype=torch.float32),
            torch.tensor(row['exercise'],    dtype=torch.long),
        )


# Quick shape check
_ds = MultiViewEarlyFusionDataset(train_df)
_s, _q, _e = _ds[0]
assert _s.shape == (TARGET_FRAMES, FUSED_JOINTS, 6), f"Bad shape: {_s.shape}"
print(f"✓ Dataset OK — sample shape: {_s.shape}")
del _ds, _s, _q, _e


# ── Cell 9 — Skeleton edges & adjacency ───────────────────────────────────

SKELETON_EDGES = [
    (0, 1), (1, 2),  (2, 3),           # right leg
    (0, 4), (4, 5),  (5, 6),           # left leg
    (0, 7), (7, 8),  (8, 9), (9, 10),  # spine → head
    (8, 11),(11, 12),(12, 13),          # left arm
    (8, 14),(14, 15),(15, 16),          # right arm
]


def build_adjacency(num_joints, intra_edges, num_views, cross_view=False):
    """
    Kipf-normalised adjacency for a multi-view fused graph.

    cross_view=False (recommended):
        Only intra-view skeleton edges. Views are independent blocks.

    cross_view=True (not recommended unless skeletons are in a shared
        world coordinate frame):
        Adds same-joint edges between all camera pairs. Hurts performance
        when cameras have different coordinate frames (which is typical).
    """
    N = num_joints * num_views
    A = np.zeros((N, N), dtype=np.float32)

    # Intra-view skeleton edges
    for v in range(num_views):
        off = v * num_joints
        for (i, j) in intra_edges:
            A[off + i, off + j] = 1.0
            A[off + j, off + i] = 1.0

    # Cross-view edges (only if skeletons are world-aligned)
    if cross_view:
        for u in range(num_views):
            for v in range(num_views):
                if u != v:
                    for j in range(num_joints):
                        A[u * num_joints + j, v * num_joints + j] = 1.0

    # Kipf: Â = D̃^{-½}(A+I)D̃^{-½}
    A_hat   = A + np.eye(N, dtype=np.float32)
    deg     = A_hat.sum(axis=1)
    d_inv_s = np.diag(np.power(deg, -0.5))
    A_hat   = d_inv_s @ A_hat @ d_inv_s
    return torch.tensor(A_hat, dtype=torch.float32)


# ── Cell 10 — Model ───────────────────────────────────────────────────────

class GraphConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.W = nn.Linear(in_c, out_c, bias=True)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, H, A_hat):
        """H: (B*T, J, in_c) → (B*T, J, out_c)"""
        return torch.bmm(
            A_hat.unsqueeze(0).expand(H.size(0), -1, -1),
            self.W(H)
        )


class GCNBlock(nn.Module):
    """GraphConv → BN → ReLU → Dropout"""
    def __init__(self, in_c, out_c, dropout=0.5, last=False):
        super().__init__()
        self.gcn  = GraphConv(in_c, out_c)
        self.bn   = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout)
        self.last = last

    def forward(self, H, A_hat):
        B_T, J, _ = H.shape
        H = self.gcn(H, A_hat)
        H = self.bn(H.reshape(B_T * J, -1)).reshape(B_T, J, -1)
        if not self.last:
            H = F.relu(H)
            H = self.drop(H)
        return H


class TemporalEncoder(nn.Module):
    """
    BiGRU over time. Takes max over joints before GRU to preserve
    peak motion information better than mean-pooling.
    """
    def __init__(self, feat_dim, hidden_dim=128, num_layers=2, dropout=0.4):
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

    def forward(self, x):
        """x: (B, T, J, C) → (B, 2*hidden_dim)"""
        x = x.max(dim=2)[0]                        # (B, T, C)  max over joints
        _, h = self.gru(x)                          # h: (2*L, B, H)
        out = torch.cat([h[-2], h[-1]], dim=1)      # (B, 2H)
        return self.drop(out)


NUM_EXERCISES = 10

class MultiViewEarlyFusionGCN(nn.Module):
    """
    Input  : (B, T, V*J=51, 6)
    Output : (B,)  quality score in [~0.6, ~5.4]

    Architecture:
      BN on input channels
      3 GCN blocks: 6→64→128→256
      BiGRU temporal encoder (max-pool over joints)
      Exercise embedding (32-d)
      Regression head: 288 → 256 → 128 → 1
      Output: 3 + 2.4 * tanh(head) ∈ (0.6, 5.4)
    """
    def __init__(self,
                 fused_joints = FUSED_JOINTS,
                 in_channels  = 6,
                 gcn_dims     = (64, 128, 256),
                 dropout      = 0.5,
                 cross_view   = CROSS_VIEW):
        super().__init__()

        A_hat = build_adjacency(NUM_JOINTS, SKELETON_EDGES,
                                NUM_VIEWS, cross_view=cross_view)
        self.register_buffer('A_hat', A_hat)

        # Input batch-norm applied per channel
        self.data_bn = nn.BatchNorm1d(in_channels)

        # GCN backbone
        dims = (in_channels,) + tuple(gcn_dims)
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(dims[i], dims[i+1], dropout=dropout,
                     last=(i == len(gcn_dims) - 1))
            for i in range(len(gcn_dims))
        ])

        # Temporal encoder
        self.temporal = TemporalEncoder(
            feat_dim   = gcn_dims[-1],
            hidden_dim = 128,
            num_layers = 2,
            dropout    = 0.4,
        )

        # Exercise conditioning
        self.ex_embed = nn.Embedding(NUM_EXERCISES, 32)

        # Regression head
        head_in = self.temporal.out_dim + 32   # 256 + 32 = 288
        self.reg_head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(p)
                    elif 'bias'  in name: nn.init.zeros_(p)

    def forward(self, x, exercise_id):
        """
        x           : (B, T, J=51, C=6)
        exercise_id : (B,) long
        """
        B, T, J, C = x.shape

        # Input BN: apply per channel across (T*J) positions
        x = x.permute(0, 3, 1, 2).reshape(B, C, T * J)   # (B, C, T*J)
        x = self.data_bn(x)
        x = x.reshape(B, C, T, J).permute(0, 2, 3, 1)    # (B, T, J, C)

        # GCN blocks
        h = x.reshape(B * T, J, C)
        for blk in self.gcn_blocks:
            h = blk(h, self.A_hat)
        h = h.reshape(B, T, J, -1)                        # (B, T, J, 256)

        # Temporal encoding
        h = self.temporal(h)                               # (B, 256)

        # Concatenate exercise embedding
        ex = self.ex_embed(exercise_id)                    # (B, 32)
        h  = torch.cat([h, ex], dim=1)                     # (B, 288)

        # Regression output in (0.6, 5.4) — wider than (1, 5) to avoid
        # gradient saturation at boundary labels
        out = 3.0 + 2.4 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# Sanity check
_x  = torch.zeros(2, TARGET_FRAMES, FUSED_JOINTS, 6)
_ex = torch.zeros(2, dtype=torch.long)
_m  = MultiViewEarlyFusionGCN()
_o  = _m(_x, _ex)
assert _o.shape == (2,)
n_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(f"✓ Model OK — output shape: {_o.shape}  params: {n_params:,}")
del _x, _ex, _m, _o


# ── Cell 11 — Training utilities ──────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_mae   = float('inf')
        self.counter    = 0
        self.best_wts   = None
        self.best_epoch = 1

    def step(self, val_mae, model, epoch):
        improved = val_mae < self.best_mae - self.min_delta
        if improved:
            self.best_mae   = val_mae
            self.counter    = 0
            self.best_wts   = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience, improved


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


def run_epoch(model, loader, loss_fn, optimiser=None):
    """
    optimiser=None → eval mode (no gradients, no weight update).
    Returns dict of loss / rmse / mae / r2 / pcc.
    """
    training = optimiser is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for skels, qualities, ex_ids in loader:
            skels     = skels.to(DEVICE)
            qualities = qualities.to(DEVICE)
            ex_ids    = ex_ids.to(DEVICE)

            preds = model(skels, ex_ids)
            loss  = loss_fn(preds, qualities)

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            total_loss += loss.item()
            q_true.extend(qualities.cpu().numpy())
            q_pred.extend(preds.detach().cpu().numpy())

    qt  = np.array(q_true)
    qp  = np.array(q_pred)
    n   = max(1, len(loader))
    pcc = float(pearsonr(qt, qp)[0]) if len(qt) > 1 else 0.0
    return {
        'loss': total_loss / n,
        'rmse': float(np.sqrt(np.mean((qt - qp) ** 2))),
        'mae' : float(np.mean(np.abs(qt - qp))),
        'r2'  : float(r2_score(qt, qp)) if len(qt) > 1 else 0.0,
        'pcc' : pcc,
    }


# ── Cell 12 — DataLoaders ─────────────────────────────────────────────────

num_workers = min(4, os.cpu_count() or 1)

train_loader = DataLoader(
    MultiViewEarlyFusionDataset(train_df, augment=True),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers, pin_memory=(DEVICE == 'cuda'),
    persistent_workers=(num_workers > 0),
)
val_loader = DataLoader(
    MultiViewEarlyFusionDataset(val_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=(DEVICE == 'cuda'),
    persistent_workers=(num_workers > 0),
)
test_loader = DataLoader(
    MultiViewEarlyFusionDataset(test_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=(DEVICE == 'cuda'),
    persistent_workers=(num_workers > 0),
)

# ── Cell 13 — Training loop ────────────────────────────────────────────────

model     = MultiViewEarlyFusionGCN(cross_view=CROSS_VIEW).to(DEVICE)
loss_fn   = nn.SmoothL1Loss(beta=1.0)       # Huber: robust to outlier labels
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=15,             # reduce LR after 15 epochs no improvement
    min_lr=1e-6, verbose=True
)
early_stop    = EarlyStopping()
history       = {f'{s}_{m}': [] for s in ('train', 'val')
                 for m in ('loss', 'rmse', 'mae', 'r2', 'pcc')}
stopped_epoch = EPOCHS

log.info("=" * 70)
log.info(f"TRAINING START  cross_view={CROSS_VIEW}  LR={LR}  WD={WEIGHT_DECAY}")
log.info(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
log.info("=" * 70)

print(f'\n{"═"*72}')
print(f'  Multi-View Early Fusion | cross_view={CROSS_VIEW} | joints={FUSED_JOINTS}')
print(f'  Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}')
print(f'  LR={LR} WD={WEIGHT_DECAY} Patience={PATIENCE} Batch={BATCH_SIZE}')
print(f'  SV baseline: MAE={SV_TEST_MAE} R²={SV_TEST_R2}')
print(f'{"═"*72}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, loss_fn, optimiser=optimiser)
    vl = run_epoch(model, val_loader,   loss_fn, optimiser=None)
    scheduler.step(vl['mae'])

    for split, res in [('train', tr), ('val', vl)]:
        for m in ('loss', 'rmse', 'mae', 'r2', 'pcc'):
            history[f'{split}_{m}'].append(res[m])

    stop, improved = early_stop.step(vl['mae'], model, epoch)
    star = ' ★' if improved else ''
    msg  = (f"  Ep {epoch:3d}/{EPOCHS} | "
            f"Tr loss={tr['loss']:.4f} mae={tr['mae']:.3f} r2={tr['r2']:.3f} | "
            f"Vl loss={vl['loss']:.4f} mae={vl['mae']:.3f} r2={vl['r2']:.3f} | "
            f"ES {early_stop.counter}/{PATIENCE}{star}")
    print(msg);  log.info(msg)

    if stop:
        stopped_epoch = epoch
        print(f"\n  ⏹  Early stopping at epoch {epoch}  (best={early_stop.best_epoch})")
        log.info(f"Early stopping at epoch {epoch} best={early_stop.best_epoch}")
        break

print("\n✓ Training complete")
model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch

# ── Cell 14 — Evaluation ──────────────────────────────────────────────────

final_te = run_epoch(model, test_loader, loss_fn, optimiser=None)

print(f"\n── Test Results (best epoch = {best_epoch}) ──")
print(f"  {'Metric':<8} {'MV-EF':>10} {'SV-Base':>10} {'Δ':>8}")
print(f"  {'─'*40}")
for name, mv, sv in [
    ('MAE',  final_te['mae'],  SV_TEST_MAE),
    ('RMSE', final_te['rmse'], SV_TEST_RMSE),
    ('R²',   final_te['r2'],   SV_TEST_R2),
    ('PCC',  final_te['pcc'],  SV_TEST_PCC),
]:
    d     = mv - sv
    arrow = '↑' if (name in ('R²','PCC') and d > 0) or \
                   (name in ('MAE','RMSE') and d < 0) else '↓'
    print(f"  {name:<8} {mv:>10.4f} {sv:>10.4f} {d:>+8.4f} {arrow}")

# Collect predictions for per-exercise metrics
model.eval()
all_true, all_pred, all_ex = [], [], []
with torch.no_grad():
    for skels, qualities, ex_ids in test_loader:
        preds = model(skels.to(DEVICE), ex_ids.to(DEVICE))
        all_true.extend(qualities.numpy())
        all_pred.extend(preds.cpu().numpy())
        all_ex.extend(ex_ids.numpy())

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_ex   = np.array(all_ex)

print(f'\n{"═"*72}')
print(f'  {"Exercise":<12} {"n":>5} {"MAE":>8} {"RMSE":>8} {"R²":>8} {"PCC":>8}')
print(f'{"─"*72}')
per_ex = {}
for ex_id in sorted(np.unique(all_ex)):
    mask = all_ex == ex_id
    qt, qp = all_true[mask], all_pred[mask]
    n   = mask.sum()
    mae = float(np.mean(np.abs(qt - qp)))
    rmse= float(np.sqrt(np.mean((qt - qp) ** 2)))
    r2  = float(r2_score(qt, qp))    if n > 1 else np.nan
    pcc = float(pearsonr(qt, qp)[0]) if n > 1 else np.nan
    per_ex[ex_id] = dict(n=n, mae=mae, rmse=rmse, r2=r2, pcc=pcc)
    print(f"  E{ex_id:<11} {n:>5} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pcc:>8.4f}")
print(f'{"─"*72}')
print(f'  {"Overall":<12} {len(all_true):>5} {final_te["mae"]:>8.4f} '
      f'{final_te["rmse"]:>8.4f} {final_te["r2"]:>8.4f} {final_te["pcc"]:>8.4f}')
print(f'{"═"*72}')

# ── Cell 15 — Save plots & summary ────────────────────────────────────────

def _save(fig, name):
    p = os.path.join(PLOTS_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {p}")

epochs_x = range(1, len(history['train_loss']) + 1)

# Loss curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs_x, history['train_loss'], label='Train')
ax.plot(epochs_x, history['val_loss'],   label='Val')
ax.axhline(final_te['loss'], linestyle='-.', color='green', label=f"Test={final_te['loss']:.4f}")
ax.set_title('Loss (Huber)'); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
_save(fig, 'loss_curve.png')

# MAE & RMSE
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m, sv in [(axes[0], 'mae', SV_TEST_MAE), (axes[1], 'rmse', SV_TEST_RMSE)]:
    ax.plot(epochs_x, history[f'train_{m}'], label='Train')
    ax.plot(epochs_x, history[f'val_{m}'],   label='Val')
    ax.axhline(final_te[m], linestyle='-.', color='green', label=f"Test={final_te[m]:.4f}")
    ax.axhline(sv,           linestyle=':',  color='purple',label=f"SV={sv:.4f}")
    ax.set_title(m.upper()); ax.legend(fontsize=8); ax.grid(alpha=0.3)
_save(fig, 'mae_rmse.png')

# R² & PCC
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m, sv in [(axes[0], 'r2', SV_TEST_R2), (axes[1], 'pcc', SV_TEST_PCC)]:
    ax.plot(epochs_x, history[f'train_{m}'], label='Train')
    ax.plot(epochs_x, history[f'val_{m}'],   label='Val')
    ax.axhline(final_te[m], linestyle='-.', color='green', label=f"Test={final_te[m]:.4f}")
    ax.axhline(sv,           linestyle=':',  color='purple',label=f"SV={sv:.4f}")
    ax.set_title(m.upper()); ax.legend(fontsize=8); ax.grid(alpha=0.3)
_save(fig, 'r2_pcc.png')

# Scatter: true vs predicted
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(all_true, all_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.3)
lo, hi = min(all_true.min(), all_pred.min()) - 0.2, max(all_true.max(), all_pred.max()) + 0.2
ax.plot([lo, hi], [lo, hi], 'r--')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('True Quality'); ax.set_ylabel('Predicted Quality')
ax.set_title('Test: True vs Predicted')
ax.text(0.05, 0.95,
        f'MAE={final_te["mae"]:.4f}\nRMSE={final_te["rmse"]:.4f}\nR²={final_te["r2"]:.4f}',
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
_save(fig, 'scatter_test.png')

# Per-exercise bar
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ex_labels = [f'E{e}' for e in sorted(per_ex)]
colors    = plt.cm.tab10(np.linspace(0, 1, len(per_ex)))
for ax, metric, ylabel in [
    (axes[0], 'mae',  'MAE'),
    (axes[1], 'rmse', 'RMSE'),
    (axes[2], 'r2',   'R²'),
    (axes[3], 'pcc',  'PCC'),
]:
    vals = [per_ex[e][metric] for e in sorted(per_ex)]
    ax.bar(ex_labels, vals, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(final_te[metric], color='red', linestyle='--', linewidth=1.5)
    ax.set_title(ylabel); ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', alpha=0.3)
_save(fig, 'per_exercise_bar.png')

# Save training history
with open(os.path.join(LOGS_DIR, 'history.json'), 'w') as f:
    json.dump(history, f, indent=2)

# Save summary CSV
summary_rows = [
    {'model': 'single_view',
     'test_mae': SV_TEST_MAE, 'test_rmse': SV_TEST_RMSE,
     'test_r2':  SV_TEST_R2,  'test_pcc':  SV_TEST_PCC},
    {'model': 'multiview_early_fusion',
     'cross_view': CROSS_VIEW,
     'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
     'test_mae': final_te['mae'],  'test_rmse': final_te['rmse'],
     'test_r2':  final_te['r2'],   'test_pcc':  final_te['pcc'],
     'delta_mae': final_te['mae']  - SV_TEST_MAE,
     'delta_r2' : final_te['r2']   - SV_TEST_R2},
]
for ex_id, vals in per_ex.items():
    summary_rows.append({'model': 'per_exercise', 'exercise': f'E{ex_id}', **vals})

csv_path = os.path.join(OUT_DIR, 'training_summary.csv')
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\n✓ Summary CSV → {csv_path}")

log.info(f"DONE — MAE={final_te['mae']:.4f} R²={final_te['r2']:.4f} "
         f"ΔMAE={final_te['mae']-SV_TEST_MAE:+.4f}")

sys.stdout.restore()
print("✓ Done.")
