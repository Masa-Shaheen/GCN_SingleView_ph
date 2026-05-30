# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR     = os.path.join(DATASET_DIR, "by_person")
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"
CAMERA_IDS    = [0, 1, 2]          # ← ALL 3 cameras used in fusion
NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
TARGET_FRAMES = 120
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 300
LR            = 1e-4
BATCH_SIZE    = 24               # ← reduced: each sample now loads 3× skeletons
WEIGHT_DECAY  = 5e-4
OUT_DIR       = "/mvdlph/masa/GCN_MultiView_MiddleFusion_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Exercise Filter ───────────────────────────────────────────────────────
EXCLUDED_EXERCISES = {3, 7, 9}
EXERCISE_REMAP     = {}           # filled automatically in Cell 7

print('✓ Configuration loaded (Multi-View Middle Fusion)')
print(f'  DATASET_DIR : {DATASET_DIR}')
print(f'  SPLIT_DIR   : {SPLIT_DIR}')
print(f'  CAMERAS     : {CAMERA_IDS}')
print(f'  EXISTS      : {os.path.exists(DATASET_DIR)}')
print(f'  SPLIT EXISTS: {os.path.exists(SPLIT_DIR)}')
print(f'  PATIENCE    : {PATIENCE} epochs')


# ══════════════════════════════════════════════════════════════════════════
# Cell 2 — Imports & output folders
# ══════════════════════════════════════════════════════════════════════════

import os, re, glob, json, logging, datetime, copy, sys, io
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
import random

run_name = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
RUN_DIR  = os.path.join(OUT_DIR, run_name)
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
LOGS_DIR  = os.path.join(RUN_DIR, "logs")

for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)

print(f'✓ Global seed fixed to {SEED}')
print("✓ Run directory created:", RUN_DIR)
print("✓ Libraries imported")


# ══════════════════════════════════════════════════════════════════════════
# Cell 3 — Explore dataset
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

print(f'\nColumns : {df_csv.columns.tolist()}')
print(f'Shape   : {df_csv.shape}')
print(df_csv.to_string())


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
log = logging.getLogger("MultiView-Fusion-Regression")
log.info("=" * 70)
log.info("Multi-View Middle Fusion ST-GCN | BZU Physiotherapy Dataset")
log.info(f"Cameras : {CAMERA_IDS}  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
log.info(f"Log file : {log_file}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader
# ══════════════════════════════════════════════════════════════════════════

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
    try:
        data = np.load(fpath, allow_pickle=True)
        arr  = data[key] if key in data else data[list(data.keys())[0]]
        arr  = arr.astype(np.float32)
        if arr.ndim == 1:
            return None
        if arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], 17, 3)
        elif arr.ndim == 4:
            arr = arr.squeeze(0)
        if arr.shape[1] != 17 or arr.shape[2] != 3:
            return None
        return arr
    except Exception:
        return None

print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build dataset index FROM PRE-SPLIT DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════

def build_index_from_split(split_name, df_csv, camera_id=None):
    split_path = os.path.join(SPLIT_DIR, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    all_files = sorted(glob.glob(
        os.path.join(split_path, '**/*.npz'), recursive=True))
    print(f'\n[{split_name.upper()}] NPZ files found : {len(all_files)}')

    df_csv         = df_csv.copy()
    df_csv.columns = df_csv.columns.str.strip()

    records = []
    for fpath in all_files:
        meta = parse_filename(fpath)
        if meta is None:
            continue
        if camera_id is not None and meta['camera'] != camera_id:
            continue

        row = df_csv[
            (df_csv['exercise'] == f"E{meta['exercise']}") &
            (df_csv['person']   == meta['person'])          &
            (df_csv['trial']    == meta['trial_id'])
        ]
        meta['quality']   = float(row.iloc[0]['mean']) if len(row) > 0 else np.nan
        meta['trial_key'] = f"E{meta['exercise']}_{meta['person']}_{meta['trial_id']}"
        meta['split']     = split_name
        records.append(meta)

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

    print(f'  Samples          : {len(df)}')
    print(f'  Unique trials    : {df["trial_key"].nunique()}')
    print(f'  Quality mean±std : {df["quality"].mean():.3f} ± {df["quality"].std():.3f}')
    return df


def filter_complete_camera_groups(df, label=''):
    """Keep only trial_keys that have ALL 3 cameras, then drop camera filter."""
    REQUIRED_CAMERAS = {0, 1, 2}
    coverage   = df.groupby('trial_key')['camera'].apply(set)
    complete   = coverage[coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))].index
    incomplete = coverage[~coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))]

    n_before = len(df)
    df = df[df['trial_key'].isin(complete)].reset_index(drop=True)
    n_skipped = n_before - len(df)

    if len(incomplete):
        print(f'  [{label}] ⚠️  Skipped {n_skipped} files '
              f'from {len(incomplete)} groups with incomplete cameras')
    else:
        print(f'  [{label}] ✓ All groups have complete 3-camera coverage')

    # ── KEY DIFFERENCE vs single-view:
    # We do NOT filter to a single camera here.
    # All 3 cameras are kept; MultiViewDataset groups them per trial_key.
    print(f'  [{label}] Keeping all cameras for fusion: {len(df)} samples')
    return df


def remove_corrupted(df, label=''):
    bad = []
    for fpath in df['filepath']:
        if load_skeleton(fpath) is None:
            bad.append(fpath)
    if bad:
        print(f'  [{label}] Removing {len(bad)} corrupted file(s)')
        # Remove any trial_key that has at least one bad file
        bad_keys = df[df['filepath'].isin(bad)]['trial_key'].unique()
        df = df[~df['trial_key'].isin(bad_keys)].reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


def exclude_exercises(df, excluded=EXCLUDED_EXERCISES, label=''):
    before = len(df)
    df = df[~df['exercise'].isin(excluded)].reset_index(drop=True)
    print(f'  [{label}] Excluded E{sorted(excluded)} → '
          f'{before - len(df)} rows dropped, {len(df)} remain')
    return df


# ── STEP 1: Build raw splits (no camera_id filter) ───────────────────────
train_df = build_index_from_split('train', df_csv, camera_id=None)
val_df   = build_index_from_split('valid', df_csv, camera_id=None)
test_df  = build_index_from_split('test',  df_csv, camera_id=None)

# ── STEP 2: Camera completeness filter ───────────────────────────────────
train_df = filter_complete_camera_groups(train_df, 'TRAIN')
val_df   = filter_complete_camera_groups(val_df,   'VALID')
test_df  = filter_complete_camera_groups(test_df,  'TEST')

# ── STEP 3: Remove corrupted files (removes whole trial if any cam bad) ───
print('\nChecking for corrupted files...')
train_df = remove_corrupted(train_df, 'TRAIN')
val_df   = remove_corrupted(val_df,   'VALID')
test_df  = remove_corrupted(test_df,  'TEST')

# ── STEP 4: Exclude exercises ────────────────────────────────────────────
print('\nExcluding exercises...')
train_df = exclude_exercises(train_df, label='TRAIN')
val_df   = exclude_exercises(val_df,   label='VALID')
test_df  = exclude_exercises(test_df,  label='TEST')

# ── STEP 5: Remap exercise IDs ───────────────────────────────────────────
remaining_exercises = sorted(
    set(train_df['exercise'].unique()) |
    set(val_df['exercise'].unique())   |
    set(test_df['exercise'].unique())
)
EXERCISE_REMAP = {orig: new for new, orig in enumerate(remaining_exercises)}
print(f'\n  Exercise ID remap : {EXERCISE_REMAP}')

for df_ in [train_df, val_df, test_df]:
    df_['exercise'] = df_['exercise'].map(EXERCISE_REMAP)

# ── STEP 6: Build trial-level DataFrames (one row per trial_key) ──────────
# Each row has 3 filepath columns: path_c0, path_c1, path_c2
def build_trial_df(df):
    """
    Collapse per-file rows → one row per (trial_key, segment) with
    three filepath columns (one per camera).
    quality, exercise, trial_num, person are the same for all cameras.
    """
    rows = []
    # Group by trial_key + segment (each segment is one sample in the original code)
    for (trial_key, segment), grp in df.groupby(['trial_key', 'segment']):
        cam_map = {int(r['camera']): r['filepath'] for _, r in grp.iterrows()}
        if not all(c in cam_map for c in [0, 1, 2]):
            continue   # skip incomplete (shouldn't happen after filter)
        row_ref = grp.iloc[0]
        rows.append({
            'trial_key' : trial_key,
            'segment'   : segment,
            'exercise'  : row_ref['exercise'],
            'person'    : row_ref['person'],
            'trial_num' : row_ref['trial_num'],
            'trial_id'  : row_ref['trial_id'],
            'quality'   : row_ref['quality'],
            'path_c0'   : cam_map[0],
            'path_c1'   : cam_map[1],
            'path_c2'   : cam_map[2],
        })
    return pd.DataFrame(rows).reset_index(drop=True)


train_trial_df = build_trial_df(train_df)
val_trial_df   = build_trial_df(val_df)
test_trial_df  = build_trial_df(test_df)

# ── Leakage check on trial level ─────────────────────────────────────────
tr_keys = set(train_trial_df['trial_key'])
vl_keys = set(val_trial_df['trial_key'])
te_keys = set(test_trial_df['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

print(f'\n{"═"*68}')
print(f'  {"Split":<8} {"Trials":>8} {"Samples":>9}')
print(f'  {"─"*30}')
for name, d in [('Train', train_trial_df), ('Val', val_trial_df), ('Test', test_trial_df)]:
    print(f'  {name:<8} {d["trial_key"].nunique():>8} {len(d):>9}')
print(f'{"═"*68}')
print('\n✓ Trial-level DataFrames ready (path_c0, path_c1, path_c2 per row)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Skeleton visualisation helpers  (unchanged from single-view)
# ══════════════════════════════════════════════════════════════════════════

SKELETON_EDGES = [
    (0, 1), (1, 2),  (2, 3),
    (0, 4), (4, 5),  (5, 6),
    (0, 7), (7, 8),  (8, 9),
    (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

JOINT_NAMES = [
    'Hip', 'R-Hip', 'R-Knee', 'R-Ankle',
    'L-Hip', 'L-Knee', 'L-Ankle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'L-Shoulder', 'L-Elbow', 'L-Wrist',
    'R-Shoulder', 'R-Elbow', 'R-Wrist',
]

JOINT_COLORS = {
    'head' : [9, 10],
    'arms' : [11, 12, 13, 14, 15, 16],
    'torso': [0, 7, 8],
    'legs' : [1, 2, 3, 4, 5, 6],
}
PART_COLOR = {
    'head': 'gold', 'arms': 'dodgerblue',
    'torso': 'limegreen', 'legs': 'tomato',
}

print('✓ Skeleton visualisation helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — ST-GCN Adjacency (identical to single-view)
# ══════════════════════════════════════════════════════════════════════════

from collections import deque

def get_joint_distances(num_joints, edges, center_joint=0):
    adj = {i: [] for i in range(num_joints)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    dist  = [-1] * num_joints
    dist[center_joint] = 0
    queue = deque([center_joint])
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if dist[nbr] == -1:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def build_stgcn_adjacency(num_joints, edges, center_joint=0):
    dist = get_joint_distances(num_joints, edges, center_joint)
    A = np.zeros((3, num_joints, num_joints), dtype=np.float32)
    for i in range(num_joints):
        A[0, i, i] = 1.0
    for (i, j) in edges:
        if dist[j] < dist[i]:
            A[1, i, j] = 1.0
        elif dist[j] > dist[i]:
            A[2, i, j] = 1.0
        else:
            A[1, i, j] = 1.0
        if dist[i] < dist[j]:
            A[1, j, i] = 1.0
        elif dist[i] > dist[j]:
            A[2, j, i] = 1.0
        else:
            A[1, j, i] = 1.0
    for k in range(3):
        row_sum  = A[k].sum(axis=1)
        d_inv_sq = np.where(row_sum > 0, np.power(row_sum, -0.5), 0.0)
        D_inv_sq = np.diag(d_inv_sq)
        A[k]     = D_inv_sq @ A[k] @ D_inv_sq
    return torch.tensor(A, dtype=torch.float32)


print('✓ ST-GCN adjacency builders defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — ST-GCN Building Blocks  (identical to single-view)
# ══════════════════════════════════════════════════════════════════════════

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K    = K
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)
        self.M    = nn.Parameter(torch.zeros(K, NUM_JOINTS, NUM_JOINTS))
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        B, C, T, J = x.shape
        x   = self.conv(x)
        x   = x.view(B, self.K, -1, T, J)
        A_eff = A + self.M
        out = torch.einsum('bkctj,kjv->bctv', x, A_eff)
        out = self.bn(out)
        return self.relu(out)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, K=3,
                 temporal_kernel=9, stride=1, dropout=0.5, residual=True):
        super().__init__()
        pad = (temporal_kernel - 1) // 2
        self.spatial  = SpatialGraphConv(in_channels, out_channels, K)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1),
                      padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x   = self.spatial(x, A)
        x   = self.temporal(x)
        return self.relu(x + res)


print('✓ SpatialGraphConv and STGCNBlock defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — Attention Fusion Module  ← NEW
# ══════════════════════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    """
    Middle Fusion via multi-head self-attention across K=3 camera tokens.

    Each ST-GCN branch produces a 256-d feature vector (after GAP).
    These 3 vectors are treated as a sequence of tokens:
        x ∈ (B, 3, 256)

    The module learns:
      (1) which camera views are more informative  (via softmax pooling weights)
      (2) cross-view interactions                  (via self-attention)
      (3) camera-specific context                  (via camera positional embedding)

    Returns:
      fused        : (B, feat_dim)   — fused representation
      attn_weights : (B, heads, 3, 3) — interpretable attention map
    """
    def __init__(self, feat_dim=256, num_heads=4, dropout=0.1, num_cameras=3):
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_cameras = num_cameras

        # Learnable camera positional embedding: tells the model which view it is
        self.cam_embed = nn.Embedding(num_cameras, feat_dim)

        # Optional projection (identity if feat_dim unchanged, useful if you change it)
        self.input_proj = nn.Linear(feat_dim, feat_dim)

        # Single multi-head self-attention layer
        # batch_first=True → input shape (B, seq_len, feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim    = feat_dim,
            num_heads    = num_heads,
            dropout      = dropout,
            batch_first  = True,
        )

        # LayerNorm + FFN (Transformer-style post-attention processing)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Dropout(dropout),
        )

        # Learned pooling: score each of the 3 tokens → weighted sum
        # Replaces naive mean-pooling; model learns which camera to trust
        self.pool_gate = nn.Linear(feat_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cam_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.pool_gate.weight)
        nn.init.zeros_(self.pool_gate.bias)

    def forward(self, feats):
        """
        feats : list of num_cameras tensors, each (B, feat_dim)
        returns:
          fused        : (B, feat_dim)
          attn_weights : (B, num_cameras, num_cameras)  — averaged over heads
        """
        # Stack → (B, num_cameras, feat_dim)
        x = torch.stack(feats, dim=1)                          # (B, 3, D)

        # Add camera positional embedding
        cam_ids = torch.arange(self.num_cameras, device=x.device)
        x = x + self.cam_embed(cam_ids).unsqueeze(0)           # (B, 3, D)

        # Project input
        x = self.input_proj(x)                                 # (B, 3, D)

        # ── Self-Attention ────────────────────────────────────────────────
        # need_weights=True → returns average attention over heads
        attn_out, attn_weights = self.attn(x, x, x,
                                           need_weights=True,
                                           average_attn_weights=True)
        # attn_weights: (B, 3, 3)

        x = self.norm1(x + attn_out)                           # residual

        # ── FFN ───────────────────────────────────────────────────────────
        x = self.norm2(x + self.ffn(x))                        # (B, 3, D)

        # ── Weighted pooling (learned) ────────────────────────────────────
        scores = self.pool_gate(x)                             # (B, 3, 1)
        scores = torch.softmax(scores, dim=1)                  # (B, 3, 1)
        fused  = (x * scores).sum(dim=1)                       # (B, D)

        return fused, attn_weights


print('✓ AttentionFusion module defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — ST-GCN Backbone (shared, feature-extraction mode)  ← UPDATED
# ══════════════════════════════════════════════════════════════════════════

NUM_EXERCISES = len(EXERCISE_REMAP)

class STGCNBackbone(nn.Module):
    """
    Shared ST-GCN backbone used by ALL camera branches.

    Differences from single-view STGCN_Regression:
      • No exercise embedding here  (added once after fusion)
      • No regression head          (added once after fusion)
      • forward() always returns 256-d feature vector (after GAP)
      • Weights are shared across cameras → view-invariant features,
        fewer parameters, less overfitting
    """
    def __init__(self, in_features=6, K=3, dropout=0.5):
        super().__init__()

        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_features * NUM_JOINTS)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_features, 64,  K=K, residual=False, dropout=dropout),
            STGCNBlock(64,  64,           K=K,                dropout=dropout),
            STGCNBlock(64,  64,           K=K,                dropout=dropout),
            STGCNBlock(64,  128, K=K, stride=2,               dropout=dropout),
            STGCNBlock(128, 128,          K=K,                dropout=dropout),
            STGCNBlock(128, 128,          K=K,                dropout=dropout),
            STGCNBlock(128, 256, K=K, stride=2,               dropout=dropout),
            STGCNBlock(256, 256,          K=K,                dropout=dropout),
            STGCNBlock(256, 256,          K=K,                dropout=dropout),
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x : (B, T, J, C)
        returns : (B, 256)  — feature vector, ready for fusion
        """
        B, T, J, C = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, C * J, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, J, T).permute(0, 1, 3, 2)         # (B, C, T, J)
        for block in self.blocks:
            x = block(x, self.A)
        x = self.gap(x).squeeze(-1).squeeze(-1)                 # (B, 256)
        return x


print('✓ STGCNBackbone (shared, 256-d output) defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Full Multi-View Fusion Model  ← NEW
# ══════════════════════════════════════════════════════════════════════════

class MultiViewFusionModel(nn.Module):
    """
    Middle Fusion Architecture:

        Camera 0 ─→ STGCNBackbone ─→ 256-d ──┐
        Camera 1 ─→ STGCNBackbone ─→ 256-d ──┼─→ AttentionFusion ─→ 256-d
        Camera 2 ─→ STGCNBackbone ─→ 256-d ──┘
                                                        │
                                              Exercise Embedding (32-d)
                                                        │
                                               Concatenate (288-d)
                                                        │
                                               Regression Head
                                                        │
                                              Quality Score [1, 5]

    Key design choices:
      • SHARED backbone across all 3 cameras
          → forces view-invariant feature learning
          → 3× fewer parameters than separate backbones
          → regularises against overfitting on medium dataset
      • Attention fusion learns:
          → cross-view dependencies (e.g. front+side together)
          → which camera contributes most per sample
          → camera-specific context via positional embedding
      • Exercise embedding injected ONCE after fusion
          → avoids redundancy of conditioning each backbone separately
    """
    def __init__(self,
                 in_features   = 6,
                 feat_dim      = 256,
                 K             = 3,
                 backbone_drop = 0.5,
                 fusion_heads  = 4,
                 fusion_drop   = 0.1,
                 head_drop     = 0.3,
                 num_cameras   = 3):
        super().__init__()

        # ── Shared backbone (all cameras share weights) ───────────────────
        self.backbone = STGCNBackbone(
            in_features = in_features,
            K           = K,
            dropout     = backbone_drop,
        )

        # ── Attention Fusion ──────────────────────────────────────────────
        self.fusion = AttentionFusion(
            feat_dim    = feat_dim,
            num_heads   = fusion_heads,
            dropout     = fusion_drop,
            num_cameras = num_cameras,
        )

        # ── Exercise embedding ────────────────────────────────────────────
        self.ex_embed = nn.Embedding(NUM_EXERCISES, 32)

        # ── Regression Head ───────────────────────────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim + 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(head_drop * 0.67),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_head()

    def _init_head(self):
        for m in self.reg_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.ex_embed.weight, std=0.01)

    def forward(self, x_c0, x_c1, x_c2, exercise_id,
                return_attn=False):
        """
        x_c0, x_c1, x_c2 : (B, T, J, C)   — one tensor per camera
        exercise_id       : (B,)            — remapped exercise index
        return_attn       : bool            — also return attention weights

        returns : (B,) quality scores in [1, 5]
        optionally: also (B, num_cameras, num_cameras) attention weights
        """
        # ── Feature extraction (shared backbone) ─────────────────────────
        f0 = self.backbone(x_c0)   # (B, 256)
        f1 = self.backbone(x_c1)   # (B, 256)
        f2 = self.backbone(x_c2)   # (B, 256)

        # ── Attention fusion ──────────────────────────────────────────────
        fused, attn_w = self.fusion([f0, f1, f2])   # (B, 256), (B, 3, 3)

        # ── Exercise conditioning ─────────────────────────────────────────
        ex  = self.ex_embed(exercise_id)             # (B, 32)
        h   = torch.cat([fused, ex], dim=1)          # (B, 288)

        # ── Regression ───────────────────────────────────────────────────
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))

        if return_attn:
            return out, attn_w
        return out


# ── Sanity check ──────────────────────────────────────────────────────────
_dummy_x  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, 6)
_dummy_ex = torch.zeros(2, dtype=torch.long)
_model    = MultiViewFusionModel()
_out      = _model(_dummy_x, _dummy_x, _dummy_x, _dummy_ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"

total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'\n✓ MultiViewFusionModel sanity check passed — output shape: {_out.shape}')
print(f'✓ Total trainable parameters: {total_params:,}')
del _dummy_x, _dummy_ex, _model, _out


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — MultiViewDataset  ← NEW (replaces BZUDataset)
# ══════════════════════════════════════════════════════════════════════════

class MultiViewDataset(Dataset):
    """
    Each sample loads 3 skeletons simultaneously (one per camera).

    Returns:
        skel_c0, skel_c1, skel_c2 : (T, J, 6)  position + velocity
        quality                   : scalar float
        exercise_id               : int
    """
    def __init__(self, trial_df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = trial_df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        skels = []
        for cam_col in ['path_c0', 'path_c1', 'path_c2']:
            skel = load_skeleton(row[cam_col])
            if skel is None:
                skel = np.zeros((self.target_frames, 17, 3), dtype=np.float32)
            skel = self._normalise_length(skel)
            if self.augment:
                skel = self._augment(skel)
            skel = self._add_velocity(skel)    # (T, J, 6)
            skels.append(torch.tensor(skel, dtype=torch.float32))

        quality     = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)

        return skels[0], skels[1], skels[2], quality, exercise_id

    def _normalise_length(self, skel):
        T = skel.shape[0]
        if T == self.target_frames:
            return skel
        old_idx = np.linspace(0, 1, T)
        new_idx = np.linspace(0, 1, self.target_frames)
        out = np.zeros((self.target_frames, skel.shape[1], skel.shape[2]),
                       dtype=np.float32)
        for j in range(skel.shape[1]):
            for ax in range(skel.shape[2]):
                out[:, j, ax] = np.interp(new_idx, old_idx, skel[:, j, ax])
        return out

    def _add_velocity(self, skel):
        """Append finite-difference velocity → (T, J, 6)."""
        velocity      = np.zeros_like(skel)
        velocity[1:]  = skel[1:] - skel[:-1]
        return np.concatenate([skel, velocity], axis=-1)   # (T, J, 6)

    def _augment(self, skel):
        T = skel.shape[0]
        # 1. Random temporal speed warp
        speed = np.random.uniform(0.75, 1.25)
        n_new = max(10, int(T * speed))
        idxs  = np.linspace(0, T - 1, n_new).astype(int)
        skel  = self._normalise_length(skel[idxs])
        # 2. Random frame drop
        keep_ratio = np.random.uniform(0.80, 1.0)
        n_keep     = max(10, int(self.target_frames * keep_ratio))
        keep_idxs  = np.sort(
            np.random.choice(self.target_frames, n_keep, replace=False))
        skel = self._normalise_length(skel[keep_idxs])
        return skel


print('✓ MultiViewDataset defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Device, normalisation & run_epoch  ← UPDATED
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    x: (B, T, J, 6)  — first 3 = position, last 3 = velocity
    Centres on hip midpoint, scales by torso height.
    Applied identically to all camera branches.
    """
    pos = x[:, :, :, :3]
    vel = x[:, :, :, 3:]
    hip      = (pos[:, :, 1:2, :] + pos[:, :, 4:5, :]) / 2.0
    pos      = pos - hip
    shoulder = (pos[:, :, 11:12, :] + pos[:, :, 14:15, :]) / 2.0
    torso_h  = shoulder[:, :, :, 1:2].abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
    pos      = pos / torso_h
    vel      = vel / torso_h
    return torch.cat([pos, vel], dim=-1)


def run_epoch(model, loader, reg_fn, is_train=True, optimiser=None):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        # ── Unpack 5 values from MultiViewDataset ─────────────────────────
        for x_c0, x_c1, x_c2, qualities, exercise_ids in loader:
            x_c0         = centre_and_scale(x_c0.to(DEVICE))
            x_c1         = centre_and_scale(x_c1.to(DEVICE))
            x_c2         = centre_and_scale(x_c2.to(DEVICE))
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)

            preds = model(x_c0, x_c1, x_c2, exercise_ids)
            loss  = reg_fn(preds, qualities)

            if is_train:
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


print('✓ centre_and_scale and run_epoch (multi-view) defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Plotting helpers  (identical to single-view)
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def _add_test_line(ax, val, label, color='green'):
    ax.axhline(val, color=color, linestyle='-.', linewidth=1.5,
               label=f'Test {label}={val:.4f}')


def plot_loss_curves(history, save_dir, test_loss=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    if test_loss is not None:
        _add_test_line(ax, test_loss, 'Loss')
    ax.set_title('Regression Loss (SmoothL1)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curve.png'))


def plot_rmse_mae(history, save_dir, test_rmse=None, test_mae=None):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('RMSE & MAE — Train / Val / Test', fontsize=14, fontweight='bold')
    for ax, metric, title, test_val in [
        (axes[0], 'rmse', 'RMSE', test_rmse),
        (axes[1], 'mae',  'MAE',  test_mae),
    ]:
        ax.plot(epochs, history[f'train_{metric}'], label='Train',      color='steelblue')
        ax.plot(epochs, history[f'val_{metric}'],   label='Validation', color='darkorange')
        if test_val is not None:
            _add_test_line(ax, test_val, title)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'rmse_mae.png'))


def plot_r2(history, save_dir, test_r2=None):
    epochs = range(1, len(history['val_r2']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    if test_r2 is not None:
        _add_test_line(ax, test_r2, 'R²')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (R²=1)')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='Baseline (R²=0)')
    ax.set_title('R² Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('R²')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'r2_curve.png'))


def plot_pcc(history, save_dir, test_pcc=None):
    epochs = range(1, len(history['val_pcc']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['train_pcc'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_pcc'],   label='Validation', color='darkorange')
    if test_pcc is not None:
        _add_test_line(ax, test_pcc, 'PCC')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (PCC=1)')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='No correlation')
    ax.set_title('Pearson Correlation Coefficient', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('PCC')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'pcc_curve.png'))


def plot_regression_scatter(q_true, q_pred, split_name='Test', save_dir=None):
    qt   = np.array(q_true)
    qp   = np.array(q_pred)
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp) ** 2)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(qt, qp, alpha=0.6, edgecolors='black', linewidths=0.4,
               color='steelblue', s=60)
    lo = min(qt.min(), qp.min()) - 0.2
    hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True Quality Score',      fontsize=12)
    ax.set_ylabel('Predicted Quality Score', fontsize=12)
    ax.set_title(f'{split_name} Set — True vs Predicted Quality',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    textstr = f'R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    if save_dir:
        save_and_show(fig, os.path.join(save_dir,
                      f'regression_scatter_{split_name.lower()}.png'))
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
    ax.set_title('MAE + Early Stopping', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))


def plot_attention_heatmap(attn_weights_list, save_dir):
    """
    Visualise learned attention between camera views.
    attn_weights_list: list of (B, 3, 3) tensors from test set
    """
    all_attn = torch.cat(attn_weights_list, dim=0).cpu().numpy()  # (N, 3, 3)
    mean_attn = all_attn.mean(axis=0)                              # (3, 3)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_attn, cmap='Blues', vmin=0, vmax=mean_attn.max())
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['C0', 'C1', 'C2'])
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['C0', 'C1', 'C2'])
    ax.set_title('Mean Attention Weights\n(Query → Key across Camera Views)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Key Camera')
    ax.set_ylabel('Query Camera')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mean_attn[i, j]:.3f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if mean_attn[i, j] > mean_attn.max() * 0.6 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'attention_heatmap.png'))


print('✓ Plotting helpers (multi-view) defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Early Stopping  (identical to single-view)
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
# Cell 18 — DataLoaders  ← UPDATED (uses MultiViewDataset + trial_df)
# ══════════════════════════════════════════════════════════════════════════

def make_weighted_sampler(trial_df):
    """
    Same weighting logic as single-view:
      lower quality → higher sampling probability
      erroneous trials (trial_num >= 3) → ×2 weight
    """
    q = trial_df['quality'].values.astype(np.float32)
    raw_weights = (q.max() + 1.0) - q
    erroneous_mask = (trial_df['trial_num'].values >= 3).astype(np.float32)
    raw_weights   *= (1.0 + erroneous_mask)
    weights_tensor = torch.DoubleTensor(raw_weights)
    return torch.utils.data.WeightedRandomSampler(
        weights     = weights_tensor,
        num_samples = len(trial_df),
        replacement = True,
    )


train_sampler = make_weighted_sampler(train_trial_df)

train_loader = DataLoader(
    MultiViewDataset(train_trial_df, augment=True),
    batch_size  = BATCH_SIZE,
    sampler     = train_sampler,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)
val_loader = DataLoader(
    MultiViewDataset(val_trial_df,   augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)
test_loader = DataLoader(
    MultiViewDataset(test_trial_df,  augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)

print(f'✓ DataLoaders ready')
print(f'  Train batches : {len(train_loader)}  ({len(train_trial_df)} samples)')
print(f'  Val   batches : {len(val_loader)}  ({len(val_trial_df)} samples)')
print(f'  Test  batches : {len(test_loader)}  ({len(test_trial_df)} samples)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 19 — Model, Optimiser, Scheduler  ← UPDATED
# ══════════════════════════════════════════════════════════════════════════

reg_fn = nn.SmoothL1Loss(beta=1.0)

model = MultiViewFusionModel(
    in_features   = 6,
    feat_dim      = 256,
    K             = 3,
    backbone_drop = 0.5,
    fusion_heads  = 4,
    fusion_drop   = 0.1,
    head_drop     = 0.3,
    num_cameras   = 3,
).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(),
    lr           = LR,
    weight_decay = WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=50, min_lr=1e-6, verbose=True,
)

early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val']
METRICS = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info('=' * 70)
log.info('STARTING MULTI-VIEW MIDDLE FUSION TRAINING')
log.info(f'Model params : {total_params:,}')
log.info(f'train={len(train_trial_df)} val={len(val_trial_df)} test={len(test_trial_df)}')
log.info('=' * 70)

print(f'\n{"═"*68}')
print(f'  Multi-View Middle Fusion | Cameras: {CAMERA_IDS}')
print(f'  Train: {len(train_trial_df)}  Val: {len(val_trial_df)}  Test: {len(test_trial_df)}')
print(f'  Params: {total_params:,}  |  Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 20 — Training Loop  ← UPDATED
# ══════════════════════════════════════════════════════════════════════════

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, reg_fn, is_train=True,  optimiser=optimiser)
    vl = run_epoch(model, val_loader,   reg_fn, is_train=False)
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
final_te = run_epoch(model, test_loader, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')
print(f'  PCC  : {final_te["pcc"]:.4f}')

# ── Collect predictions + attention weights ───────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
all_attn_weights = []

with torch.no_grad():
    for x_c0, x_c1, x_c2, qualities, exercise_ids in test_loader:
        x_c0         = centre_and_scale(x_c0.to(DEVICE))
        x_c1         = centre_and_scale(x_c1.to(DEVICE))
        x_c2         = centre_and_scale(x_c2.to(DEVICE))
        exercise_ids = exercise_ids.to(DEVICE)

        preds, attn_w = model(x_c0, x_c1, x_c2, exercise_ids, return_attn=True)

        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())
        all_attn_weights.append(attn_w.cpu())


# ══════════════════════════════════════════════════════════════════════════
# Cell 21 — Save all plots
# ══════════════════════════════════════════════════════════════════════════

plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR,    test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR,          test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR,         test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)
plot_attention_heatmap(all_attn_weights, PLOTS_DIR)   # ← NEW: shows which camera matters

# ── Save history JSON ─────────────────────────────────────────────────────
json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')

# ── Save NPZs ─────────────────────────────────────────────────────────────
np.savez(os.path.join(LOGS_DIR, 'training_history.npz'),
         **{k: np.array(v) for k, v in history.items()})

np.savez(os.path.join(LOGS_DIR, 'test_predictions.npz'),
         q_true       = np.array(all_true_q),
         q_pred       = np.array(all_pred_q),
         exercise_ids = np.array(all_exercise_ids),
)
print(f'  ✓ Predictions NPZ saved')


# ══════════════════════════════════════════════════════════════════════════
# Cell 22 — Per-exercise test metrics  (identical logic to single-view)
# ══════════════════════════════════════════════════════════════════════════

all_true_q_arr  = np.array(all_true_q)
all_pred_q_arr  = np.array(all_pred_q)
all_ex_arr      = np.array(all_exercise_ids)

unique_exercises = sorted(np.unique(all_ex_arr))
per_ex_results   = {}

print('=' * 72)
print(f'  {"Exercise":<12} {"n":>5} {"MAE":>8} {"RMSE":>8} {"R²":>8} {"PCC":>8}')
print('─' * 72)

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
    print(f'  E{ex_id:<11} {n:>5} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pcc:>8.4f}')

print('─' * 72)
print(f'  {"Overall":<12} {len(all_true_q_arr):>5} '
      f'{final_te["mae"]:>8.4f} {final_te["rmse"]:>8.4f} '
      f'{final_te["r2"]:>8.4f} {final_te["pcc"]:>8.4f}')
print('=' * 72)

# ── Save per-exercise CSV ──────────────────────────────────────────────────
per_ex_df = pd.DataFrame([
    {'exercise': f'E{ex}', **vals}
    for ex, vals in per_ex_results.items()
])
per_ex_csv = os.path.join(LOGS_DIR, 'per_exercise_metrics.csv')
per_ex_df.to_csv(per_ex_csv, index=False)
print(f'\n  ✓ Per-exercise CSV → {per_ex_csv}')

# ── Per-exercise scatter plots ────────────────────────────────────────────
n_ex   = len(unique_exercises)
n_cols = 3
n_rows = int(np.ceil(n_ex / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle('Per-Exercise: True vs Predicted Quality (Test Set)',
             fontsize=14, fontweight='bold')

for i, ex_id in enumerate(unique_exercises):
    ax   = axes[i]
    mask = all_ex_arr == ex_id
    qt   = all_true_q_arr[mask]
    qp   = all_pred_q_arr[mask]
    res  = per_ex_results[ex_id]
    ax.scatter(qt, qp, alpha=0.65, edgecolors='black',
               linewidths=0.4, color='steelblue', s=55)
    lo = min(qt.min(), qp.min()) - 0.2
    hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f'Exercise E{ex_id}  (n={res["n"]})', fontsize=10, fontweight='bold')
    ax.set_xlabel('True Quality', fontsize=9)
    ax.set_ylabel('Predicted Quality', fontsize=9)
    ax.grid(alpha=0.3)
    textstr = (f'MAE  = {res["mae"]:.3f}\n'
               f'RMSE = {res["rmse"]:.3f}\n'
               f'R²   = {res["r2"]:.3f}\n'
               f'PCC  = {res["pcc"]:.3f}')
    ax.text(0.05, 0.97, textstr, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
scatter_grid_path = os.path.join(PLOTS_DIR, 'per_exercise_scatter.png')
plt.savefig(scatter_grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise scatter grid → {scatter_grid_path}')

# ── Bar chart ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Per-Exercise Test Metrics', fontsize=13, fontweight='bold')
ex_labels = [f'E{ex}' for ex in unique_exercises]
colors    = plt.cm.tab10(np.linspace(0, 1, len(unique_exercises)))

for ax, metric, ylabel, ylim in [
    (axes[0], 'mae',  'MAE',  None),
    (axes[1], 'rmse', 'RMSE', None),
    (axes[2], 'r2',   'R²',   (-1.05, 1.05)),
    (axes[3], 'pcc',  'PCC',  (-1.05, 1.05)),
]:
    vals = [per_ex_results[ex][metric] for ex in unique_exercises]
    bars = ax.bar(ex_labels, vals, color=colors, edgecolor='black',
                  linewidth=0.6, alpha=0.85)
    ax.set_title(ylabel, fontweight='bold')
    ax.set_ylabel(ylabel); ax.set_xlabel('Exercise')
    ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    for bar, v in zip(bars, vals):
        va_pos = bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, va_pos,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

for ax, metric in zip(axes, ['mae', 'rmse', 'r2', 'pcc']):
    overall = final_te[metric]
    ax.axhline(overall, color='red', linestyle='--',
               linewidth=1.5, label=f'Overall={overall:.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
bar_path = os.path.join(PLOTS_DIR, 'per_exercise_bar.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise bar chart → {bar_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 23 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 60)
print('  TRAINING SUMMARY — Multi-View Middle Fusion')
print('=' * 60)
print(f'  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')
print(f'  Best Val PCC     : {best_val_pcc:.4f}')
print('─' * 60)
print(f'  Test MAE         : {final_te["mae"]:.4f}')
print(f'  Test RMSE        : {final_te["rmse"]:.4f}')
print(f'  Test R²          : {final_te["r2"]:.4f}')
print(f'  Test PCC         : {final_te["pcc"]:.4f}')
print('=' * 60)

log.info(f'Best Epoch={best_epoch}  stopped_epoch={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f}')
log.info(f'Test RMSE={final_te["rmse"]:.4f}')
log.info(f'Test R²={final_te["r2"]:.4f}')
log.info(f'Test PCC={final_te["pcc"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [{'split': 'test_overall', 'exercise': 'all',
         'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
         'val_mae': best_val_mae,  'val_rmse': best_val_rmse,
         'val_r2':  best_val_r2,   'val_pcc':  best_val_pcc,
         'test_mae': final_te['mae'], 'test_rmse': final_te['rmse'],
         'test_r2':  final_te['r2'], 'test_pcc':  final_te['pcc']}]
for ex, vals in per_ex_results.items():
    rows.append({'split': 'test_per_exercise', 'exercise': f'E{ex}',
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
