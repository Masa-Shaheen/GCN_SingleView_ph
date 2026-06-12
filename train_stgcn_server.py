# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR     = os.path.join(DATASET_DIR, "by_person")
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"
NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
TARGET_FRAMES = 120
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 300
LR            = 1e-4
BATCH_SIZE    = 32
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_MultiView_LateFusion_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Camera setup ─────────────────────────────────────────────────────────
ALL_CAMERAS = [0, 1, 2]

# ── Late Fusion strategy ──────────────────────────────────────────────────
# Options:
#   'mean'      — simple average of the 3 camera scores
#   'weighted'  — learned scalar weight per camera (softmax-normalised)
#   'mlp'       — tiny MLP takes the 3 scores as input → final score
LATE_FUSION_MODE = 'weighted'   # ← change to 'mean' / 'weighted' / 'mlp'

# ── Exercise Filter ───────────────────────────────────────────────────────
EXCLUDED_EXERCISES = {0, 2, 3, 4, 5, 6, 7, 8, 9}#E0 THEN E1
EXERCISE_REMAP     = {}    # filled automatically in Cell 7

print('✓ Configuration loaded  (LATE FUSION)')
print(f'  DATASET_DIR      : {DATASET_DIR}')
print(f'  SPLIT_DIR        : {SPLIT_DIR}')
print(f'  NPZ_KEY          : {NPZ_KEY}')
print(f'  CAMERAS          : {ALL_CAMERAS}')
print(f'  LATE_FUSION_MODE : {LATE_FUSION_MODE}')
print(f'  EXISTS           : {os.path.exists(DATASET_DIR)}')
print(f'  SPLIT EXISTS     : {os.path.exists(SPLIT_DIR)}')
print(f'  PATIENCE         : {PATIENCE} epochs')


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

print("✓ Run directory created:", RUN_DIR)

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


# ══════════════════════════════════════════════════════════════════════════
# Cell 3 — Explore dataset
# ══════════════════════════════════════════════════════════════════════════

all_npz = sorted(glob.glob(os.path.join(DATASET_DIR, '**/*.npz'), recursive=True))
print(f'Total NPZ files found : {len(all_npz)}')

if len(all_npz) > 0:
    sample = np.load(all_npz[0])
    print(f'First file keys: {list(sample.keys())}')
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
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger("GCN-LateFusion")
log.info("=" * 70)
log.info("ST-GCN Multi-View LATE FUSION Regression | BZU Physiotherapy Dataset")
log.info(f"Cameras : {ALL_CAMERAS}  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
log.info(f"Late Fusion Mode : {LATE_FUSION_MODE}")
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
        meta['quality']    = float(row.iloc[0]['mean']) if len(row) > 0 else np.nan
        meta['trial_key']  = f"E{meta['exercise']}_{meta['person']}_{meta['trial_id']}"
        meta['sample_key'] = f"{meta['trial_key']}_seg{meta['segment']}"
        meta['split']      = split_name
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

    print(f'  Samples (all cams): {len(df)}')
    print(f'  Unique trial keys : {df["trial_key"].nunique()}')
    return df


def build_multiview_index(df, label=''):
    REQUIRED = set(ALL_CAMERAS)
    groups   = df.groupby('sample_key')
    rows     = []
    skipped  = 0
    for key, grp in groups:
        cams = set(grp['camera'].values)
        if not REQUIRED.issubset(cams):
            skipped += 1
            continue
        rep = grp.iloc[0]
        row = {
            'sample_key' : key,
            'trial_key'  : rep['trial_key'],
            'exercise'   : rep['exercise'],
            'person'     : rep['person'],
            'trial_num'  : rep['trial_num'],
            'trial_id'   : rep['trial_id'],
            'segment'    : rep['segment'],
            'quality'    : rep['quality'],
            'split'      : rep['split'],
        }
        for cam in ALL_CAMERAS:
            fp = grp.loc[grp['camera'] == cam, 'filepath'].values[0]
            row[f'filepath_c{cam}'] = fp
        rows.append(row)

    if skipped:
        print(f'  [{label}] ⚠️  Skipped {skipped} sample_keys with incomplete camera coverage')

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f'  [{label}] Multi-view samples : {len(result)}')
    print(f'  [{label}] Quality mean±std   : '
          f'{result["quality"].mean():.3f} ± {result["quality"].std():.3f}')
    return result


def remove_corrupted_mv(df, label=''):
    bad_keys = set()
    for _, row in df.iterrows():
        for cam in ALL_CAMERAS:
            if load_skeleton(row[f'filepath_c{cam}']) is None:
                bad_keys.add(row['sample_key'])
                break
    if bad_keys:
        print(f'  [{label}] Removing {len(bad_keys)} corrupted sample(s)')
        df = df[~df['sample_key'].isin(bad_keys)].reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


def exclude_exercises(df, excluded=EXCLUDED_EXERCISES, label=''):
    before = len(df)
    df = df[~df['exercise'].isin(excluded)].reset_index(drop=True)
    print(f'  [{label}] Excluded E{sorted(excluded)} → '
          f'{before - len(df)} rows dropped, {len(df)} remain')
    return df


# ── STEP 1: Build raw per-file splits ─────────────────────────────────────
train_df_raw = build_index_from_split('train', df_csv, camera_id=None)
val_df_raw   = build_index_from_split('valid', df_csv, camera_id=None)
test_df_raw  = build_index_from_split('test',  df_csv, camera_id=None)

# ── STEP 2: Convert to per-sample (multi-view) index ─────────────────────
print('\nBuilding multi-view index...')
train_df = build_multiview_index(train_df_raw, 'TRAIN')
val_df   = build_multiview_index(val_df_raw,   'VALID')
test_df  = build_multiview_index(test_df_raw,  'TEST')

# ── STEP 3: Remove corrupted ──────────────────────────────────────────────
print('\nChecking for corrupted files...')
train_df = remove_corrupted_mv(train_df, 'TRAIN')
val_df   = remove_corrupted_mv(val_df,   'VALID')
test_df  = remove_corrupted_mv(test_df,  'TEST')

# ── STEP 4: Exclude exercises ─────────────────────────────────────────────
print('\nExcluding exercises...')
train_df = exclude_exercises(train_df, label='TRAIN')
val_df   = exclude_exercises(val_df,   label='VALID')
test_df  = exclude_exercises(test_df,  label='TEST')

# ── STEP 5: Remap exercise IDs ────────────────────────────────────────────
remaining_exercises = sorted(
    set(train_df['exercise'].unique()) |
    set(val_df['exercise'].unique())   |
    set(test_df['exercise'].unique())
)
EXERCISE_REMAP = {orig: new for new, orig in enumerate(remaining_exercises)}

print(f'\n  Exercise ID remap : {EXERCISE_REMAP}')

for df_ in [train_df, val_df, test_df]:
    df_['exercise'] = df_['exercise'].map(EXERCISE_REMAP)

# ── STEP 6: Leakage check ─────────────────────────────────────────────────
tr_keys = set(train_df['trial_key'])
vl_keys = set(val_df['trial_key'])
te_keys = set(test_df['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

# ── Summary ───────────────────────────────────────────────────────────────
print(f'\n{"═"*70}')
print(f'  {"Split":<8} {"Samples":>8} {"Correct":>9} {"Erroneous":>11} '
      f'{"Q mean":>8} {"Q std":>7} {"Q min":>7} {"Q max":>7}')
print(f'  {"─"*68}')
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    cor = (d['trial_num'] <= 2).sum()
    err = (d['trial_num'] >= 3).sum()
    q   = d['quality']
    print(f'  {name:<8} {len(d):>8} {cor:>9} {err:>11} '
          f'{q.mean():>8.3f} {q.std():>7.3f} '
          f'{q.min():>7.3f} {q.max():>7.3f}')
print(f'{"═"*70}')
print('\n✓ Multi-view index ready  →  train / val / test DataFrames built')


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Skeleton visualisation helpers
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
# Cell 10 — BZUMultiViewDataset  (LATE FUSION)
#
#   Identical to mid-fusion dataset: returns 3 separate (T, J, 6) tensors.
#   The difference is entirely in the MODEL — each camera's backbone now
#   outputs a SCORE (scalar) rather than a feature vector, and those
#   scores are fused at the very end.
# ══════════════════════════════════════════════════════════════════════════

class BZUMultiViewDataset(Dataset):
    """
    Returns (cam_skels, quality_score, exercise_id) where:
      cam_skels : list of 3 tensors, each (T, J, 6)  — one per camera
    """
    NUM_CAMERAS = len(ALL_CAMERAS)

    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        skels = []
        for cam in ALL_CAMERAS:
            skel = load_skeleton(row[f'filepath_c{cam}'])
            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3),
                                dtype=np.float32)
            skel = self._normalise_length(skel)
            skels.append(skel)

        if self.augment:
            skels = self._augment_multiview(skels)

        cam_tensors = []
        for skel in skels:
            velocity     = np.zeros_like(skel)
            velocity[1:] = skel[1:] - skel[:-1]
            feat = np.concatenate([skel, velocity], axis=-1)   # (T, J, 6)
            cam_tensors.append(torch.tensor(feat, dtype=torch.float32))

        quality     = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)
        return cam_tensors, quality, exercise_id

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

    def _augment_multiview(self, skels):
        T = skels[0].shape[0]
        speed = np.random.uniform(0.75, 1.25)
        n_new = max(10, int(T * speed))
        idxs  = np.linspace(0, T - 1, n_new).astype(int)
        skels = [self._normalise_length(s[idxs]) for s in skels]

        keep_ratio = np.random.uniform(0.80, 1.0)
        n_keep     = max(10, int(self.target_frames * keep_ratio))
        keep_idxs  = np.sort(
            np.random.choice(self.target_frames, n_keep, replace=False))
        skels = [self._normalise_length(s[keep_idxs]) for s in skels]
        return skels


def late_fusion_collate(batch):
    """
    Same as mid_fusion_collate — cameras stay as separate tensors.
    Each camera score is computed independently before fusion.
    """
    n_cams       = len(batch[0][0])
    cam_batch    = [torch.stack([item[0][c] for item in batch]) for c in range(n_cams)]
    qualities    = torch.stack([item[1] for item in batch])
    exercise_ids = torch.stack([item[2] for item in batch])
    return cam_batch, qualities, exercise_ids


print('✓ BZUMultiViewDataset defined  '
      '(Late Fusion — 3 separate (T,J,6) tensors per sample)')
print('✓ late_fusion_collate defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN backbone + LATE FUSION Model
#
#   Architecture overview
#   ─────────────────────
#
#   C0 skeleton (T,J,6) ──► STGCNCameraHead ──► score_c0  (B, 1) ─┐
#   C1 skeleton (T,J,6) ──► STGCNCameraHead ──► score_c1  (B, 1) ─┤ fuse
#   C2 skeleton (T,J,6) ──► STGCNCameraHead ──► score_c2  (B, 1) ─┘
#                                                     │
#                              LateFusionAggregator   │
#                              ┌─────────────────────┤
#                              │  'mean'    → average │
#                              │  'weighted'→ learned │
#                              │             softmax  │
#                              │             weights  │
#                              │  'mlp'     → 3-score │
#                              │             MLP head │
#                              └─────────────────────┘
#                                           │
#                                  final quality score (B,)
#
#   KEY DIFFERENCE from mid-fusion:
#     Mid  → each backbone produces a FEATURE VECTOR, fusion in feature space
#     Late → each backbone produces a SCORE (scalar), fusion in score space
#
#   Each camera head also receives the exercise embedding so that
#   per-camera predictions are exercise-aware.
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
    A    = np.zeros((3, num_joints, num_joints), dtype=np.float32)

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
        x     = self.conv(x)
        x     = x.view(B, self.K, -1, T, J)
        A_eff = A + self.M
        out   = torch.einsum('bkctj,kjv->bctv', x, A_eff)
        out   = self.bn(out)
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


NUM_EXERCISES = len(EXERCISE_REMAP)


# ────────────────────────────────────────────────────────────────────────
# STGCNCameraHead
#
#   Full ST-GCN (9 blocks) for ONE camera.
#   Input  : (B, T, J, 6)
#   Output : (B,)  — a quality score ∈ [1, 5] for that single camera
#
#   This is the "deepest" possible late-fusion variant: each camera
#   produces its own complete prediction before anything is shared.
# ────────────────────────────────────────────────────────────────────────

class STGCNCameraHead(nn.Module):
    """
    Single-camera ST-GCN with its own regression head.
    Produces one quality score per sample.
    """
    def __init__(self, in_features=6, K=3, dropout=0.5):
        super().__init__()

        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_features * NUM_JOINTS)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_features, 64,  K=K, residual=False, dropout=dropout),
            STGCNBlock(64,  64,          K=K,                 dropout=dropout),
            STGCNBlock(64,  64,          K=K,                 dropout=dropout),
            STGCNBlock(64,  128, K=K, stride=2,               dropout=dropout),
            STGCNBlock(128, 128,         K=K,                 dropout=dropout),
            STGCNBlock(128, 128,         K=K,                 dropout=dropout),
            STGCNBlock(128, 256, K=K, stride=2,               dropout=dropout),
            STGCNBlock(256, 256,         K=K,                 dropout=dropout),
            STGCNBlock(256, 256,         K=K,                 dropout=dropout),
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ex_embed = nn.Embedding(NUM_EXERCISES, 32)

        # Per-camera regression head → single score
        self.reg_head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, exercise_id):
        """
        x           : (B, T, J, 6)
        exercise_id : (B,)
        returns     : (B,)   — score ∈ [1, 5]
        """
        B, T, J, C = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, C * J, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, J, T).permute(0, 1, 3, 2)   # (B, C, T, J)

        for block in self.blocks:
            x = block(x, self.A)

        x   = self.gap(x).squeeze(-1).squeeze(-1)          # (B, 256)
        ex  = self.ex_embed(exercise_id)                    # (B, 32)
        h   = torch.cat([x, ex], dim=1)                     # (B, 288)
        return 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))   # (B,)


# ────────────────────────────────────────────────────────────────────────
# LateFusionAggregator
#
#   Takes 3 per-camera scores (B,) and fuses them into one final score.
#
#   mode = 'mean'      → simple average, zero extra parameters
#   mode = 'weighted'  → 3 learned scalars, softmax-normalised so they
#                        sum to 1 (the network learns which camera to trust)
#   mode = 'mlp'       → tiny 2-layer MLP; can learn non-linear combos
#                        e.g. "if cam0 and cam1 disagree, trust cam2"
# ────────────────────────────────────────────────────────────────────────

class LateFusionAggregator(nn.Module):
    def __init__(self, n_cams=3, mode=LATE_FUSION_MODE):
        super().__init__()
        self.mode   = mode
        self.n_cams = n_cams

        if mode == 'weighted':
            # One raw scalar per camera; softmax gives normalised weights
            self.raw_weights = nn.Parameter(torch.ones(n_cams))

        elif mode == 'mlp':
            # Input: n_cams scores  →  Output: 1 fused score (in raw space)
            # The final tanh + scaling is applied in STGCN_LateFusion.forward
            self.mlp = nn.Sequential(
                nn.Linear(n_cams, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            # initialise to near-mean behaviour
            nn.init.constant_(self.mlp[0].weight, 1.0 / n_cams)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.ones_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, cam_scores):
        """
        cam_scores : (B, n_cams)  — one score per camera per sample
        returns    : (B,)         — fused score
        """
        if self.mode == 'mean':
            return cam_scores.mean(dim=1)

        elif self.mode == 'weighted':
            w = F.softmax(self.raw_weights, dim=0)         # (n_cams,)  sums to 1
            return (cam_scores * w.unsqueeze(0)).sum(dim=1)

        elif self.mode == 'mlp':
            # MLP operates on the raw per-camera values (already in [1,5])
            return self.mlp(cam_scores).squeeze(1)

        else:
            raise ValueError(f"Unknown LATE_FUSION_MODE: {self.mode!r}")

    def extra_repr(self):
        if self.mode == 'weighted':
            w = F.softmax(self.raw_weights.detach(), dim=0).numpy()
            return f"mode=weighted  weights={np.round(w, 3)}"
        return f"mode={self.mode}"


# ────────────────────────────────────────────────────────────────────────
# STGCN_LateFusion  — the full model
# ────────────────────────────────────────────────────────────────────────

class STGCN_LateFusion(nn.Module):
    """
    ST-GCN with LATE FUSION of 3 camera views.

    Forward inputs:
      cam_skels   : list of 3 tensors, each (B, T, J, 6)
      exercise_id : (B,)

    Forward output:
      fused_score : (B,)  ∈ [1, 5]
      cam_scores  : (B, 3)  — individual per-camera scores (for aux loss)

    Each camera has its own full ST-GCN + regression head that produces
    a complete quality prediction.  The LateFusionAggregator then
    combines those 3 scalar predictions into the final score.
    """
    def __init__(self,
                 K              = 3,
                 dropout        = 0.5,
                 fusion_mode    = LATE_FUSION_MODE,
                 shared_backbone= False):
        super().__init__()

        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.camera_head = STGCNCameraHead(K=K, dropout=dropout)
        else:
            self.camera_heads = nn.ModuleList([
                STGCNCameraHead(K=K, dropout=dropout)
                for _ in ALL_CAMERAS
            ])

        self.aggregator = LateFusionAggregator(
            n_cams=len(ALL_CAMERAS), mode=fusion_mode)

    def forward(self, cam_skels, exercise_id):
        """
        Returns fused_score (B,) and per-camera scores (B, n_cams).
        Per-camera scores are used for the auxiliary loss during training.
        """
        per_cam_scores = []
        for i, skel in enumerate(cam_skels):
            if self.shared_backbone:
                score = self.camera_head(skel, exercise_id)
            else:
                score = self.camera_heads[i](skel, exercise_id)
            per_cam_scores.append(score)                   # each (B,)

        # Stack → (B, n_cams)
        cam_scores_tensor = torch.stack(per_cam_scores, dim=1)

        # ── LATE FUSION: combine scalar scores ────────────────────────────
        fused = self.aggregator(cam_scores_tensor)         # (B,)

        # Clamp fused output to [1, 5] regardless of fusion mode
        fused = fused.clamp(1.0, 5.0)

        return fused, cam_scores_tensor


# ── Sanity check ──────────────────────────────────────────────────────────
_dummy_cams = [torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, 6) for _ in ALL_CAMERAS]
_dummy_ex   = torch.zeros(2, dtype=torch.long)
_model      = STGCN_LateFusion()
_fused, _cam_scores = _model(_dummy_cams, _dummy_ex)
assert _fused.shape      == (2,),   f"Expected (2,),    got {_fused.shape}"
assert _cam_scores.shape == (2, 3), f"Expected (2, 3),  got {_cam_scores.shape}"
print(f'\n✓ Late-Fusion ST-GCN sanity check passed')
print(f'  fused shape      : {_fused.shape}')
print(f'  cam_scores shape : {_cam_scores.shape}')

total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters: {total_params:,}')
del _dummy_cams, _dummy_ex, _model, _fused, _cam_scores

print(f'✓ STGCN_LateFusion defined  '
      f'(3 × STGCNCameraHead → scores → LateFusionAggregator[{LATE_FUSION_MODE}] → score)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

# ── Auxiliary loss weight ─────────────────────────────────────────────────
# Each camera head is supervised directly with a fraction of the main loss.
# This prevents the per-camera heads from collapsing and forces each one
# to learn a meaningful individual prediction.
AUX_LOSS_WEIGHT = 0.3   # final_loss = main_loss + 0.3 × mean(cam_losses)


def centre_and_scale_single(x):
    """
    x : (B, T, J, 6)   — pos(3) + vel(3) for one camera
    Returns normalised (B, T, J, 6).
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


def centre_and_scale_multiview(cam_skels):
    return [centre_and_scale_single(s) for s in cam_skels]


def run_epoch(model, loader, reg_fn, is_train=True, optimiser=None):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    # Track per-camera MAE for diagnostics
    cam_mae_accum = np.zeros(len(ALL_CAMERAS))
    n_batches     = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for cam_skels, qualities, exercise_ids in loader:
            cam_skels    = [s.to(DEVICE) for s in cam_skels]
            cam_skels    = centre_and_scale_multiview(cam_skels)
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)

            fused_scores, cam_scores = model(cam_skels, exercise_ids)

            # ── Main loss: fused prediction vs ground truth ───────────────
            main_loss = reg_fn(fused_scores, qualities)

            # ── Auxiliary loss: each camera head supervised individually ──
            aux_loss = torch.stack([
                reg_fn(cam_scores[:, i], qualities)
                for i in range(len(ALL_CAMERAS))
            ]).mean()

            loss = main_loss + AUX_LOSS_WEIGHT * aux_loss

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            total_loss += main_loss.item()   # log main loss only
            q_true.extend(qualities.cpu().numpy())
            q_pred.extend(fused_scores.detach().cpu().numpy())

            # Accumulate per-camera MAE
            for i in range(len(ALL_CAMERAS)):
                cam_mae_accum[i] += float(
                    (cam_scores[:, i].detach().cpu() - qualities.cpu()).abs().mean())
            n_batches += 1

    n  = max(1, len(loader))
    qt = np.array(q_true)
    qp = np.array(q_pred)
    pcc = float(pearsonr(qt, qp)[0]) if len(qt) > 1 else 0.0

    result = {
        'loss'   : total_loss / n,
        'rmse'   : float(np.sqrt(np.mean((qt - qp) ** 2))),
        'mae'    : float(np.mean(np.abs(qt - qp))),
        'r2'     : float(r2_score(qt, qp)) if len(qt) > 1 else 0.0,
        'pcc'    : pcc,
    }
    # Per-camera MAE (for monitoring)
    for i in range(len(ALL_CAMERAS)):
        result[f'cam{i}_mae'] = cam_mae_accum[i] / max(1, n_batches)

    return result

print('✓ centre_and_scale_multiview (per-camera) and run_epoch defined')
print(f'✓ AUX_LOSS_WEIGHT = {AUX_LOSS_WEIGHT}  '
      f'(each camera head also supervised directly)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13.5 — Split quality distribution audit
# ══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Quality score distribution audit")
print("=" * 65)

splits      = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
trial_types = [('Correct (T≤2)', lambda d: d[d['trial_num'] <= 2]),
               ('Erroneous (T≥3)', lambda d: d[d['trial_num'] >= 3])]

for split_name, split_df in splits:
    for type_name, filter_fn in trial_types:
        sub = filter_fn(split_df)
        if len(sub) == 0:
            continue
        q = sub['quality']
        print(f"{split_name:<6} {type_name:<16}  "
              f"n={len(sub):4d}  mean={q.mean():.3f}  std={q.std():.3f}  "
              f"[{q.min():.2f}, {q.max():.2f}]")
    print()

print("Person distribution:")
for split_name, split_df in splits:
    print(f"  {split_name}: {sorted(split_df['person'].unique())}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers
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
    ax.set_title(f'Regression Loss (SmoothL1) — Late Fusion [{LATE_FUSION_MODE}]',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curve.png'))


def plot_rmse_mae(history, save_dir, test_rmse=None, test_mae=None):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'RMSE & MAE — Train / Val / Test  (Late Fusion [{LATE_FUSION_MODE}])',
                 fontsize=14, fontweight='bold')
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
    ax.set_title(f'R² Score — Late Fusion [{LATE_FUSION_MODE}]',
                 fontsize=13, fontweight='bold')
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
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1)
    ax.set_title(f'Pearson Correlation Coefficient — Late Fusion [{LATE_FUSION_MODE}]',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('PCC')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'pcc_curve.png'))


def plot_per_camera_mae(history, save_dir):
    """Plot how each camera's individual MAE evolves — unique to late fusion."""
    epochs = range(1, len(history['train_mae']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Per-Camera MAE over Training (Late Fusion)',
                 fontsize=13, fontweight='bold')
    colors = ['steelblue', 'darkorange', 'seagreen']
    for ax, split in zip(axes, ['train', 'val']):
        for i, color in enumerate(colors[:len(ALL_CAMERAS)]):
            key = f'{split}_cam{i}_mae'
            if key in history:
                ax.plot(epochs, history[key], label=f'Camera {i}', color=color)
        ax.plot(epochs, history[f'{split}_mae'], 'k--',
                linewidth=2, label='Fused (final)')
        ax.set_title(f'{split.capitalize()} split')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'per_camera_mae.png'))


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
    ax.set_title(
        f'{split_name} Set — True vs Predicted Quality\n'
        f'Late Fusion [{LATE_FUSION_MODE}]',
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


def plot_camera_score_comparison(all_true_q, all_pred_q, all_cam_scores,
                                 save_dir):
    """
    Scatter grid: one panel per camera showing that camera's raw
    prediction vs ground truth, plus the final fused prediction.
    Unique to late fusion — lets you see which camera is most accurate.
    """
    qt       = np.array(all_true_q)
    qp_fused = np.array(all_pred_q)
    n_cams   = all_cam_scores.shape[1]

    fig, axes = plt.subplots(1, n_cams + 1,
                             figsize=(5.5 * (n_cams + 1), 5))
    fig.suptitle('Per-Camera Scores vs Fused Score (Test Set)',
                 fontsize=14, fontweight='bold')

    def _scatter(ax, true, pred, title):
        mae  = float(np.mean(np.abs(true - pred)))
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        r2   = float(r2_score(true, pred)) if len(true) > 1 else float('nan')
        ax.scatter(true, pred, alpha=0.55, edgecolors='black',
                   linewidths=0.3, s=45, color='steelblue')
        lo = min(true.min(), pred.min()) - 0.2
        hi = max(true.max(), pred.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('True Quality'); ax.set_ylabel('Predicted Quality')
        ax.grid(alpha=0.3)
        ax.text(0.05, 0.97,
                f'MAE={mae:.3f}\nRMSE={rmse:.3f}\nR²={r2:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    for i in range(n_cams):
        _scatter(axes[i], qt, all_cam_scores[:, i], f'Camera {i}')
    _scatter(axes[n_cams], qt, qp_fused,
             f'Fused [{LATE_FUSION_MODE}]')

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'camera_score_comparison.png'))


def plot_early_stop(history, stopped_epoch, best_epoch, save_dir):
    epochs = range(1, len(history['val_mae']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_mae'], label='Train MAE', color='steelblue')
    ax.plot(epochs, history['val_mae'],   label='Val MAE',   color='darkorange')
    ax.axvline(best_epoch,    color='purple', linestyle=':',  linewidth=2,
               label=f'Best epoch ({best_epoch})')
    ax.axvline(stopped_epoch, color='red',    linestyle='--', linewidth=2,
               label=f'Early stop ({stopped_epoch})')
    ax.set_title(f'MAE + Early Stopping — Late Fusion [{LATE_FUSION_MODE}]',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))

print('✓ Plotting helpers defined  (includes per-camera diagnostic plots)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Early Stopping
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
# Cell 16 — Training Loop
# ══════════════════════════════════════════════════════════════════════════

reg_fn = nn.SmoothL1Loss(beta=1.0)


def make_weighted_sampler(df):
    q = df['quality'].values.astype(np.float32)
    raw_weights      = (q.max() + 1.0) - q
    erroneous_mask   = (df['trial_num'].values >= 3).astype(np.float32)
    raw_weights     *= (1.0 + erroneous_mask)
    weights_tensor   = torch.DoubleTensor(raw_weights)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights_tensor, num_samples=len(df), replacement=True)


train_sampler = make_weighted_sampler(train_df)

train_loader = DataLoader(
    BZUMultiViewDataset(train_df, augment=True),
    batch_size   = BATCH_SIZE,
    sampler      = train_sampler,
    num_workers  = 0,
    pin_memory   = (DEVICE == 'cuda'),
    collate_fn   = late_fusion_collate,
)
val_loader  = DataLoader(
    BZUMultiViewDataset(val_df,  augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE == 'cuda'),
    collate_fn=late_fusion_collate,
)
test_loader = DataLoader(
    BZUMultiViewDataset(test_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE == 'cuda'),
    collate_fn=late_fusion_collate,
)

model = STGCN_LateFusion(
    K            = 3,
    dropout      = 0.5,
    fusion_mode  = LATE_FUSION_MODE,
).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=50, min_lr=1e-6, verbose=True)

early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val']
METRICS = ['loss', 'rmse', 'mae', 'r2', 'pcc'] + \
          [f'cam{i}_mae' for i in range(len(ALL_CAMERAS))]
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info('STARTING LATE-FUSION REGRESSION TRAINING')
log.info('=' * 70)
log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)} '
         f'(multi-view samples, 3 cameras — late fusion)')
log.info(f'LATE_FUSION_MODE={LATE_FUSION_MODE}  AUX_LOSS_WEIGHT={AUX_LOSS_WEIGHT}')

print(f'\n{"═"*70}')
print(f'  Late Fusion  |  Mode: {LATE_FUSION_MODE}  |  Cameras: {ALL_CAMERAS}')
print(f'  Aux loss weight: {AUX_LOSS_WEIGHT}  (per-camera supervision)')
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Patience: {PATIENCE}  |  LR: {LR}  |  Batch: {BATCH_SIZE}')
print(f'{"═"*70}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, reg_fn, is_train=True,  optimiser=optimiser)
    vl = run_epoch(model, val_loader,   reg_fn, is_train=False)
    scheduler.step(vl['mae'])

    for split, res in [('train', tr), ('val', vl)]:
        for m in METRICS:
            if m in res:
                history[f'{split}_{m}'].append(res[m])

    stop, improved = early_stop.step(vl['mae'], model, epoch)

    # Build per-camera MAE string for log
    cam_str = '  '.join(
        f'C{i}={tr[f"cam{i}_mae"]:.3f}' for i in range(len(ALL_CAMERAS)))

    star = ' ★' if improved else ''
    msg = (f'  Ep {epoch:3d}/{EPOCHS} | '
           f'Tr loss={tr["loss"]:.4f} mae={tr["mae"]:.3f} '
           f'r2={tr["r2"]:.3f} pcc={tr["pcc"]:.3f} | '
           f'Vl loss={vl["loss"]:.4f} mae={vl["mae"]:.3f} '
           f'r2={vl["r2"]:.3f} pcc={vl["pcc"]:.3f} | '
           f'ES {early_stop.counter}/{PATIENCE}{star}')
    cam_msg = f'         Tr cam MAE → {cam_str}'
    print(msg)
    print(cam_msg)
    log.info(msg)

    if improved:
        print(f'    ★ val_mae={early_stop.best_mae:.4f}  '
              f'rmse={vl["rmse"]:.4f}  r2={vl["r2"]:.4f}  pcc={vl["pcc"]:.4f}')

    # Print learned fusion weights (weighted mode only)
    if LATE_FUSION_MODE == 'weighted' and epoch % 20 == 0:
        w = F.softmax(model.aggregator.raw_weights.detach().cpu(), dim=0).numpy()
        print(f'         Fusion weights → '
              f'C0={w[0]:.3f}  C1={w[1]:.3f}  C2={w[2]:.3f}')

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

# ── Print final learned weights if applicable ─────────────────────────────
if LATE_FUSION_MODE == 'weighted':
    w = F.softmax(model.aggregator.raw_weights.detach().cpu(), dim=0).numpy()
    print(f'\n  Learned fusion weights (best epoch):')
    for i, wi in enumerate(w):
        print(f'    Camera {i} : {wi:.4f}')

# ── Final test evaluation ─────────────────────────────────────────────────
final_te = run_epoch(model, test_loader, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')
print(f'  PCC  : {final_te["pcc"]:.4f}')
for i in range(len(ALL_CAMERAS)):
    print(f'  Camera {i} MAE : {final_te[f"cam{i}_mae"]:.4f}')


# ── Collect predictions (fused + per-camera) ──────────────────────────────
def collect_predictions(loader):
    model.eval()
    all_true, all_fused, all_cam, all_ex = [], [], [], []
    with torch.no_grad():
        for cam_skels, qualities, exercise_ids in loader:
            cam_skels    = [s.to(DEVICE) for s in cam_skels]
            cam_skels    = centre_and_scale_multiview(cam_skels)
            exercise_ids = exercise_ids.to(DEVICE)
            fused, cam_scores = model(cam_skels, exercise_ids)
            all_true.extend(qualities.numpy())
            all_fused.extend(fused.cpu().numpy())
            all_cam.append(cam_scores.cpu().numpy())
            all_ex.extend(exercise_ids.cpu().numpy())
    return (np.array(all_true),
            np.array(all_fused),
            np.concatenate(all_cam, axis=0),
            np.array(all_ex))

all_true_q,     all_pred_q,     all_cam_scores,  all_exercise_ids = collect_predictions(test_loader)
all_true_train, all_pred_train, all_cam_train,   _                = collect_predictions(train_loader)
all_true_val,   all_pred_val,   all_cam_val,     _                = collect_predictions(val_loader)

# ── Save plots ────────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR,  test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR,     test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR,           test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR,          test_pcc=final_te['pcc'])
plot_per_camera_mae(history, PLOTS_DIR)
plot_regression_scatter(all_true_q,     all_pred_q,     split_name='Test',  save_dir=PLOTS_DIR)
plot_regression_scatter(all_true_train, all_pred_train, split_name='Train', save_dir=PLOTS_DIR)
plot_regression_scatter(all_true_val,   all_pred_val,   split_name='Val',   save_dir=PLOTS_DIR)
plot_camera_score_comparison(all_true_q, all_pred_q, all_cam_scores, PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

# ── Save history & predictions ────────────────────────────────────────────
json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump({k: v for k, v in history.items()}, f, indent=2)
print(f'  ✓ History → {json_path}')

np.savez(os.path.join(LOGS_DIR, 'training_history.npz'),
         **{k: np.array(v) for k, v in history.items()})
np.savez(os.path.join(LOGS_DIR, 'test_predictions.npz'),
         q_true       = all_true_q,
         q_pred_fused = all_pred_q,
         cam_scores   = all_cam_scores,
         exercise_ids = all_exercise_ids)
np.savez(os.path.join(LOGS_DIR, 'train_predictions.npz'),
         q_true=all_true_train, q_pred=all_pred_train, cam_scores=all_cam_train)
np.savez(os.path.join(LOGS_DIR, 'val_predictions.npz'),
         q_true=all_true_val,   q_pred=all_pred_val,   cam_scores=all_cam_val)
print('  ✓ NPZ files saved')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16.5 — Per-exercise test metrics
# ══════════════════════════════════════════════════════════════════════════

unique_exercises = sorted(np.unique(all_exercise_ids))
per_ex_results   = {}

print('=' * 72)
print(f'  {"Exercise":<12} {"n":>5} {"MAE":>8} {"RMSE":>8} {"R²":>8} {"PCC":>8}')
print('─' * 72)

for ex_id in unique_exercises:
    mask = all_exercise_ids == ex_id
    qt   = all_true_q[mask]
    qp   = all_pred_q[mask]
    n    = mask.sum()
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp) ** 2)))
    r2   = float(r2_score(qt, qp))    if n > 1 else float('nan')
    pcc  = float(pearsonr(qt, qp)[0]) if n > 1 else float('nan')
    per_ex_results[ex_id] = dict(n=n, mae=mae, rmse=rmse, r2=r2, pcc=pcc)
    print(f'  E{ex_id:<11} {n:>5} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pcc:>8.4f}')

print('─' * 72)
print(f'  {"Overall":<12} {len(all_true_q):>5} '
      f'{final_te["mae"]:>8.4f} {final_te["rmse"]:>8.4f} '
      f'{final_te["r2"]:>8.4f} {final_te["pcc"]:>8.4f}')
print('=' * 72)

per_ex_df = pd.DataFrame([
    {'exercise': f'E{ex}', **vals} for ex, vals in per_ex_results.items()])
per_ex_csv = os.path.join(LOGS_DIR, 'per_exercise_metrics.csv')
per_ex_df.to_csv(per_ex_csv, index=False)
print(f'\n  ✓ Per-exercise CSV → {per_ex_csv}')

# ── Per-exercise scatter grid ─────────────────────────────────────────────
n_ex   = len(unique_exercises)
n_cols = 3
n_rows = int(np.ceil(n_ex / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle(f'Per-Exercise: True vs Predicted Quality — Test Set '
             f'(Late Fusion [{LATE_FUSION_MODE}])',
             fontsize=14, fontweight='bold')

for i, ex_id in enumerate(unique_exercises):
    ax   = axes[i]
    mask = all_exercise_ids == ex_id
    qt   = all_true_q[mask]
    qp   = all_pred_q[mask]
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
    ax.text(0.05, 0.97,
            f'MAE={res["mae"]:.3f}\nRMSE={res["rmse"]:.3f}\n'
            f'R²={res["r2"]:.3f}\nPCC={res["pcc"]:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
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
fig.suptitle(f'Per-Exercise Test Metrics (Late Fusion [{LATE_FUSION_MODE}])',
             fontsize=13, fontweight='bold')
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
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Exercise')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.04,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

for ax, metric in zip(axes, ['mae', 'rmse', 'r2', 'pcc']):
    ax.axhline(final_te[metric], color='red', linestyle='--',
               linewidth=1.5, label=f'Overall={final_te[metric]:.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
bar_path = os.path.join(PLOTS_DIR, 'per_exercise_bar.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise bar chart → {bar_path}')

np.savez(os.path.join(LOGS_DIR, 'per_exercise_metrics.npz'),
         exercise_ids = np.array(unique_exercises),
         n            = np.array([per_ex_results[e]['n']    for e in unique_exercises]),
         mae          = np.array([per_ex_results[e]['mae']  for e in unique_exercises]),
         rmse         = np.array([per_ex_results[e]['rmse'] for e in unique_exercises]),
         r2           = np.array([per_ex_results[e]['r2']   for e in unique_exercises]),
         pcc          = np.array([per_ex_results[e]['pcc']  for e in unique_exercises]),
)
print(f'  ✓ Per-exercise metrics NPZ saved')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv            = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 65)
print('  TRAINING SUMMARY — ST-GCN Late Fusion (3 cameras)')
print('=' * 65)
print(f'  Cameras          : {ALL_CAMERAS}')
print(f'  Fusion type      : LATE  (score-level)')
print(f'  Fusion mode      : {LATE_FUSION_MODE}')
print(f'  Aux loss weight  : {AUX_LOSS_WEIGHT}')
print(f'  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')
print(f'  Best Val PCC     : {best_val_pcc:.4f}')
print('─' * 65)
print(f'  Test MAE         : {final_te["mae"]:.4f}')
print(f'  Test RMSE        : {final_te["rmse"]:.4f}')
print(f'  Test R²          : {final_te["r2"]:.4f}')
print(f'  Test PCC         : {final_te["pcc"]:.4f}')
if LATE_FUSION_MODE == 'weighted':
    w = F.softmax(model.aggregator.raw_weights.detach().cpu(), dim=0).numpy()
    print(f'  Fusion weights   : ' +
          '  '.join(f'C{i}={w[i]:.3f}' for i in range(len(ALL_CAMERAS))))
print('=' * 65)

log.info(f'Best Epoch={best_epoch}  stopped_epoch={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f}  RMSE={final_te["rmse"]:.4f}  '
         f'R²={final_te["r2"]:.4f}  PCC={final_te["pcc"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')

summary_row = {
    'split'          : 'test_overall',
    'exercise'       : 'all',
    'fusion'         : 'late',
    'fusion_mode'    : LATE_FUSION_MODE,
    'aux_loss_weight': AUX_LOSS_WEIGHT,
    'best_epoch'     : best_epoch,
    'stopped_epoch'  : stopped_epoch,
    'val_mae'        : best_val_mae,
    'val_rmse'       : best_val_rmse,
    'val_r2'         : best_val_r2,
    'val_pcc'        : best_val_pcc,
    'test_mae'       : final_te['mae'],
    'test_rmse'      : final_te['rmse'],
    'test_r2'        : final_te['r2'],
    'test_pcc'       : final_te['pcc'],
}
if LATE_FUSION_MODE == 'weighted':
    w = F.softmax(model.aggregator.raw_weights.detach().cpu(), dim=0).numpy()
    for i, wi in enumerate(w):
        summary_row[f'cam{i}_weight'] = float(wi)

rows = [summary_row]
for ex, vals in per_ex_results.items():
    rows.append({'split': 'test_per_exercise', 'exercise': f'E{ex}',
                 'fusion': 'late', 'fusion_mode': LATE_FUSION_MODE,
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

final_tr = {
    'mae' : float(np.mean(np.abs(all_true_train - all_pred_train))),
    'rmse': float(np.sqrt(np.mean((all_true_train - all_pred_train) ** 2))),
    'r2'  : float(r2_score(all_true_train, all_pred_train)),
    'pcc' : float(pearsonr(all_true_train, all_pred_train)[0]),
}
final_vl = {
    'mae' : float(np.mean(np.abs(all_true_val - all_pred_val))),
    'rmse': float(np.sqrt(np.mean((all_true_val - all_pred_val) ** 2))),
    'r2'  : float(r2_score(all_true_val, all_pred_val)),
    'pcc' : float(pearsonr(all_true_val, all_pred_val)[0]),
}

print('=' * 60)
print('  FIT DIAGNOSIS (best weights)')
print('=' * 60)
print(f'  {"Metric":<8}  {"Train":>8}  {"Val":>8}  {"Test":>8}')
print('─' * 60)
for m in ['mae', 'rmse', 'r2', 'pcc']:
    print(f'  {m.upper():<8}  {final_tr[m]:>8.4f}  {final_vl[m]:>8.4f}  {final_te[m]:>8.4f}')
print('=' * 60)
if final_tr['mae'] < 0.15 and final_te['mae'] > 0.40:
    print('  ⚠  OVERFITTING likely')
elif final_tr['mae'] > 0.40 and final_te['mae'] > 0.40:
    print('  ⚠  UNDERFITTING likely')
else:
    print('  ✓  Fit looks reasonable')

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
