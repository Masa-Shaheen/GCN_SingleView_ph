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
EPOCHS        = 600
LR            = 1e-4
BATCH_SIZE    = 24   # reduced: each sample loads 3 cameras
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_SingleView_Regression_Results"

# ── Mid-Fusion specific ───────────────────────────────────────────────────
IN_FEATURES       = 6      # pos(3) + vel(3) per camera
ENCODER_DIM       = 64     # each camera's raw 6 features → 64 learned features
FUSED_IN_FEATURES = 64     # what the shared ST-GCN backbone receives per joint
#
# Mid-fusion flow:
#   C0: (B, T, J, 6) ──┐
#   C1: (B, T, J, 6) ──┤→ per-camera PointWise FC (6→64) → concat → (B,T,J,192)
#   C2: (B, T, J, 6) ──┘        ↓
#                         shared ST-GCN backbone (IN=64, 3 cameras stacked)
#                                ↓
#                         regression head → quality score
#
# NOTE: We treat the 3×64 = 192 channels as the joint feature dimension
# fed into the first ST-GCN block (in_features=192 for block[0]).

CAMERAS = [0, 1, 2]

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 300
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Exercise Filter ───────────────────────────────────────────────────────
EXCLUDED_EXERCISES = {3, 7, 9}
EXERCISE_REMAP     = {}

print('✓ Configuration loaded')
print(f'  DATASET_DIR      : {DATASET_DIR}')
print(f'  SPLIT_DIR        : {SPLIT_DIR}')
print(f'  FUSION           : MID (per-camera encoder {IN_FEATURES}→{ENCODER_DIM}, then concat → {ENCODER_DIM*3} channels)')
print(f'  EXISTS           : {os.path.exists(DATASET_DIR)}')
print(f'  SPLIT EXISTS     : {os.path.exists(SPLIT_DIR)}')


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

run_name = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
RUN_DIR  = os.path.join(OUT_DIR, run_name)
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
LOGS_DIR  = os.path.join(RUN_DIR, "logs")

for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Run directory:", RUN_DIR)

import random

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
# Cell 3 — Explore dataset folder
# ══════════════════════════════════════════════════════════════════════════

all_npz = sorted(glob.glob(os.path.join(DATASET_DIR, '**/*.npz'), recursive=True))
print(f'Total NPZ files found : {len(all_npz)}')
if all_npz:
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
                print(f'✓ CSV loaded ({enc})')
                break
        except Exception:
            pass
else:
    raise FileNotFoundError(f'CSV not found: {CSV_PATH}')

if df_csv is None:
    raise FileNotFoundError(f'CSV failed to load: {CSV_PATH}')

print(f'Columns: {df_csv.columns.tolist()}')
print(f'Shape  : {df_csv.shape}')
print(df_csv.to_string())


# ══════════════════════════════════════════════════════════════════════════
# Cell 4.5 — Person-level data audit
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Quality stats per person:")
print("=" * 60)
print(df_csv.groupby('person')['mean'].agg(['mean','std','min','max','count']).round(3))

print("\n" + "=" * 60)
print("Missing exercises per person:")
print("=" * 60)
all_exercises = sorted(df_csv['exercise'].unique())
for person in sorted(df_csv['person'].unique()):
    exercises = df_csv[df_csv['person'] == person]['exercise'].unique()
    missing   = [e for e in all_exercises if e not in exercises]
    print(f"{person}: {len(exercises)} exercises | missing={missing if missing else 'None'}")

print("\n" + "=" * 60)
print("Trials per person (correct vs erroneous):")
print("=" * 60)
for person in sorted(df_csv['person'].unique()):
    p_df      = df_csv[df_csv['person'] == person]
    correct   = p_df[p_df['trial'].isin(['T0','T1','T2'])]
    erroneous = p_df[~p_df['trial'].isin(['T0','T1','T2'])]
    print(f"{person}: correct={len(correct):3d} rows | "
          f"erroneous={len(erroneous):3d} rows | "
          f"quality mean={p_df['mean'].mean():.3f}")


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
print(f'✓ Logging to {log_file}')

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger("GCN-MidFusion")
log.info("=" * 70)
log.info("ST-GCN Mid-Fusion Multi-View Regression | BZU Physiotherapy Dataset")
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

JOINT_NAMES = [
    'Hip', 'R-Hip', 'R-Knee', 'R-Ankle',
    'L-Hip', 'L-Knee', 'L-Ankle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'L-Shoulder', 'L-Elbow', 'L-Wrist',
    'R-Shoulder', 'R-Elbow', 'R-Wrist',
]


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
# Cell 8 — Skeleton visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

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


def plot_skeleton_3d(skel, frame_idx=0, title='Skeleton Sanity Check', save_path=None):
    pts = skel[frame_idx]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    views = [
        (axes[0], x,  y,  'Front View  (X–Y)', 'X (left/right)', 'Y (up/down)',  False),
        (axes[1], z,  y,  'Side View   (Z–Y)', 'Z (depth)',      'Y (up/down)',  False),
        (axes[2], x, -z,  'Top View    (X–Z)', 'X (left/right)', '-Z (forward)', False),
    ]
    for ax, hx, hy, view_title, xlabel, ylabel, invert_y in views:
        for (i, j) in SKELETON_EDGES:
            ax.plot([hx[i], hx[j]], [hy[i], hy[j]], color='dimgray', lw=2, zorder=1)
        for part, idxs in JOINT_COLORS.items():
            ax.scatter(hx[idxs], hy[idxs], c=PART_COLOR[part], s=80, zorder=3,
                       edgecolors='black', linewidths=0.5, label=part)
        for j_idx in range(len(hx)):
            ax.annotate(str(j_idx), (hx[j_idx], hy[j_idx]),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=7, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                  alpha=0.6, edgecolor='none'))
        ax.set_title(view_title, fontweight='bold', fontsize=10)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        if invert_y:
            ax.invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Skeleton plot saved → {save_path}')
    plt.close()


def plot_skeleton_frames(skel, n_frames=5, title='Skeleton Motion', save_path=None):
    T    = skel.shape[0]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    for col, fi in enumerate(idxs):
        ax  = axes[col]
        pts = skel[fi]
        x, y = pts[:, 0], pts[:, 1]
        for (i, j) in SKELETON_EDGES:
            ax.plot([x[i], x[j]], [y[i], y[j]], color='dimgray', lw=2)
        for part, pidxs in JOINT_COLORS.items():
            ax.scatter(x[pidxs], y[pidxs], c=PART_COLOR[part],
                       s=60, edgecolors='black', linewidths=0.4, zorder=3)
        ax.set_title(f'Frame {fi}', fontsize=9)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Motion frames saved → {save_path}')
    plt.close()


print('✓ Skeleton visualisation helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build dataset index from pre-split directories
# ══════════════════════════════════════════════════════════════════════════

def build_index_from_split(split_name, df_csv, camera_id=None):
    split_path = os.path.join(SPLIT_DIR, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")
    all_files = sorted(glob.glob(os.path.join(split_path, '**/*.npz'), recursive=True))
    print(f'\n[{split_name.upper()}] NPZ files found: {len(all_files)}')

    df_csv = df_csv.copy()
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
    REQUIRED_CAMERAS = {0, 1, 2}
    coverage   = df.groupby('trial_key')['camera'].apply(set)
    complete   = coverage[coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))].index
    incomplete = coverage[~coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))]
    n_before = len(df)
    df = df[df['trial_key'].isin(complete)].reset_index(drop=True)
    n_skipped_files  = n_before - len(df)
    n_skipped_groups = len(incomplete)
    if n_skipped_groups:
        print(f'  [{label}] ⚠️  Skipped {n_skipped_files} files from {n_skipped_groups} incomplete groups')
    else:
        print(f'  [{label}] ✓ All groups have complete 3-camera coverage')
    return df


def remove_corrupted(df, label=''):
    bad = []
    for fpath in df['filepath']:
        if load_skeleton(fpath) is None:
            bad.append(fpath)
    if bad:
        print(f'  [{label}] Removing {len(bad)} corrupted file(s)')
        df = df[~df['filepath'].isin(bad)].reset_index(drop=True)
    print(f'  [{label}] Clean samples: {len(df)}')
    return df


def exclude_exercises(df, excluded=EXCLUDED_EXERCISES, label=''):
    before = len(df)
    df = df[~df['exercise'].isin(excluded)].reset_index(drop=True)
    print(f'  [{label}] Excluded E{sorted(excluded)} → {before - len(df)} dropped, {len(df)} remain')
    return df


# ── Build splits ──────────────────────────────────────────────────────────
train_df = build_index_from_split('train', df_csv)
val_df   = build_index_from_split('valid', df_csv)
test_df  = build_index_from_split('test',  df_csv)

print('\nFiltering camera completeness...')
train_df = filter_complete_camera_groups(train_df, 'TRAIN')
val_df   = filter_complete_camera_groups(val_df,   'VALID')
test_df  = filter_complete_camera_groups(test_df,  'TEST')

print('\nChecking for corrupted files...')
train_df = remove_corrupted(train_df, 'TRAIN')
val_df   = remove_corrupted(val_df,   'VALID')
test_df  = remove_corrupted(test_df,  'TEST')

print('\nExcluding exercises...')
train_df = exclude_exercises(train_df, label='TRAIN')
val_df   = exclude_exercises(val_df,   label='VALID')
test_df  = exclude_exercises(test_df,  label='TEST')

# ── Remap exercise IDs ────────────────────────────────────────────────────
remaining_exercises = sorted(
    set(train_df['exercise'].unique()) |
    set(val_df['exercise'].unique())   |
    set(test_df['exercise'].unique())
)
EXERCISE_REMAP = {orig: new for new, orig in enumerate(remaining_exercises)}
print(f'\n  Exercise ID remap: {EXERCISE_REMAP}')
for df_ in [train_df, val_df, test_df]:
    df_['exercise'] = df_['exercise'].map(EXERCISE_REMAP)

# ── Build multi-view trial index ──────────────────────────────────────────
# Each sample = one trial_key (all 3 cameras present).
# We need one row per (trial_key, segment) combination.

def build_multiview_index(df):
    """
    Collapse per-file rows into one row per (trial_key, segment).
    Only keeps segments where ALL 3 cameras are present.
    Groups by trial_key+segment, maps cameras, skips incomplete segments.
    """
    rows = []
    # Sort to ensure deterministic ordering
    df_sorted = df.sort_values(['trial_key', 'segment', 'camera']).reset_index(drop=True)
    
    for (trial_key, segment), grp in df_sorted.groupby(['trial_key', 'segment']):
        cam_map = {int(r['camera']): r['filepath'] for _, r in grp.iterrows()}
        
        # Strict check: all three cameras must be present for this segment
        if not (0 in cam_map and 1 in cam_map and 2 in cam_map):
            continue
            
        ref = grp.iloc[0]
        rows.append({
            'trial_key' : trial_key,
            'segment'   : int(segment),
            'exercise'  : ref['exercise'],
            'person'    : ref['person'],
            'trial_num' : int(ref['trial_num']),
            'trial_id'  : ref['trial_id'],
            'quality'   : float(ref['quality']),
            'path_c0'   : cam_map[0],
            'path_c1'   : cam_map[1],
            'path_c2'   : cam_map[2],
        })
    
    result = pd.DataFrame(rows).reset_index(drop=True)
    
    # Verification print
    print(f"  build_trial_df: {len(result)} samples across "
          f"{result['trial_key'].nunique()} trials")
    print(f"  Per-exercise counts:\n"
          f"{result.groupby('exercise').size().to_string()}")
    return result

train_mv = build_multiview_index(train_df)
val_mv   = build_multiview_index(val_df)
test_mv  = build_multiview_index(test_df)

# ── Leakage check ─────────────────────────────────────────────────────────
tr_keys = set(train_mv['trial_key'])
vl_keys = set(val_mv['trial_key'])
te_keys = set(test_mv['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

print(f'\n{"═"*68}')
print(f'  {"Split":<8} {"Samples":>8} {"Unique trials":>15} {"Q mean":>8} {"Q std":>7}')
print(f'  {"─"*66}')
for name, d in [('Train', train_mv), ('Val', val_mv), ('Test', test_mv)]:
    print(f'  {name:<8} {len(d):>8} {d["trial_key"].nunique():>15} '
          f'{d["quality"].mean():>8.3f} {d["quality"].std():>7.3f}')
print(f'{"═"*68}')
print('\n✓ Multi-view index ready')

# Rebuild a combined flat df_index for diagnostics (all cameras, all splits)
df_index = pd.concat([train_df, val_df, test_df], ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════
# Cell 7.5 — Camera Distribution Check
# ══════════════════════════════════════════════════════════════════════════

print(df_index['camera'].value_counts().sort_index())
print(f"\nالكاميرات الموجودة: {sorted(df_index['camera'].unique())}")
print(df_index.groupby(['exercise', 'camera']).size().unstack(fill_value=0))


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.6 — Trials per Exercise Check
# ══════════════════════════════════════════════════════════════════════════

for ex_id, ex_df in df_index.groupby('exercise'):
    correct   = sorted(ex_df[ex_df['trial_num'] <= 2]['trial_id'].unique())
    erroneous = sorted(ex_df[ex_df['trial_num'] >= 3]['trial_id'].unique())
    print(f"E{ex_id}: correct={len(correct)} trials, erroneous={len(erroneous)} trials")


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.6.5 — Mean & Variance per Exercise per Split
# ══════════════════════════════════════════════════════════════════════════

for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"\n{'='*50}")
    print(f"  {split_name}")
    print(f"{'='*50}")
    print(split_df.groupby('exercise')['quality']
          .agg(['mean', 'var', 'count'])
          .round(4)
          .to_string())

print(f"\n{'='*60}")
print("  All Splits Combined")
print(f"{'='*60}")
df_all = pd.concat([
    train_df.assign(split='Train'),
    val_df.assign(split='Val'),
    test_df.assign(split='Test')
])
print(df_all.groupby(['split', 'exercise'])['quality']
      .agg(['mean', 'var', 'count'])
      .round(4)
      .to_string())


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.7 — Frame length distribution
# ══════════════════════════════════════════════════════════════════════════

lengths = []
sample_files = df_index['filepath'].sample(min(500, len(df_index)), random_state=42)

for fpath in sample_files:
    skel = load_skeleton(fpath)
    if skel is not None:
        lengths.append(skel.shape[0])

lengths = np.array(lengths)
print(f"Frame length distribution (sample of {len(lengths)} files):")
print(f"  min    = {lengths.min()}")
print(f"  max    = {lengths.max()}")
print(f"  mean   = {lengths.mean():.1f}")
print(f"  median = {np.median(lengths):.1f}")
print(f"  std    = {lengths.std():.1f}")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p:3d}th = {np.percentile(lengths, p):.0f}")

print(f"\nValue counts (top 10):")
unique, counts = np.unique(lengths, return_counts=True)
top10 = sorted(zip(counts, unique), reverse=True)[:10]
for cnt, val in top10:
    print(f"  {int(val):4d} frames → {cnt:4d} files ({cnt/len(lengths)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — Visualise a sample skeleton
# ══════════════════════════════════════════════════════════════════════════

idx         = 10
sample_skel = load_skeleton(df_index.iloc[idx]['filepath'])

print(df_index.iloc[idx][['person', 'exercise', 'trial_id', 'segment', 'filepath']])
print(f'Skeleton shape : {sample_skel.shape}')
print(f'X range : [{sample_skel[:,:,0].min():.3f}, {sample_skel[:,:,0].max():.3f}]')
print(f'Y range : [{sample_skel[:,:,1].min():.3f}, {sample_skel[:,:,1].max():.3f}]')
print(f'Z range : [{sample_skel[:,:,2].min():.3f}, {sample_skel[:,:,2].max():.3f}]')

plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton · {df_index.iloc[idx]['trial_key']} · E{df_index.iloc[idx]['exercise']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
)
plot_skeleton_frames(
    sample_skel, n_frames=5,
    title=f"Motion Sequence · {df_index.iloc[idx]['trial_key']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_motion.png'),
)

print("Axis ranges across all frames:")
print(f"X: [{sample_skel[:,:,0].min():.3f}, {sample_skel[:,:,0].max():.3f}] — likely left/right")
print(f"Y: [{sample_skel[:,:,1].min():.3f}, {sample_skel[:,:,1].max():.3f}] — likely ???")
print(f"Z: [{sample_skel[:,:,2].min():.3f}, {sample_skel[:,:,2].max():.3f}] — likely height")

hip  = sample_skel[:, 0, :]
head = sample_skel[:, 10, :]
print("\nHip  mean XYZ:", hip.mean(axis=0))
print("Head mean XYZ:", head.mean(axis=0))
print("\nDifference (head - hip):", head.mean(axis=0) - hip.mean(axis=0))


# ══════════════════════════════════════════════════════════════════════════
# Cell 9.5 — Camera Angle Check
# ══════════════════════════════════════════════════════════════════════════

for cam in [0, 1, 2]:
    files = df_index[
        (df_index['person'] == 'P0') &
        (df_index['exercise'] == 0) &
        (df_index['camera'] == cam) &
        (df_index['trial_id'] == 'T0') &
        (df_index['segment'] == 0)
    ]
    if len(files) > 0:
        skel = load_skeleton(files.iloc[0]['filepath'])
        print(f"\nCamera {cam}:")
        print(f"  Hip  XYZ: {skel[:, 0, :].mean(axis=0).round(3)}")
        print(f"  Head XYZ: {skel[:,10, :].mean(axis=0).round(3)}")
        print(f"  X range: [{skel[:,:,0].min():.2f}, {skel[:,:,0].max():.2f}]")
        print(f"  Z range: [{skel[:,:,2].min():.2f}, {skel[:,:,2].max():.2f}]")
    else:
        print(f"\nCamera {cam}: no file found")


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUMultiViewDataset (mid fusion)
# ══════════════════════════════════════════════════════════════════════════

class BZUMultiViewDataset(Dataset):
    """
    Returns (skel_c0, skel_c1, skel_c2, quality, exercise_id)
    Each skel_cX has shape (T, J, 6) — position + velocity.
    """
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        skels = []
        for cam in CAMERAS:
            skel = load_skeleton(row[f'path_c{cam}'])
            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3), dtype=np.float32)
            skel = self._normalise_length(skel)
            if self.augment:
                skel = self._augment(skel)

            velocity       = np.zeros_like(skel)
            velocity[1:]   = skel[1:] - skel[:-1]
            skel_vel       = np.concatenate([skel, velocity], axis=-1)  # (T, J, 6)
            skels.append(torch.tensor(skel_vel, dtype=torch.float32))

        quality     = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)
        return skels[0], skels[1], skels[2], quality, exercise_id

    def _normalise_length(self, skel):
        T = skel.shape[0]
        if T == self.target_frames:
            return skel
        old_idx = np.linspace(0, 1, T)
        new_idx = np.linspace(0, 1, self.target_frames)
        out = np.zeros((self.target_frames, skel.shape[1], skel.shape[2]), dtype=np.float32)
        for j in range(skel.shape[1]):
            for ax in range(skel.shape[2]):
                out[:, j, ax] = np.interp(new_idx, old_idx, skel[:, j, ax])
        return out

    def _augment(self, skel):
        T     = skel.shape[0]
        speed = np.random.uniform(0.75, 1.25)
        n_new = max(10, int(T * speed))
        idxs  = np.linspace(0, T - 1, n_new).astype(int)
        skel  = self._normalise_length(skel[idxs])

        keep_ratio = np.random.uniform(0.80, 1.0)
        n_keep     = max(10, int(self.target_frames * keep_ratio))
        keep_idxs  = np.sort(np.random.choice(self.target_frames, n_keep, replace=False))
        skel       = self._normalise_length(skel[keep_idxs])
        return skel


print('✓ BZUMultiViewDataset defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN backbone + Camera Cross-Transformer Fusion
#
# Mid-Fusion بعد التعديل:
#   C0: (B, T, J, 6) → per-camera PointWise FC (6→64) ──┐
#   C1: (B, T, J, 6) → per-camera PointWise FC (6→64) ──┤→ concat (B,T,J,192)
#   C2: (B, T, J, 6) → per-camera PointWise FC (6→64) ──┘
#                                   ↓
#                     Shared ST-GCN Backbone → (B, 256) per camera
#                     [كل كاميرا تعطي token مستقل]
#                                   ↓
#                ┌─────────────────────────────────────────┐
#                │  CameraFusionTransformer                │
#                │   Input:  (B, 3, 256)  [3 cam tokens]  │
#                │   Cross-Attention: each cam attends     │
#                │   to the other two                      │
#                │   Output: (B, 3, 256)  → mean pool      │
#                │           → (B, 256)                    │
#                └─────────────────────────────────────────┘
#                                   ↓
#                    Exercise Embedding (32)
#                                   ↓
#                    Regression Head → quality ∈ [1, 5]
# ══════════════════════════════════════════════════════════════════════════

from collections import deque


def get_joint_distances(num_joints, edges, center_joint=0):
    adj  = {i: [] for i in range(num_joints)}
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


# ── Spatial Graph Convolution ─────────────────────────────────────────────

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
        pad           = (temporal_kernel - 1) // 2
        self.spatial  = SpatialGraphConv(in_channels, out_channels, K)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x   = self.spatial(x, A)
        x   = self.temporal(x)
        return self.relu(x + res)


# ── Per-Camera Encoder ────────────────────────────────────────────────────

class CameraEncoder(nn.Module):
    """
    Point-wise encoder: FC(in_features → encoder_dim) + BN + ReLU + Dropout
    Applied per (frame, joint).
    """
    def __init__(self, in_features=IN_FEATURES, encoder_dim=ENCODER_DIM, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_features, encoder_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, C_in, T, J) → (B, encoder_dim, T, J)"""
        return self.encoder(x)


# ══════════════════════════════════════════════════════════════════════════
# ── Camera Fusion Transformer  ────────────────────────────────────────────
#
# Input:  (B, 3, D)  — 3 camera tokens, each a 256-dim vector
# Output: (B, 3, D)  — updated tokens after cross-attention
#
# Each camera "queries" the other two via multi-head cross-attention.
# Stacks L identical layers of:
#    1. Pre-norm cross-attention (each cam → all cams)
#    2. Pre-norm Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════

class CameraFusionTransformerLayer(nn.Module):
    """
    Single cross-attention layer for N camera tokens.

    - Uses standard nn.MultiheadAttention with Q=K=V from the same set
      of camera tokens (self-attention across cameras).
      This lets each camera token attend to all others (cross-camera).
    - Pre-LayerNorm residual connections for stable training.
    - Pointwise FFN expands to ffn_dim then contracts back.
    """
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        # Cross-attention: Q, K, V all come from camera tokens
        # Each token attends to every other token (including itself)
        self.attn    = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,   # (B, N, D) convention
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)   N = number of cameras (3)
        returns: (B, N, D)
        """
        # ── Cross-attention (pre-norm) ─────────────────────────────────
        h     = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)   # Q=K=V → each cam attends all
        x     = x + attn_out               # residual

        # ── FFN (pre-norm) ────────────────────────────────────────────
        x     = x + self.ffn(self.norm2(x))
        return x


class CameraFusionTransformer(nn.Module):
    """
    Stack of L cross-attention layers over camera tokens.

    Takes the GAP output of each camera's ST-GCN backbone —
    3 vectors of shape (B, D) — stacks them into (B, 3, D),
    runs L transformer layers, then mean-pools back to (B, D).

    Args:
        d_model   : feature dim from ST-GCN backbone (256)
        num_heads : attention heads (must divide d_model evenly)
        num_layers: how many cross-attention layers to stack
        ffn_dim   : inner FFN dimension (typically 2×–4× d_model)
        dropout   : dropout inside attention and FFN
    """
    def __init__(
        self,
        d_model:    int = 256,
        num_heads:  int = 8,
        num_layers: int = 2,
        ffn_dim:    int = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.layers = nn.ModuleList([
            CameraFusionTransformerLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)   # final norm before pooling

    def forward(self, cam_features):
        """
        cam_features: list of 3 tensors, each (B, D)
        returns     : (B, D)  — mean-pooled fused representation
        """
        # Stack: [(B,D), (B,D), (B,D)] → (B, 3, D)
        x = torch.stack(cam_features, dim=1)

        # Pass through L cross-attention layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)            # (B, 3, D)

        # Mean-pool over camera tokens → (B, D)
        return x.mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════
# ── Full Model: ST-GCN + CameraFusionTransformer ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════

NUM_EXERCISES = 0   # filled after EXERCISE_REMAP is built


class STGCN_MidFusion(nn.Module):
    """
    ST-GCN Mid-Fusion + Cross-Transformer Regression.

    Architecture:
      C0 (B,T,J,6) ──┐
      C1 (B,T,J,6) ──┤→ 3× CameraEncoder(6→64)
      C2 (B,T,J,6) ──┘        ↓
                       concat → (B, 192, T, J)
                               ↓
                       DataBN (192×J)
                               ↓
                       Shared ST-GCN 9 blocks → (B, 256, T', J')
                               ↓
                       GAP per camera branch split:
                           [Wait — we run ONE shared backbone on the concat,
                            so we get ONE (B,256) global vector. To get 3
                            separate camera tokens we run 3 INDEPENDENT
                            backbones, one per camera.]

    NOTE: to get 3 meaningful camera-specific tokens for the cross-transformer
    we run each camera through its own ST-GCN backbone (sharing weights saves
    params and acts as regularisation — set share_backbone=True).

    share_backbone=True  → 1 backbone, weights tied across cameras
    share_backbone=False → 3 independent backbones
    """
    def __init__(
        self,
        in_features:    int   = IN_FEATURES,
        encoder_dim:    int   = ENCODER_DIM,
        K:              int   = 3,
        dropout:        float = 0.5,
        # Cross-Transformer args
        xfmr_heads:     int   = 8,
        xfmr_layers:    int   = 2,
        xfmr_ffn_dim:   int   = 512,
        xfmr_dropout:   float = 0.1,
        share_backbone: bool  = True,
    ):
        super().__init__()

        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer('A', A)

        self.share_backbone = share_backbone

        # ── Per-camera encoders (6 → 64) ─────────────────────────────
        self.encoder_c0 = CameraEncoder(in_features, encoder_dim, dropout=0.3)
        self.encoder_c1 = CameraEncoder(in_features, encoder_dim, dropout=0.3)
        self.encoder_c2 = CameraEncoder(in_features, encoder_dim, dropout=0.3)

        # ── Input BN on fused channels ────────────────────────────────
        # When feeding each camera independently: each cam has encoder_dim channels
        self.data_bn = nn.BatchNorm1d(encoder_dim * NUM_JOINTS)

        # ── ST-GCN Backbone(s) ────────────────────────────────────────
        # Each backbone takes encoder_dim (64) channels → 256 channels
        backbone_cfg = [
            dict(in_c=encoder_dim, out_c=64,  stride=1, residual=False),
            dict(in_c=64,          out_c=64,  stride=1),
            dict(in_c=64,          out_c=64,  stride=1),
            dict(in_c=64,          out_c=128, stride=2),
            dict(in_c=128,         out_c=128, stride=1),
            dict(in_c=128,         out_c=128, stride=1),
            dict(in_c=128,         out_c=256, stride=2),
            dict(in_c=256,         out_c=256, stride=1),
            dict(in_c=256,         out_c=256, stride=1),
        ]

        def _make_backbone():
            blocks = []
            for cfg in backbone_cfg:
                residual = cfg.get('residual', True)
                blocks.append(STGCNBlock(
                    cfg['in_c'], cfg['out_c'],
                    K=K, stride=cfg['stride'],
                    dropout=dropout, residual=residual,
                ))
            return nn.ModuleList(blocks)

        if share_backbone:
            # One set of weights, called 3 times
            self.backbone = _make_backbone()
        else:
            self.backbone_c0 = _make_backbone()
            self.backbone_c1 = _make_backbone()
            self.backbone_c2 = _make_backbone()

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Camera Fusion Transformer ─────────────────────────────────
        self.camera_transformer = CameraFusionTransformer(
            d_model    = 256,
            num_heads  = xfmr_heads,
            num_layers = xfmr_layers,
            ffn_dim    = xfmr_ffn_dim,
            dropout    = xfmr_dropout,
        )

        # ── Regression Head ───────────────────────────────────────────
        self.ex_embed = nn.Embedding(NUM_EXERCISES, 32)
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

    def _prep(self, x):
        """(B, T, J, 6) → (B, 6, T, J)"""
        return x.permute(0, 3, 1, 2)

    def _run_backbone(self, x, cam_idx=None):
        """
        x: (B, encoder_dim, T, J)
        returns: (B, 256)  via GAP
        """
        B  = x.shape[0]
        J  = x.shape[3]
        FC = x.shape[1]   # encoder_dim = 64

        # Data BatchNorm
        xbn = x.reshape(B, FC * J, x.shape[2])
        xbn = self.data_bn(xbn)
        xbn = xbn.reshape(B, FC, J, x.shape[2]).permute(0, 1, 3, 2)  # (B,FC,T,J)

        # ST-GCN blocks
        if self.share_backbone:
            blocks = self.backbone
        else:
            blocks = [self.backbone_c0, self.backbone_c1, self.backbone_c2][cam_idx]

        for block in blocks:
            xbn = block(xbn, self.A)

        # Global Average Pool → (B, 256)
        return self.gap(xbn).squeeze(-1).squeeze(-1)

    def forward(self, x_c0, x_c1, x_c2, exercise_id):
        """
        x_c0, x_c1, x_c2 : (B, T, J, 6)
        exercise_id       : (B,)
        returns           : (B,)  quality scores ∈ [1, 5]
        """
        # ── Step 1: per-camera point-wise encoding ─────────────────
        f0 = self.encoder_c0(self._prep(x_c0))   # (B, 64, T, J)
        f1 = self.encoder_c1(self._prep(x_c1))
        f2 = self.encoder_c2(self._prep(x_c2))

        # ── Step 2: each camera through ST-GCN → (B, 256) ──────────
        g0 = self._run_backbone(f0, cam_idx=0)   # (B, 256)
        g1 = self._run_backbone(f1, cam_idx=1)
        g2 = self._run_backbone(f2, cam_idx=2)

        # ── Step 3: Camera Fusion Transformer ──────────────────────
        # Input:  [(B,256), (B,256), (B,256)]
        # Output: (B, 256)  — cross-attended and mean-pooled
        fused = self.camera_transformer([g0, g1, g2])   # (B, 256)

        # ── Step 4: exercise embedding + regression ─────────────────
        ex  = self.ex_embed(exercise_id)                # (B, 32)
        h   = torch.cat([fused, ex], dim=1)             # (B, 288)
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# ── Patch NUM_EXERCISES ───────────────────────────────────────────────────
NUM_EXERCISES = len(EXERCISE_REMAP)

# ── Sanity check ──────────────────────────────────────────────────────────
_c0  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, 6)
_c1  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, 6)
_c2  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, 6)
_ex  = torch.zeros(2, dtype=torch.long)
_mdl = STGCN_MidFusion(share_backbone=True)
_out = _mdl(_c0, _c1, _c2, _ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
total_params = sum(p.numel() for p in _mdl.parameters() if p.requires_grad)
print(f'\n✓ STGCN_MidFusion + CameraFusionTransformer — output: {_out.shape}')
print(f'✓ Total trainable parameters: {total_params:,}')

# ── Model parameter breakdown ─────────────────────────────────────────────
def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

print(f'\n  Parameter breakdown:')
print(f'  Camera encoders     : {count_params(_mdl.encoder_c0)*3:,}  (×3)')
if _mdl.share_backbone:
    print(f'  Shared backbone     : {count_params(nn.ModuleList(_mdl.backbone)):,}')
print(f'  Camera Transformer  : {count_params(_mdl.camera_transformer):,}')
print(f'  Exercise embedding  : {count_params(_mdl.ex_embed):,}')
print(f'  Regression head     : {count_params(_mdl.reg_head):,}')

del _c0, _c1, _c2, _ex, _mdl, _out
print('\n✓ STGCN_MidFusion with Cross-Transformer defined')

# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    x: (B, T, J, 6) — first 3 = position, last 3 = velocity
    Hip-centred + torso-height normalised.
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


from scipy.stats import pearsonr


def run_epoch(model, loader, reg_fn, is_train=True, optimiser=None):
    model.train() if is_train else model.eval()
    total_loss   = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
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


print('✓ centre_and_scale and run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13.5 — Split quality distribution audit
# ══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Quality score distribution audit")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Quality Score Distributions per Split", fontsize=14, fontweight='bold')

splits_audit = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
trial_types  = [('Correct (T≤2)',   lambda d: d[d['trial_num'] <= 2]),
                ('Erroneous (T≥3)', lambda d: d[d['trial_num'] >= 3])]

for col, (split_name, split_df) in enumerate(splits_audit):
    for row, (type_name, filter_fn) in enumerate(trial_types):
        ax  = axes[row][col]
        sub = filter_fn(split_df)
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{split_name} — {type_name}')
            continue
        q = sub['quality']
        ax.hist(q, bins=30, color='steelblue' if row == 0 else 'tomato',
                edgecolor='black', alpha=0.8)
        ax.axvline(q.mean(),   color='red',  linestyle='--', linewidth=2,
                   label=f'mean={q.mean():.3f}')
        ax.axvline(q.median(), color='gold', linestyle=':',  linewidth=2,
                   label=f'med={q.median():.3f}')
        ax.set_title(f'{split_name} — {type_name}\n'
                     f'n={len(sub)}  std={q.std():.3f}  '
                     f'[{q.min():.2f}, {q.max():.2f}]', fontsize=9)
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
save_path_audit = os.path.join(PLOTS_DIR, 'split_quality_distributions.png')
plt.savefig(save_path_audit, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved → {save_path_audit}")

print(f"\n{'Split':<8} {'Type':<12} {'n':>6} {'mean':>7} {'std':>7} "
      f"{'min':>6} {'25%':>6} {'50%':>6} {'75%':>6} {'max':>6}")
print("-" * 70)
for split_name, split_df in splits_audit:
    for type_name, filter_fn in trial_types:
        sub = filter_fn(split_df)
        if len(sub) == 0:
            continue
        q = sub['quality']
        print(f"{split_name:<8} {type_name:<12} {len(sub):>6} "
              f"{q.mean():>7.3f} {q.std():>7.3f} "
              f"{q.min():>6.2f} {q.quantile(.25):>6.2f} "
              f"{q.median():>6.2f} {q.quantile(.75):>6.2f} {q.max():>6.2f}")
    print()

print("=" * 65)
print("Person distribution across splits")
print("=" * 65)
for split_name, split_df in splits_audit:
    persons = sorted(split_df['person'].unique())
    print(f"  {split_name:<6}: {persons}")

print("\nPer-person quality mean per split:")
for split_name, split_df in splits_audit:
    print(f"\n  {split_name}:")
    summary = (split_df.groupby('person')['quality']
               .agg(['mean', 'std', 'count'])
               .round(3))
    print(summary.to_string())

print("\n" + "=" * 65)
print("Exercise balance across splits")
print("=" * 65)
ex_counts = pd.DataFrame({
    name: split_df['exercise'].value_counts().sort_index()
    for name, split_df in splits_audit
})
ex_counts.index   = [f'E{i}' for i in ex_counts.index]
ex_counts.columns = ['Train', 'Val', 'Test']
ex_counts['Train%'] = (ex_counts['Train'] / ex_counts['Train'].sum() * 100).round(1)
ex_counts['Val%']   = (ex_counts['Val']   / ex_counts['Val'].sum()   * 100).round(1)
ex_counts['Test%']  = (ex_counts['Test']  / ex_counts['Test'].sum()  * 100).round(1)
print(ex_counts.to_string())


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
    ax.set_title('MAE + Early Stopping', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))


print('✓ Plotting helpers defined')


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


def make_ds(df, aug):
    return BZUMultiViewDataset(df, augment=aug)


def make_weighted_sampler(df):
    q = df['quality'].values.astype(np.float32)
    raw_weights = (q.max() + 1.0) - q
    erroneous_mask = (df['trial_num'].values >= 3).astype(np.float32)
    raw_weights   *= (1.0 + erroneous_mask)
    weights_tensor = torch.DoubleTensor(raw_weights)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights_tensor, num_samples=len(df), replacement=True)


train_sampler = make_weighted_sampler(train_mv)

train_loader = DataLoader(
    make_ds(train_mv, aug=True),
    batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=0, pin_memory=(DEVICE == 'cuda'),
)
val_loader  = DataLoader(make_ds(val_mv,  False), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=(DEVICE=='cuda'))
test_loader = DataLoader(make_ds(test_mv, False), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=(DEVICE=='cuda'))
# بعد:
model = STGCN_MidFusion(
    in_features    = IN_FEATURES,
    encoder_dim    = ENCODER_DIM,
    K              = 3,
    dropout        = 0.5,
    xfmr_heads     = 8,      # عدد attention heads (256/8=32 dim per head)
    xfmr_layers    = 2,      # عدد cross-attention layers
    xfmr_ffn_dim   = 512,    # inner FFN dimension
    xfmr_dropout   = 0.1,    # dropout داخل الـ transformer
    share_backbone = True,   # True = backbone مشترك، False = 3 backbones مستقلة
).to(DEVICE)

optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=50, min_lr=1e-6, verbose=True,
)
early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val']
METRICS = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info('STARTING MID-FUSION ST-GCN TRAINING  (Early Stopping on MAE)')
log.info(f'  Encoder: {IN_FEATURES}→{ENCODER_DIM} per camera  |  Fused: {ENCODER_DIM*3} channels')
log.info(f'  train={len(train_mv)}  val={len(val_mv)}  test={len(test_mv)}')
log.info('=' * 70)

print(f'\n{"═"*68}')
print(f'  MID-FUSION | Train: {len(train_mv)}  Val: {len(val_mv)}  Test: {len(test_mv)}')
print(f'  Encoder: {IN_FEATURES}→{ENCODER_DIM} per cam → concat {ENCODER_DIM*3} channels → ST-GCN')
print(f'  Patience: {PATIENCE}  |  LR: {LR}  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')

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
        print(f'\n  ⏹  Early stopping at epoch {epoch} (best={early_stop.best_epoch})')
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

# ── Collect predictions ───────────────────────────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for x_c0, x_c1, x_c2, qualities, exercise_ids in test_loader:
        x_c0         = centre_and_scale(x_c0.to(DEVICE))
        x_c1         = centre_and_scale(x_c1.to(DEVICE))
        x_c2         = centre_and_scale(x_c2.to(DEVICE))
        exercise_ids = exercise_ids.to(DEVICE)
        preds        = model(x_c0, x_c1, x_c2, exercise_ids)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# ── Save plots ────────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR,    test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR,          test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR,         test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History JSON → {json_path}')

npz_history_path = os.path.join(LOGS_DIR, 'training_history.npz')
np.savez(npz_history_path, **{k: np.array(v) for k, v in history.items()})
print(f'  ✓ History NPZ → {npz_history_path}')

npz_preds_path = os.path.join(LOGS_DIR, 'test_predictions.npz')
np.savez(npz_preds_path,
         q_true       = np.array(all_true_q),
         q_pred       = np.array(all_pred_q),
         exercise_ids = np.array(all_exercise_ids))
print(f'  ✓ Test predictions NPZ → {npz_preds_path}')


# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC CELL — Data analysis
# ══════════════════════════════════════════════════════════════════════════
from collections import defaultdict

print("=" * 60)
print("DIAGNOSTIC 1: Per-exercise skeleton variance")
print("=" * 60)

axis_var = defaultdict(list)
for _, row in df_index.sample(min(300, len(df_index)), random_state=0).iterrows():
    skel = load_skeleton(row['filepath'])
    if skel is None:
        continue
    axis_var[row['exercise']].append({
        'x_var': skel[:,:,0].var(),
        'y_var': skel[:,:,1].var(),
        'z_var': skel[:,:,2].var(),
    })

print(f"{'Exercise':>10} {'X-var':>10} {'Y-var':>10} {'Z-var':>10} {'Z/X ratio':>10}")
print("-" * 55)
for ex in sorted(axis_var.keys()):
    vals = axis_var[ex]
    xv = np.mean([v['x_var'] for v in vals])
    yv = np.mean([v['y_var'] for v in vals])
    zv = np.mean([v['z_var'] for v in vals])
    print(f"{f'E{ex}':>10} {xv:>10.4f} {yv:>10.4f} {zv:>10.4f} {zv/max(xv,1e-6):>10.3f}")

print()
print("=" * 60)
print("DIAGNOSTIC 2: Quality score distribution per split")
print("=" * 60)
for name, df_ in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    q = df_['quality']
    trials_correct   = (df_['trial_num'] <= 2).sum()
    trials_erroneous = (df_['trial_num'] >= 3).sum()
    print(f"{name:>6}: mean={q.mean():.3f} std={q.std():.3f} "
          f"min={q.min():.2f} max={q.max():.2f} | "
          f"correct={trials_correct} erroneous={trials_erroneous}")

# ── Save history as NPZ ───────────────────────────────────────────────────
npz_history_path = os.path.join(LOGS_DIR, 'training_history.npz')
np.savez(npz_history_path, **{k: np.array(v) for k, v in history.items()})
print(f'  ✓ History NPZ → {npz_history_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16.5 — Per-exercise test metrics
# ══════════════════════════════════════════════════════════════════════════

all_true_q_arr = np.array(all_true_q)
all_pred_q_arr = np.array(all_pred_q)
all_ex_arr     = np.array(all_exercise_ids)

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

per_ex_df  = pd.DataFrame([{'exercise': f'E{ex}', **vals} for ex, vals in per_ex_results.items()])
per_ex_csv = os.path.join(LOGS_DIR, 'per_exercise_metrics.csv')
per_ex_df.to_csv(per_ex_csv, index=False)
print(f'  ✓ Per-exercise CSV → {per_ex_csv}')

n_ex   = len(unique_exercises)
n_cols = 3
n_rows = int(np.ceil(n_ex / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle('Per-Exercise: True vs Predicted Quality (Test Set)',
             fontsize=14, fontweight='bold')

for i, ex_id in enumerate(unique_exercises):
    ax   = axes[i]
    mask = all_ex_arr == ex_id
    qt   = all_true_q_arr[mask]
    qp   = all_pred_q_arr[mask]
    res  = per_ex_results[ex_id]
    ax.scatter(qt, qp, alpha=0.65, edgecolors='black', linewidths=0.4,
               color='steelblue', s=55)
    lo = min(qt.min(), qp.min()) - 0.2
    hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f'Exercise E{ex_id}  (n={res["n"]})', fontsize=10, fontweight='bold')
    ax.set_xlabel('True Quality', fontsize=9)
    ax.set_ylabel('Predicted Quality', fontsize=9)
    ax.grid(alpha=0.3)
    textstr = (f'MAE  = {res["mae"]:.3f}\nRMSE = {res["rmse"]:.3f}\n'
               f'R²   = {res["r2"]:.3f}\nPCC  = {res["pcc"]:.3f}')
    ax.text(0.05, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
scatter_grid_path = os.path.join(PLOTS_DIR, 'per_exercise_scatter.png')
plt.savefig(scatter_grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise scatter grid → {scatter_grid_path}')

# ── Bar chart comparison across exercises ────────────────────────────────
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
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Exercise')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
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


npz_per_ex_path = os.path.join(LOGS_DIR, 'per_exercise_metrics.npz')
np.savez(npz_per_ex_path,
         exercise_ids = np.array(unique_exercises),
         n    = np.array([per_ex_results[e]['n']    for e in unique_exercises]),
         mae  = np.array([per_ex_results[e]['mae']  for e in unique_exercises]),
         rmse = np.array([per_ex_results[e]['rmse'] for e in unique_exercises]),
         r2   = np.array([per_ex_results[e]['r2']   for e in unique_exercises]),
         pcc  = np.array([per_ex_results[e]['pcc']  for e in unique_exercises]),
)
print(f'  ✓ Per-exercise metrics NPZ → {npz_per_ex_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv           = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 60)
print('  TRAINING SUMMARY — ST-GCN Mid-Fusion (3 Cameras)')
print('=' * 60)
print(f'  Fusion           : MID ({IN_FEATURES}→{ENCODER_DIM} per cam → concat {ENCODER_DIM*3})')
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
log.info(f'Test MAE={final_te["mae"]:.4f}  RMSE={final_te["rmse"]:.4f}  '
         f'R²={final_te["r2"]:.4f}  PCC={final_te["pcc"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [{'split': 'test_overall', 'exercise': 'all',
         'fusion': 'mid',
         'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
         'val_mae': best_val_mae,  'val_rmse': best_val_rmse,
         'val_r2':  best_val_r2,   'val_pcc':  best_val_pcc,
         'test_mae': final_te['mae'], 'test_rmse': final_te['rmse'],
         'test_r2':  final_te['r2'], 'test_pcc':  final_te['pcc']}]
for ex, vals in per_ex_results.items():
    rows.append({'split': 'test_per_exercise', 'exercise': f'E{ex}',
                 'fusion': 'mid',
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
