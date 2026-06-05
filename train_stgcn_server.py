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
BATCH_SIZE    = 48
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_MidFusion_MultiView_Regression_Results"

# ── Mid-Fusion specific ───────────────────────────────────────────────────
REQUIRED_CAMERAS  = {0, 1, 2}          # all 3 cameras must be present
IN_FEATURES       = 6                  # pos(3) + vel(3) per joint per camera
ENCODER_OUT_DIM   = 64                 # per-camera encoder output features per joint
# After concat of 3 cameras: 64*3 = 192 features per joint
# Projection layer maps 192 → PROJ_DIM before entering ST-GCN blocks
PROJ_DIM          = 64

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Exercise Filter ───────────────────────────────────────────────────────
EXCLUDED_EXERCISES = {3, 7, 9}
EXERCISE_REMAP     = {}

print('✓ Configuration loaded')
print(f'  DATASET_DIR  : {DATASET_DIR}')
print(f'  SPLIT_DIR    : {SPLIT_DIR}')
print(f'  NPZ_KEY      : {NPZ_KEY}')
print(f'  CAMERAS      : {sorted(REQUIRED_CAMERAS)}')
print(f'  ENCODER_OUT  : {ENCODER_OUT_DIM} per camera → {ENCODER_OUT_DIM*3} fused → proj {PROJ_DIM}')
print(f'  EXISTS       : {os.path.exists(DATASET_DIR)}')
print(f'  SPLIT EXISTS : {os.path.exists(SPLIT_DIR)}')
print(f'  PATIENCE     : {PATIENCE} epochs')


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

run_name = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
RUN_DIR  = os.path.join(OUT_DIR, run_name)
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
LOGS_DIR  = os.path.join(RUN_DIR, "logs")

for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Run directory created:", RUN_DIR)
print("✓ Libraries imported")
print("✓ Output folders ready:")
for d in [PLOTS_DIR, LOGS_DIR]:
    print("  ", d)

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
# Cell 3 — Explore dataset folder & one NPZ file
# ══════════════════════════════════════════════════════════════════════════

all_npz = sorted(glob.glob(os.path.join(DATASET_DIR, '**/*.npz'), recursive=True))
print(f'Total NPZ files found : {len(all_npz)}')

if len(all_npz) == 0:
    print('\n❌ No NPZ files found! Checking folder contents...')
    try:
        for item in sorted(os.listdir(DATASET_DIR))[:20]:
            print(' ', item)
    except Exception as e:
        print(f'  Cannot list: {e}')
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
            else:
                print(f'  {enc}: wrong columns → {tmp.columns.tolist()[:4]}')
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
print(f'✓ stdout → also writing to {log_file}')

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("GCN-MidFusion-Regression")
log.info("=" * 70)
log.info("ST-GCN Mid-Fusion Multi-View Regression | BZU Physiotherapy Dataset")
log.info(f"Cameras: {sorted(REQUIRED_CAMERAS)}  |  Encoder: {IN_FEATURES}→{ENCODER_OUT_DIM}→proj{PROJ_DIM}")
log.info(f"Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
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
        print(f'  ⚠️  No records found for split="{split_name}", camera={camera_id}')
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
    print(f'  Exercise dist    :\n{df["exercise"].value_counts().sort_index().to_string()}')
    return df


def filter_complete_camera_groups(df, label=''):
    """
    For mid-fusion: keep ONLY trial_key+segment groups that have ALL 3 cameras.
    Groups the index by (trial_key, segment) since each segment needs all cameras.
    """
    # Group by (trial_key, segment) — each physical clip must have all 3 cameras
    coverage = df.groupby(['trial_key', 'segment'])['camera'].apply(set)
    complete   = coverage[coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))].index
    incomplete = coverage[~coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))]

    n_before = len(df)
    df = df.set_index(['trial_key', 'segment'])
    df = df.loc[df.index.isin(complete)].reset_index()
    n_skipped_files  = n_before - len(df)
    n_skipped_groups = len(incomplete)

    if n_skipped_groups:
        print(f'  [{label}] ⚠️  Skipped {n_skipped_files} files '
              f'from {n_skipped_groups} (trial,segment) groups with incomplete camera coverage')
    else:
        print(f'  [{label}] ✓ All (trial,segment) groups have complete 3-camera coverage')

    return df


def remove_corrupted(df, label=''):
    bad = []
    for fpath in df['filepath']:
        if load_skeleton(fpath) is None:
            bad.append(fpath)
    if bad:
        print(f'  [{label}] Removing {len(bad)} corrupted file(s)')
        df = df[~df['filepath'].isin(bad)].reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


def exclude_exercises(df, excluded=EXCLUDED_EXERCISES, label=''):
    before = len(df)
    df = df[~df['exercise'].isin(excluded)].reset_index(drop=True)
    print(f'  [{label}] Excluded E{sorted(excluded)} → '
          f'{before - len(df)} rows dropped, {len(df)} remain')
    return df


# ── STEP 1: Build raw splits ──────────────────────────────────────────────
train_df = build_index_from_split('train', df_csv, camera_id=None)
val_df   = build_index_from_split('valid', df_csv, camera_id=None)
test_df  = build_index_from_split('test',  df_csv, camera_id=None)

# ── STEP 2: Camera completeness filter (per trial+segment) ───────────────
train_df = filter_complete_camera_groups(train_df, 'TRAIN')
val_df   = filter_complete_camera_groups(val_df,   'VALID')
test_df  = filter_complete_camera_groups(test_df,  'TEST')

# ── STEP 3: Remove corrupted files ───────────────────────────────────────
print('\nChecking for corrupted files...')
train_df = remove_corrupted(train_df, 'TRAIN')
val_df   = remove_corrupted(val_df,   'VALID')
test_df  = remove_corrupted(test_df,  'TEST')

# ── STEP 4: Exclude E3, E7, E9 ───────────────────────────────────────────
print('\nExcluding exercises...')
train_df = exclude_exercises(train_df, label='TRAIN')
val_df   = exclude_exercises(val_df,   label='VALID')
test_df  = exclude_exercises(test_df,  label='TEST')

# ── STEP 5: Remap exercise IDs to contiguous 0-based integers ────────────
remaining_exercises = sorted(
    set(train_df['exercise'].unique()) |
    set(val_df['exercise'].unique())   |
    set(test_df['exercise'].unique())
)
EXERCISE_REMAP = {orig: new for new, orig in enumerate(remaining_exercises)}
print(f'\n  Exercise ID remap : {EXERCISE_REMAP}')

for df_ in [train_df, val_df, test_df]:
    df_['exercise'] = df_['exercise'].map(EXERCISE_REMAP)

# ── STEP 6: Build multi-view index grouped by (trial_key, segment) ────────
#
# For mid-fusion, each Dataset item is ONE (trial_key, segment) group
# containing the filepaths for all 3 cameras.
# We pivot the per-camera rows into a single row with columns:
#   filepath_c0, filepath_c1, filepath_c2, quality, exercise, trial_num, ...

def build_multiview_index(df, label=''):
    """
    Pivot per-camera rows into one row per (trial_key, segment).
    Each output row has filepath_c0, filepath_c1, filepath_c2.
    """
    records = []
    for (trial_key, segment), grp in df.groupby(['trial_key', 'segment']):
        cam_map = {int(row['camera']): row['filepath'] for _, row in grp.iterrows()}
        if not REQUIRED_CAMERAS.issubset(cam_map.keys()):
            continue
        row0 = grp.iloc[0]
        records.append({
            'trial_key'  : trial_key,
            'segment'    : segment,
            'filepath_c0': cam_map[0],
            'filepath_c1': cam_map[1],
            'filepath_c2': cam_map[2],
            'quality'    : row0['quality'],
            'exercise'   : row0['exercise'],
            'trial_num'  : row0['trial_num'],
            'trial_id'   : row0['trial_id'],
            'person'     : row0['person'],
        })
    out = pd.DataFrame(records).reset_index(drop=True)
    print(f'  [{label}] Multi-view groups (items): {len(out)}')
    return out


train_mv = build_multiview_index(train_df, 'TRAIN')
val_mv   = build_multiview_index(val_df,   'VALID')
test_mv  = build_multiview_index(test_df,  'TEST')

# ── STEP 7: Combine & leakage check ──────────────────────────────────────
df_index = pd.concat([train_mv, val_mv, test_mv], ignore_index=True)
print(f'  Remaining exercises (remapped): {sorted(df_index["exercise"].unique())}')

tr_keys = set(train_mv['trial_key'])
vl_keys = set(val_mv['trial_key'])
te_keys = set(test_mv['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

print(f'\n{"═"*68}')
print(f'  {"Split":<8} {"Groups":>8} {"Correct":>9} {"Erroneous":>11} '
      f'{"Q mean":>8} {"Q std":>7} {"Q min":>7} {"Q max":>7}')
print(f'  {"─"*66}')
for name, d in [('Train', train_mv), ('Val', val_mv), ('Test', test_mv)]:
    cor = (d['trial_num'] <= 2).sum()
    err = (d['trial_num'] >= 3).sum()
    q   = d['quality']
    print(f'  {name:<8} {len(d):>8} {cor:>9} {err:>11} '
          f'{q.mean():>8.3f} {q.std():>7.3f} '
          f'{q.min():>7.3f} {q.max():>7.3f}')
print(f'{"═"*68}')

print('\n✓ Multi-view index ready  →  train_mv / val_mv / test_mv')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.5 — Camera Distribution & Diagnostic Checks
# ══════════════════════════════════════════════════════════════════════════

print(f"\nMulti-view dataset sizes:")
print(f"  Train groups : {len(train_mv)}")
print(f"  Val groups   : {len(val_mv)}")
print(f"  Test groups  : {len(test_mv)}")

print(f"\nExercise distribution in multi-view train:")
print(train_mv['exercise'].value_counts().sort_index())

# Verify all 3 camera files exist for a sample
sample_row = train_mv.iloc[0]
for cam in [0, 1, 2]:
    fp = sample_row[f'filepath_c{cam}']
    skel = load_skeleton(fp)
    status = f"shape={skel.shape}" if skel is not None else "❌ FAILED"
    print(f"  Camera {cam}: {os.path.basename(fp)} → {status}")


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

for split_name, split_df in [('Train', train_mv), ('Val', val_mv), ('Test', test_mv)]:
    print(f"\n{'='*50}")
    print(f"  {split_name}")
    print(f"{'='*50}")
    print(split_df.groupby('exercise')['quality']
          .agg(['mean', 'var', 'count'])
          .round(4)
          .to_string())


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.7 — Frame length distribution
# ══════════════════════════════════════════════════════════════════════════

lengths = []
sample_rows = df_index.sample(min(500, len(df_index)), random_state=42)

for _, row in sample_rows.iterrows():
    skel = load_skeleton(row['filepath_c0'])   # C0 as representative
    if skel is not None:
        lengths.append(skel.shape[0])

lengths = np.array(lengths)
print(f"Frame length distribution (sample of {len(lengths)} files, C0):")
print(f"  min={lengths.min()}  max={lengths.max()}  "
      f"mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  std={lengths.std():.1f}")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p:3d}th = {np.percentile(lengths, p):.0f}")


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


def plot_skeleton_3d(skel, frame_idx=0, title='Skeleton Sanity Check', save_path=None):
    pts = skel[frame_idx]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    views = [
        (axes[0], x,  y,  'Front View  (X–Y)', 'X (left/right)', 'Y (up/down)',  False),
        (axes[1], z,  y,  'Side View   (Z–Y)', 'Z (depth)',       'Y (up/down)', False),
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

print('✓ Skeleton visualisation helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — Visualise a sample skeleton (C0 as representative)
# ══════════════════════════════════════════════════════════════════════════

idx         = 10
sample_row  = df_index.iloc[idx]
sample_skel = load_skeleton(sample_row['filepath_c0'])

print(sample_row[['person', 'exercise', 'trial_id', 'segment', 'filepath_c0']])
print(f'Skeleton shape : {sample_skel.shape}')
print(f'X range : [{sample_skel[:,:,0].min():.3f}, {sample_skel[:,:,0].max():.3f}]')
print(f'Y range : [{sample_skel[:,:,1].min():.3f}, {sample_skel[:,:,1].max():.3f}]')
print(f'Z range : [{sample_skel[:,:,2].min():.3f}, {sample_skel[:,:,2].max():.3f}]')

plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton C0 · {sample_row['trial_key']} · E{sample_row['exercise']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
)

hip  = sample_skel[:, 0, :]
head = sample_skel[:, 10, :]
print("\nHip  mean XYZ:", hip.mean(axis=0))
print("Head mean XYZ:", head.mean(axis=0))
print("Difference (head - hip):", head.mean(axis=0) - hip.mean(axis=0))


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset  *** MID-FUSION VERSION ***
#
# Each __getitem__ loads the SAME (trial_key, segment) from all 3 cameras
# and returns:
#   skel_c0 : (T, J, 6)   position + velocity, camera 0
#   skel_c1 : (T, J, 6)   position + velocity, camera 1
#   skel_c2 : (T, J, 6)   position + velocity, camera 2
#   quality : scalar
#   exercise_id : int
# ══════════════════════════════════════════════════════════════════════════

class BZUDatasetMultiView(Dataset):
    """
    Loads all 3 camera views for each (trial_key, segment) group.
    Returns (skel_c0, skel_c1, skel_c2, quality, exercise_id).
    """
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        # df here is the pivoted multi-view dataframe (one row = one group)
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        skels = []
        for cam in [0, 1, 2]:
            skel = load_skeleton(row[f'filepath_c{cam}'])
            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3), dtype=np.float32)
            skel = self._normalise_length(skel)
            if self.augment:
                skel = self._augment(skel)

            # velocity
            velocity      = np.zeros_like(skel)
            velocity[1:]  = skel[1:] - skel[:-1]

            # stack → (T, J, 6)
            skel_vel = np.concatenate([skel, velocity], axis=-1)
            skels.append(skel_vel)

        skel_c0 = torch.tensor(skels[0], dtype=torch.float32)
        skel_c1 = torch.tensor(skels[1], dtype=torch.float32)
        skel_c2 = torch.tensor(skels[2], dtype=torch.float32)
        quality     = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)

        return skel_c0, skel_c1, skel_c2, quality, exercise_id

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

    def _augment(self, skel):
        T = skel.shape[0]
        speed = np.random.uniform(0.75, 1.25)
        n_new = max(10, int(T * speed))
        idxs  = np.linspace(0, T - 1, n_new).astype(int)
        skel  = self._normalise_length(skel[idxs])

        keep_ratio = np.random.uniform(0.80, 1.0)
        n_keep     = max(10, int(self.target_frames * keep_ratio))
        keep_idxs  = np.sort(
            np.random.choice(self.target_frames, n_keep, replace=False)
        )
        skel = self._normalise_length(skel[keep_idxs])
        return skel

print('✓ BZUDatasetMultiView defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN (Yan et al., AAAI 2018) + Mid-Fusion Camera Encoder
#
# Mid-Fusion pipeline:
#
#   Camera 0 (B,T,J,6) ──[CameraEncoder]──► (B,T,J,64) ─┐
#   Camera 1 (B,T,J,6) ──[CameraEncoder]──► (B,T,J,64) ──┤ concat → (B,T,J,192)
#   Camera 2 (B,T,J,6) ──[CameraEncoder]──► (B,T,J,64) ─┘
#                                                          │
#                                             [FusionProjection] → (B,T,J,64)
#                                                          │
#                                         [9 × ST-GCN Blocks]
#                                                          │
#                                            [GAP] → (B,256)
#                                                          │
#                                      [ExerciseEmbed + RegressionHead]
#                                                          │
#                                                    Quality Score
#
# The CameraEncoder is PER-JOINT: it mixes the 6 features of each joint
# independently (shared MLP weights across joints), so spatial structure
# is preserved for the GCN.
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


# ── Spatial Graph Conv (true ST-GCN) ─────────────────────────────────────

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
        x = self.conv(x)
        x = x.view(B, self.K, -1, T, J)
        A_eff = A + self.M
        out   = torch.einsum('bkctj,kjv->bctv', x, A_eff)
        out   = self.bn(out)
        return self.relu(out)


# ── ST-GCN Block ─────────────────────────────────────────────────────────

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


# ── *** NEW: Per-Camera Encoder (Mid-Fusion) *** ──────────────────────────
#
# Operates per-joint: treats each (joint, time) independently.
# Applied as a 1×1 conv along the feature axis → preserves (T, J) structure.
# Independent weights per camera (not shared) → each camera can learn its own
# view-specific feature embedding before fusion.

class CameraEncoder(nn.Module):
    """
    Encodes one camera's skeleton features per joint:
        (B, C_in, T, J) → (B, C_out, T, J)

    Uses 1×1 Conv2d so weights are shared across (T, J) positions —
    i.e., the same small MLP is applied to every joint at every frame,
    which is exactly right for skeleton data.

    In_channels  = IN_FEATURES = 6  (pos_xyz + vel_xyz)
    Out_channels = ENCODER_OUT_DIM = 64
    """
    def __init__(self, in_channels=IN_FEATURES, out_channels=ENCODER_OUT_DIM,
                 dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, C_in, T, J) → (B, C_out, T, J)"""
        return self.net(x)


# ── *** NEW: Fusion Projection *** ────────────────────────────────────────

class FusionProjection(nn.Module):
    """
    Projects concatenated 3-camera features back to PROJ_DIM:
        (B, 3*ENCODER_OUT_DIM, T, J) → (B, PROJ_DIM, T, J)

    Uses a 1×1 conv + BN + ReLU — same per-joint pattern.
    """
    def __init__(self, in_channels=ENCODER_OUT_DIM * 3,
                 out_channels=PROJ_DIM, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, 3*enc, T, J) → (B, PROJ_DIM, T, J)"""
        return self.net(x)


# ── Full Mid-Fusion Model ─────────────────────────────────────────────────

NUM_EXERCISES = len(EXERCISE_REMAP)

class STGCN_MidFusion(nn.Module):
    """
    ST-GCN with Mid-Fusion of 3 camera views.

    Architecture:
      C0 (B,T,J,6) ──[Encoder0]──► (B,T,J,64)  ─┐
      C1 (B,T,J,6) ──[Encoder1]──► (B,T,J,64)  ──┤ concat
      C2 (B,T,J,6) ──[Encoder2]──► (B,T,J,64)  ─┘
                                       ↓
                             (B, T, J, 192)
                                       ↓
                           [FusionProjection]
                                       ↓
                             (B, T, J, 64)
                                       ↓
                         [Data BN on input features]
                                       ↓
                           [9 × ST-GCN Blocks]
                                       ↓
                        [Global Average Pooling] → (B, 256)
                                       ↓
                     [Exercise Embedding (32-dim)]
                                       ↓
                         [Regression Head] → score ∈ [1,5]
    """
    def __init__(self, K=3, dropout=0.5):
        super().__init__()

        # Adjacency
        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer('A', A)

        # ── Mid-Fusion components ─────────────────────────────────────────
        # Independent encoder per camera (each learns its own view embedding)
        self.encoder_c0 = CameraEncoder(IN_FEATURES, ENCODER_OUT_DIM, dropout=0.3)
        self.encoder_c1 = CameraEncoder(IN_FEATURES, ENCODER_OUT_DIM, dropout=0.3)
        self.encoder_c2 = CameraEncoder(IN_FEATURES, ENCODER_OUT_DIM, dropout=0.3)

        # Fusion projection: 192 → PROJ_DIM (64)
        self.fusion_proj = FusionProjection(
            in_channels  = ENCODER_OUT_DIM * 3,
            out_channels = PROJ_DIM,
            dropout      = 0.3,
        )

        # ── ST-GCN backbone (input dim = PROJ_DIM) ────────────────────────
        # Data BN operates on (B, PROJ_DIM * J, T)
        self.data_bn = nn.BatchNorm1d(PROJ_DIM * NUM_JOINTS)

        self.blocks = nn.ModuleList([
            STGCNBlock(PROJ_DIM, 64,  K=K, residual=False, dropout=dropout),
            STGCNBlock(64,       64,  K=K,                  dropout=dropout),
            STGCNBlock(64,       64,  K=K,                  dropout=dropout),
            STGCNBlock(64,  128, K=K, stride=2,             dropout=dropout),
            STGCNBlock(128, 128,      K=K,                  dropout=dropout),
            STGCNBlock(128, 128,      K=K,                  dropout=dropout),
            STGCNBlock(128, 256, K=K, stride=2,             dropout=dropout),
            STGCNBlock(256, 256,      K=K,                  dropout=dropout),
            STGCNBlock(256, 256,      K=K,                  dropout=dropout),
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)

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

    def _prepare_camera(self, x):
        """
        Reshape (B, T, J, C) → (B, C, T, J) for Conv2d operations.
        """
        return x.permute(0, 3, 1, 2)   # (B, C, T, J)

    def forward(self, x_c0, x_c1, x_c2, exercise_id):
        """
        x_c0, x_c1, x_c2 : (B, T, J, 6)  — one tensor per camera
        exercise_id       : (B,)
        returns           : (B,)  quality scores in [1, 5]
        """
        B, T, J, C = x_c0.shape

        # ── 1. Reformat to (B, C, T, J) for Conv2d ───────────────────────
        f0 = self._prepare_camera(x_c0)   # (B, 6, T, J)
        f1 = self._prepare_camera(x_c1)
        f2 = self._prepare_camera(x_c2)

        # ── 2. Per-camera encoders → (B, 64, T, J) each ──────────────────
        e0 = self.encoder_c0(f0)          # (B, 64, T, J)
        e1 = self.encoder_c1(f1)
        e2 = self.encoder_c2(f2)

        # ── 3. Concatenate along feature dim → (B, 192, T, J) ────────────
        fused = torch.cat([e0, e1, e2], dim=1)

        # ── 4. Fusion projection → (B, PROJ_DIM=64, T, J) ────────────────
        x = self.fusion_proj(fused)       # (B, 64, T, J)

        # ── 5. Data BN: (B, 64, T, J) → (B, 64*J, T) → BN → back ────────
        x = x.permute(0, 1, 3, 2).reshape(B, PROJ_DIM * J, T)   # (B, 64*J, T)
        x = self.data_bn(x)
        x = x.reshape(B, PROJ_DIM, J, T).permute(0, 1, 3, 2)    # (B, 64, T, J)

        # ── 6. Nine ST-GCN blocks ─────────────────────────────────────────
        for block in self.blocks:
            x = block(x, self.A)          # (B, C', T', J)

        # ── 7. Global average pool → (B, 256) ────────────────────────────
        x = self.gap(x).squeeze(-1).squeeze(-1)

        # ── 8. Exercise embedding + regression head ───────────────────────
        ex  = self.ex_embed(exercise_id)                           # (B, 32)
        h   = torch.cat([x, ex], dim=1)                           # (B, 288)
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# ── Sanity check ──────────────────────────────────────────────────────────
_dummy_c0 = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_c1 = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_c2 = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_ex = torch.zeros(2, dtype=torch.long)
_model    = STGCN_MidFusion()
_out      = _model(_dummy_c0, _dummy_c1, _dummy_c2, _dummy_ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
print(f'\n✓ STGCN_MidFusion sanity check passed — output shape: {_out.shape}')
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters: {total_params:,}')
del _dummy_c0, _dummy_c1, _dummy_c2, _dummy_ex, _model, _out

print('\n✓ STGCN_MidFusion defined')
print(f'  CameraEncoder : {IN_FEATURES} → {ENCODER_OUT_DIM}  ×3 independent encoders')
print(f'  FusionProj    : {ENCODER_OUT_DIM*3} → {PROJ_DIM}')
print(f'  ST-GCN input  : {PROJ_DIM} features per joint')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch  (multi-view version)
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    x: (B, T, J, 6) — first 3 = position, last 3 = velocity
    Centre on mid-hip, scale by torso height.
    Identical to single-view version; called once per camera tensor.
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
        for skel_c0, skel_c1, skel_c2, qualities, exercise_ids in loader:
            # Normalise each camera independently (same transform, separate scales)
            skel_c0      = centre_and_scale(skel_c0.to(DEVICE))
            skel_c1      = centre_and_scale(skel_c1.to(DEVICE))
            skel_c2      = centre_and_scale(skel_c2.to(DEVICE))
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)

            preds = model(skel_c0, skel_c1, skel_c2, exercise_ids)
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
# Cell 13.5 — Split quality distribution audit
# ══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Quality score distribution audit (multi-view groups)")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Quality Score Distributions per Split", fontsize=14, fontweight='bold')

splits      = [('Train', train_mv), ('Val', val_mv), ('Test', test_mv)]
trial_types = [('Correct (T≤2)',   lambda d: d[d['trial_num'] <= 2]),
               ('Erroneous (T≥3)', lambda d: d[d['trial_num'] >= 3])]

for col, (split_name, split_df) in enumerate(splits):
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
save_path = os.path.join(PLOTS_DIR, 'split_quality_distributions.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved → {save_path}")

print(f"\n{'Split':<8} {'Type':<12} {'n':>6} {'mean':>7} {'std':>7} "
      f"{'min':>6} {'25%':>6} {'50%':>6} {'75%':>6} {'max':>6}")
print("-" * 70)
for split_name, split_df in splits:
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
for split_name, split_df in splits:
    persons = sorted(split_df['person'].unique())
    print(f"  {split_name:<6}: {persons}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers  (identical to single-view, no changes needed)
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

print('✓ Plotting helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Early Stopping  (unchanged)
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

print('✓ EarlyStopping defined (monitoring MAE)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Training Loop  *** MID-FUSION VERSION ***
# ══════════════════════════════════════════════════════════════════════════

reg_fn = nn.SmoothL1Loss(beta=1.0)


def make_weighted_sampler(df):
    """
    Weight each sample inversely proportional to quality score.
    Erroneous trials (trial_num >= 3) get 2× weight.
    """
    q = df['quality'].values.astype(np.float32)
    raw_weights    = (q.max() + 1.0) - q
    erroneous_mask = (df['trial_num'].values >= 3).astype(np.float32)
    raw_weights   *= (1.0 + erroneous_mask)
    weights_tensor = torch.DoubleTensor(raw_weights)
    return torch.utils.data.WeightedRandomSampler(
        weights     = weights_tensor,
        num_samples = len(df),
        replacement = True,
    )


train_sampler = make_weighted_sampler(train_mv)

train_loader = DataLoader(
    BZUDatasetMultiView(train_mv, augment=True),
    batch_size  = BATCH_SIZE,
    sampler     = train_sampler,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)
val_loader = DataLoader(
    BZUDatasetMultiView(val_mv, augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)
test_loader = DataLoader(
    BZUDatasetMultiView(test_mv, augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)

model = STGCN_MidFusion(K=3, dropout=0.5).to(DEVICE)

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

log.info('=' * 70)
log.info('STARTING MID-FUSION MULTI-VIEW REGRESSION TRAINING')
log.info(f'train={len(train_mv)} val={len(val_mv)} test={len(test_mv)} (groups)')
log.info(f'Model: STGCN_MidFusion  |  Cameras: C0+C1+C2  |  '
         f'Encoder: {IN_FEATURES}→{ENCODER_OUT_DIM}  |  Proj: {PROJ_DIM}')
log.info('=' * 70)

print(f'\n{"═"*68}')
print(f'  Mid-Fusion ST-GCN  |  C0 + C1 + C2')
print(f'  Train: {len(train_mv)} groups  Val: {len(val_mv)}  Test: {len(test_mv)}')
print(f'  Encoder: {IN_FEATURES}→{ENCODER_OUT_DIM}  Fusion: {ENCODER_OUT_DIM*3}→{PROJ_DIM}')
print(f'  Patience: {PATIENCE}  |  LR: {LR}  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, reg_fn, is_train=True,  optimiser=optimiser)
    vl = run_epoch(model, val_loader,   reg_fn, is_train=False)
    scheduler.step(vl['mae'])

    for split, res in [('train', tr), ('val', vl)]:
        history[f'{split}_loss'].append(res['loss'])
        history[f'{split}_rmse'].append(res['rmse'])
        history[f'{split}_mae'].append(res['mae'])
        history[f'{split}_r2'].append(res['r2'])
        history[f'{split}_pcc'].append(res['pcc'])

    stop, improved = early_stop.step(vl['mae'], model, epoch)

    star = ' ★' if improved else ''
    msg  = (f'  Ep {epoch:3d}/{EPOCHS} | '
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

# ── Final test evaluation (best weights only) ─────────────────────────────
final_te = run_epoch(model, test_loader, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')
print(f'  PCC  : {final_te["pcc"]:.4f}')

# ── Collect predictions for scatter plot ──────────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for skel_c0, skel_c1, skel_c2, qualities, exercise_ids in test_loader:
        skel_c0      = centre_and_scale(skel_c0.to(DEVICE))
        skel_c1      = centre_and_scale(skel_c1.to(DEVICE))
        skel_c2      = centre_and_scale(skel_c2.to(DEVICE))
        exercise_ids = exercise_ids.to(DEVICE)
        preds        = model(skel_c0, skel_c1, skel_c2, exercise_ids)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# ── Save all plots ────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR,  test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR,     test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR,           test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR,          test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')

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
# Cell 16.5 — Per-exercise test metrics
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
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
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

# ── Per-exercise bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Per-Exercise Test Metrics (Mid-Fusion ST-GCN)',
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

npz_per_ex_path = os.path.join(LOGS_DIR, 'per_exercise_metrics.npz')
np.savez(npz_per_ex_path,
         exercise_ids = np.array(unique_exercises),
         n            = np.array([per_ex_results[e]['n']    for e in unique_exercises]),
         mae          = np.array([per_ex_results[e]['mae']  for e in unique_exercises]),
         rmse         = np.array([per_ex_results[e]['rmse'] for e in unique_exercises]),
         r2           = np.array([per_ex_results[e]['r2']   for e in unique_exercises]),
         pcc          = np.array([per_ex_results[e]['pcc']  for e in unique_exercises]))
print(f'  ✓ Per-exercise metrics NPZ → {npz_per_ex_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv            = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 60)
print('  TRAINING SUMMARY — ST-GCN Mid-Fusion (C0 + C1 + C2)')
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
