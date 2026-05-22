# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR     = os.path.join(DATASET_DIR, "by_person")
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"

# ── Multi-View Early Fusion ───────────────────────────────────────────────
FUSION_MODE   = 'early'          # concatenate all camera features at input
ALL_CAMERAS   = [0, 1, 2]        # all three cameras used
N_CAMERAS     = len(ALL_CAMERAS)
IN_FEATURES   = 6 * N_CAMERAS    # 18 = 3 cameras × (3 pos + 3 vel)
# ─────────────────────────────────────────────────────────────────────────

NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
TARGET_FRAMES = 120
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 300
LR            = 1e-4
BATCH_SIZE    = 48
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_EarlyFusion_MultiView_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

print('✓ Configuration loaded  [EARLY FUSION MULTI-VIEW]')
print(f'  DATASET_DIR  : {DATASET_DIR}')
print(f'  SPLIT_DIR    : {SPLIT_DIR}')
print(f'  FUSION_MODE  : {FUSION_MODE}')
print(f'  ALL_CAMERAS  : {ALL_CAMERAS}')
print(f'  IN_FEATURES  : {IN_FEATURES}  ({N_CAMERAS} cameras × 6)')
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
    print('\n❌ No NPZ files found!')
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
log = logging.getLogger("GCN-EarlyFusion")
log.info("=" * 70)
log.info("ST-GCN Early Fusion Multi-View Regression | BZU Physiotherapy Dataset")
log.info(f"Cameras: {ALL_CAMERAS}  |  IN_FEATURES: {IN_FEATURES}  |  "
         f"Epochs: {EPOCHS}  |  Patience: {PATIENCE}")
log.info(f"Log file: {log_file}")
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
#           [CHANGED] Early Fusion: builds one row per (trial, segment)
#           with filepaths for ALL cameras, no single-camera filtering.
# ══════════════════════════════════════════════════════════════════════════

def build_index_from_split(split_name, df_csv, camera_id=None):
    """
    Scan one pre-split folder (train | valid | test), parse every NPZ filename,
    and merge quality labels from the CSV.
    camera_id=None loads all cameras (required for multi-view).
    """
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
    print(f'  Camera dist      :\n{df["camera"].value_counts().sort_index().to_string()}')
    return df


# ── [CHANGED] Keep all cameras; do NOT filter to a single camera ──────────
def filter_complete_camera_groups(df, label=''):
    """
    Keep only (trial_key, segment) groups that have ALL required cameras.
    For early fusion we KEEP all camera rows — the pivot step uses them.
    """
    REQUIRED_CAMERAS = set(ALL_CAMERAS)

    coverage   = df.groupby(['trial_key', 'segment'])['camera'].apply(set)
    complete   = coverage[coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))].index
    incomplete = coverage[~coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))]

    n_before = len(df)
    # Keep rows whose (trial_key, segment) pair has full coverage
    df = df[df.set_index(['trial_key', 'segment']).index.isin(complete)].reset_index(drop=True)
    n_skipped = n_before - len(df)

    if len(incomplete):
        print(f'  [{label}] ⚠️  Skipped {n_skipped} rows from '
              f'{len(incomplete)} groups with incomplete camera coverage')
    else:
        print(f'  [{label}] ✓ All groups have complete {len(REQUIRED_CAMERAS)}-camera coverage')

    # NOTE: we do NOT filter to a single CAMERA_ID here — early fusion uses all cameras
    print(f'  [{label}] Rows after filter (all cameras): {len(df)}')
    return df


# ── [NEW] Pivot to one row per (trial_key, segment) with 3 camera paths ──
def build_multiview_index(df, label=''):
    """
    Transform a per-camera DataFrame into a per-(trial, segment) DataFrame
    suitable for early fusion.

    Output columns:
        trial_key, segment, exercise, person, trial_id, trial_num,
        quality, split, filepath_C0, filepath_C1, filepath_C2
    """
    REQUIRED_CAMERAS = set(ALL_CAMERAS)
    records = []

    for (trial_key, segment), group in df.groupby(['trial_key', 'segment']):
        cams_present = set(group['camera'].values)
        if not REQUIRED_CAMERAS.issubset(cams_present):
            continue                          # should not happen after filter step

        base_row = group.iloc[0]
        record   = {
            'trial_key' : trial_key,
            'segment'   : segment,
            'exercise'  : base_row['exercise'],
            'person'    : base_row['person'],
            'trial_id'  : base_row['trial_id'],
            'trial_num' : base_row['trial_num'],
            'quality'   : base_row['quality'],
            'split'     : base_row['split'],
        }
        for cam in ALL_CAMERAS:
            cam_row = group[group['camera'] == cam]
            record[f'filepath_C{cam}'] = cam_row.iloc[0]['filepath']

        records.append(record)

    mv_df = pd.DataFrame(records).reset_index(drop=True)
    print(f'  [{label}] Multi-view samples (one row per trial×segment): {len(mv_df)}')
    print(f'  [{label}] Quality mean±std : {mv_df["quality"].mean():.3f} '
          f'± {mv_df["quality"].std():.3f}')
    return mv_df


# ── [CHANGED] Remove corrupted: check all camera files per row ────────────
def remove_corrupted(df, label=''):
    """Drop rows where ANY camera file cannot be loaded."""
    bad_indices = []
    for idx, row in df.iterrows():
        for cam in ALL_CAMERAS:
            if load_skeleton(row[f'filepath_C{cam}']) is None:
                bad_indices.append(idx)
                break
    if bad_indices:
        print(f'  [{label}] Removing {len(bad_indices)} rows with corrupted file(s)')
        df = df.drop(index=bad_indices).reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


# ── Build per-camera splits ───────────────────────────────────────────────
train_raw = build_index_from_split('train', df_csv, camera_id=None)
val_raw   = build_index_from_split('valid', df_csv, camera_id=None)
test_raw  = build_index_from_split('test',  df_csv, camera_id=None)

# Filter: keep only groups with all cameras present
train_raw = filter_complete_camera_groups(train_raw, 'TRAIN')
val_raw   = filter_complete_camera_groups(val_raw,   'VALID')
test_raw  = filter_complete_camera_groups(test_raw,  'TEST')

# Pivot to multi-view index (one row per trial × segment)
print('\nBuilding multi-view index (early fusion)...')
train_df = build_multiview_index(train_raw, 'TRAIN')
val_df   = build_multiview_index(val_raw,   'VALID')
test_df  = build_multiview_index(test_raw,  'TEST')

# Remove corrupted rows
print('\nChecking for corrupted files...')
train_df = remove_corrupted(train_df, 'TRAIN')
val_df   = remove_corrupted(val_df,   'VALID')
test_df  = remove_corrupted(test_df,  'TEST')

# Combine for shared analysis
df_index = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Sanity: no trial_key leaks across splits
tr_keys = set(train_df['trial_key'])
vl_keys = set(val_df['trial_key'])
te_keys = set(test_df['trial_key'])
assert tr_keys.isdisjoint(vl_keys),  'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys),  'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys),  'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

print(f'\n{"═"*68}')
print(f'  {"Split":<8} {"Samples":>8} {"Correct":>9} {"Erroneous":>11} '
      f'{"Q mean":>8} {"Q std":>7} {"Q min":>7} {"Q max":>7}')
print(f'  {"─"*66}')
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    cor = (d['trial_num'] <= 2).sum()
    err = (d['trial_num'] >= 3).sum()
    q   = d['quality']
    print(f'  {name:<8} {len(d):>8} {cor:>9} {err:>11} '
          f'{q.mean():>8.3f} {q.std():>7.3f} '
          f'{q.min():>7.3f} {q.max():>7.3f}')
print(f'{"═"*68}')
print('\n✓ Early-fusion multi-view index ready')


def augment_with_mirrors(df):
    """Double the dataset with mirrored copies."""
    mirrored      = df.copy()
    mirrored['mirrored'] = True
    df            = df.copy()
    df['mirrored']       = False
    return pd.concat([df, mirrored], ignore_index=True)


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


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.7 — Frame length distribution (sample from one camera)
# ══════════════════════════════════════════════════════════════════════════
lengths       = []
sample_rows   = df_index.sample(min(500, len(df_index)), random_state=42)

for _, row in sample_rows.iterrows():
    skel = load_skeleton(row['filepath_C0'])
    if skel is not None:
        lengths.append(skel.shape[0])

lengths = np.array(lengths)
print(f"\nFrame length distribution (sample of {len(lengths)} files via C0):")
print(f"  min={lengths.min()}  max={lengths.max()}  "
      f"mean={lengths.mean():.1f}  median={np.median(lengths):.1f}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Skeleton visualisation helpers  (unchanged)
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
        (axes[0], x,  y,  'Front View  (X–Y)', 'X', 'Y', False),
        (axes[1], z,  y,  'Side View   (Z–Y)', 'Z', 'Y', False),
        (axes[2], x, -z,  'Top View    (X–Z)', 'X', '-Z', False),
    ]
    for ax, hx, hy, view_title, xlabel, ylabel, _ in views:
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
    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Skeleton plot saved → {save_path}')
    plt.close()

print('✓ Skeleton visualisation helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — Visualise a sample skeleton (using C0)
# ══════════════════════════════════════════════════════════════════════════
idx         = 10
sample_row  = df_index.iloc[idx]
sample_skel = load_skeleton(sample_row['filepath_C0'])

print(sample_row[['person', 'exercise', 'trial_id', 'segment',
                   'filepath_C0', 'filepath_C1', 'filepath_C2']])
print(f'Skeleton shape (C0): {sample_skel.shape}')

plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton C0 · {sample_row['trial_key']} · E{sample_row['exercise']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
)


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset  [CHANGED — Early Fusion Multi-View]
#
# Each __getitem__ loads 3 camera skeletons, applies CONSISTENT augmentation
# across all cameras, then concatenates along the feature axis:
#
#   (T, J, 6_C0) + (T, J, 6_C1) + (T, J, 6_C2)  →  (T, J, 18)
#
# The model receives 18 input features = early fusion of 3 views.
# ══════════════════════════════════════════════════════════════════════════
MIRROR_PAIRS = [
    (1, 4),   # R-Hip   ↔ L-Hip
    (2, 5),   # R-Knee  ↔ L-Knee
    (3, 6),   # R-Ankle ↔ L-Ankle
    (11, 14), # L-Shoulder ↔ R-Shoulder
    (12, 15), # L-Elbow    ↔ R-Elbow
    (13, 16), # L-Wrist    ↔ R-Wrist
]

def mirror_skeleton(skel):
    skel = skel.copy()
    skel[:, :, 0] *= -1
    for i, j in MIRROR_PAIRS:
        skel[:, [i, j], :] = skel[:, [j, i], :]
    return skel


class BZUDataset(Dataset):
    """
    Early-fusion multi-view dataset.
    Returns (fused_skeleton, quality_score, exercise_id)
    where fused_skeleton has shape (T, J, 18) = 3 cameras × 6 features.
    """
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        do_mirror = row.get('mirrored', False)

        # ── Sample shared augmentation parameters once ────────────────────
        # All cameras must undergo the SAME temporal transformation so
        # the time axes remain aligned after early fusion.
        aug_params = self._sample_aug_params() if self.augment else None

        cam_features = []
        for cam in ALL_CAMERAS:
            skel = load_skeleton(row[f'filepath_C{cam}'])
            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3), dtype=np.float32)

            skel = self._normalise_length(skel)

            # Mirror all cameras consistently (same left/right swap)
            if do_mirror:
                skel = mirror_skeleton(skel)

            # Apply consistent temporal augmentation
            if aug_params is not None:
                skel = self._apply_aug(skel, aug_params, cam)

            # Velocity features
            velocity     = np.zeros_like(skel)
            velocity[1:] = skel[1:] - skel[:-1]

            cam_features.append(
                np.concatenate([skel, velocity], axis=-1)   # (T, J, 6)
            )

        # Early fusion: concatenate along feature axis → (T, J, 18)
        fused = np.concatenate(cam_features, axis=-1)

        skel_tensor  = torch.tensor(fused,           dtype=torch.float32)
        quality      = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id  = torch.tensor(row['exercise'], dtype=torch.long)
        return skel_tensor, quality, exercise_id

    # ── Helpers ───────────────────────────────────────────────────────────
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

    def _sample_aug_params(self):
        """
        Sample random parameters that will be applied IDENTICALLY to every
        camera view so that temporal alignment is preserved.
        """
        T = self.target_frames
        speed      = np.random.uniform(0.75, 1.25)
        n_new      = max(10, int(T * speed))
        warp_idxs  = np.linspace(0, T - 1, n_new).astype(int)

        scale      = np.random.uniform(0.9, 1.1)

        crop_ratio = np.random.uniform(0.80, 1.0)
        start      = np.random.randint(0, max(1, int(T * (1 - crop_ratio))))
        end        = start + int(T * crop_ratio)

        return dict(warp_idxs=warp_idxs, scale=scale,
                    crop_start=start, crop_end=end)

    def _apply_aug(self, skel, params, cam_id):
        """
        Apply augmentation using shared temporal params + per-camera noise.

        Shared (must be identical across cameras):
          - Temporal speed warp
          - Uniform scale
          - Temporal crop

        Per-camera (independent noise is fine):
          - Small Gaussian joint jitter
        """
        # 1. Temporal warp (shared)
        skel = self._normalise_length(skel[params['warp_idxs']])

        # 2. Per-camera jitter (independent — mimics sensor noise per view)
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.005

        # 3. Uniform scale (shared)
        skel *= params['scale']

        # 4. Temporal crop (shared)
        skel = self._normalise_length(
            skel[params['crop_start']:params['crop_end']]
        )
        return skel


print('✓ BZUDataset defined  [Early Fusion Multi-View, in_features=18]')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — TRUE ST-GCN  [CHANGED: in_features=18]
# ══════════════════════════════════════════════════════════════════════════
from collections import deque


def get_joint_distances(num_joints, edges, center_joint=0):
    adj  = {i: [] for i in range(num_joints)}
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
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
        for src, dst in [(i, j), (j, i)]:
            if dist[dst] < dist[src]:
                A[1, src, dst] = 1.0
            elif dist[dst] > dist[src]:
                A[2, src, dst] = 1.0
            else:
                A[1, src, dst] = 1.0

    for k in range(3):
        row_sum  = A[k].sum(axis=1)
        d_inv_sq = np.where(row_sum > 0, np.power(row_sum, -0.5), 0.0)
        D        = np.diag(d_inv_sq)
        A[k]     = D @ A[k] @ D

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
        x   = self.conv(x)
        x   = x.view(B, self.K, -1, T, J)
        out = torch.einsum('bkctj,kjv->bctv', x, A + self.M)
        return self.relu(self.bn(out))


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


NUM_EXERCISES = 10

class STGCN_Regression(nn.Module):
    """
    True ST-GCN for Early Fusion Multi-View regression.

    [CHANGED] in_features=18 instead of 6:
        channels 0–5   : camera C0  (pos_xyz + vel_xyz)
        channels 6–11  : camera C1  (pos_xyz + vel_xyz)
        channels 12–17 : camera C2  (pos_xyz + vel_xyz)

    The spatial graph and all other layers are identical to the
    single-view version; only the input width changes.
    """
    def __init__(self, in_features=IN_FEATURES, K=3, dropout=0.5):
        super().__init__()
        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer('A', A)

        # data_bn now handles in_features * J = 18 * 17 = 306 channels
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

        self.gap      = nn.AdaptiveAvgPool2d(1)
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
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, exercise_id):
        """
        x           : (B, T, J, 18)   ← 3 cameras early-fused
        exercise_id : (B,)
        returns     : (B,)  quality in [1, 5]
        """
        B, T, J, C = x.shape

        x = x.permute(0, 3, 2, 1).reshape(B, C * J, T)   # (B, 18*17, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, J, T).permute(0, 1, 3, 2)    # (B, 18, T, J)

        for block in self.blocks:
            x = block(x, self.A)

        x   = self.gap(x).squeeze(-1).squeeze(-1)         # (B, 256)
        ex  = self.ex_embed(exercise_id)                   # (B, 32)
        h   = torch.cat([x, ex], dim=1)                   # (B, 288)
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# Sanity check
_dummy_x  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_ex = torch.zeros(2, dtype=torch.long)
_model    = STGCN_Regression(in_features=IN_FEATURES)
_out      = _model(_dummy_x, _dummy_ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'\n✓ Early-fusion ST-GCN sanity check passed — output: {_out.shape}')
print(f'✓ in_features = {IN_FEATURES}  ({N_CAMERAS} cameras × 6)')
print(f'✓ Total trainable parameters: {total_params:,}')
del _dummy_x, _dummy_ex, _model, _out


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch
#            [CHANGED] centre_and_scale normalises each camera independently
# ══════════════════════════════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    x: (B, T, J, 18)  — 3 cameras × 6 features (pos_xyz + vel_xyz)

    Each camera block is normalised independently using its own hip
    centre and torso height.  This ensures each view is self-consistent
    regardless of its absolute position in the world frame.

    Layout:  [ C0_pos(3) | C0_vel(3) | C1_pos(3) | C1_vel(3) | C2_pos(3) | C2_vel(3) ]
    """
    normalised_cams = []
    for cam_idx in range(N_CAMERAS):
        offset_pos = cam_idx * 6          # start of position channels for this cam
        offset_vel = offset_pos + 3       # start of velocity channels

        pos = x[:, :, :, offset_pos : offset_pos + 3]   # (B, T, J, 3)
        vel = x[:, :, :, offset_vel : offset_vel + 3]   # (B, T, J, 3)

        # Translate: hip = midpoint of right-hip (1) and left-hip (4)
        hip   = (pos[:, :, 1:2, :] + pos[:, :, 4:5, :]) / 2.0
        pos   = pos - hip

        # Scale: torso height = distance from hip to mid-shoulder
        shoulder = (pos[:, :, 11:12, :] + pos[:, :, 14:15, :]) / 2.0
        torso_h  = shoulder[:, :, :, 1:2].abs().mean(
            dim=1, keepdim=True).clamp(min=1e-6)          # (B, 1, 1, 1)
        pos = pos / torso_h
        vel = vel / torso_h

        normalised_cams.append(torch.cat([pos, vel], dim=-1))   # (B, T, J, 6)

    return torch.cat(normalised_cams, dim=-1)   # (B, T, J, 18)


def run_epoch(model, loader, optimiser, reg_fn, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, qualities, exercise_ids in loader:
            skels        = centre_and_scale(skels.to(DEVICE))  # (B,T,J,18)
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)

            preds = model(skels, exercise_ids)
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

print('✓ centre_and_scale (per-camera) and run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13.5 — Split quality distribution audit  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Quality score distribution audit")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Quality Score Distributions per Split", fontsize=14, fontweight='bold')

splits      = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
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
        ax.set_xlabel('Quality Score'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, 'split_quality_distributions.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers  (unchanged)
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
    if test_loss is not None: _add_test_line(ax, test_loss, 'Loss')
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
        if test_val is not None: _add_test_line(ax, test_val, title)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'rmse_mae.png'))

def plot_r2(history, save_dir, test_r2=None):
    epochs = range(1, len(history['val_r2']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    if test_r2 is not None: _add_test_line(ax, test_r2, 'R²')
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
    if test_pcc is not None: _add_test_line(ax, test_pcc, 'PCC')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='No correlation')
    ax.set_title('Pearson Correlation Coefficient', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('PCC')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'pcc_curve.png'))

def plot_regression_scatter(q_true, q_pred, split_name='Test', save_dir=None):
    qt   = np.array(q_true);   qp = np.array(q_pred)
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
    ax.text(0.05, 0.95,
            f'R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
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

print('✓ EarlyStopping defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Training Loop  [CHANGED: model uses IN_FEATURES=18]
# ══════════════════════════════════════════════════════════════════════════
reg_fn = nn.SmoothL1Loss(beta=1.0)


def make_ds(df, aug):
    return BZUDataset(df, augment=aug)


def make_weighted_sampler(df):
    q              = df['quality'].values.astype(np.float32)
    raw_weights    = (q.max() + 1.0) - q
    erroneous_mask = (df['trial_num'].values >= 3).astype(np.float32)
    raw_weights   *= (1.0 + erroneous_mask)
    return torch.utils.data.WeightedRandomSampler(
        weights     = torch.DoubleTensor(raw_weights),
        num_samples = len(df),
        replacement = True,
    )


# Mirror augmentation doubles training data
train_df_augmented = augment_with_mirrors(train_df)
print(f'Train before mirroring: {len(train_df)}')
print(f'Train after  mirroring: {len(train_df_augmented)}')

train_sampler = make_weighted_sampler(train_df_augmented)

train_loader = DataLoader(
    make_ds(train_df_augmented, aug=True),
    batch_size = BATCH_SIZE,
    sampler    = train_sampler,
    num_workers= 0,
    pin_memory = (DEVICE == 'cuda'),
)
val_loader  = DataLoader(make_ds(val_df,  False), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=(DEVICE=='cuda'))
test_loader = DataLoader(make_ds(test_df, False), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=(DEVICE=='cuda'))

# [CHANGED] in_features=IN_FEATURES (18) instead of 6
model = STGCN_Regression(in_features=IN_FEATURES, K=3, dropout=0.5).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=50, min_lr=1e-6, verbose=True)

early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val']
METRICS = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info(f'EARLY FUSION MULTI-VIEW REGRESSION  |  in_features={IN_FEATURES}')
log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')
log.info('=' * 70)

print(f'\n{"═"*68}')
print(f'  [EARLY FUSION]  Cameras: {ALL_CAMERAS}  |  in_features={IN_FEATURES}')
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Patience: {PATIENCE}  |  LR: 1e-4  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, optimiser, reg_fn, is_train=True)
    vl = run_epoch(model, val_loader,   optimiser, reg_fn, is_train=False)

    for split, res in [('train', tr), ('val', vl)]:
        for m in METRICS:
            history[f'{split}_{m}'].append(res[m])

    scheduler.step(vl['mae'])
    stop, improved = early_stop.step(vl['mae'], model, epoch)

    star = ' ★' if improved else ''
    msg  = (f'  Ep {epoch:3d}/{EPOCHS} | '
            f'Tr loss={tr["loss"]:.4f} mae={tr["mae"]:.3f} '
            f'r2={tr["r2"]:.3f} pcc={tr["pcc"]:.3f} | '
            f'Vl loss={vl["loss"]:.4f} mae={vl["mae"]:.3f} '
            f'r2={vl["r2"]:.3f} pcc={vl["pcc"]:.3f} | '
            f'ES {early_stop.counter}/{PATIENCE}{star}')
    print(msg); log.info(msg)

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

model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch

final_te = run_epoch(model, test_loader, optimiser, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')
print(f'  PCC  : {final_te["pcc"]:.4f}')

# Collect test predictions
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for skels, qualities, exercise_ids in test_loader:
        skels        = centre_and_scale(skels.to(DEVICE))
        exercise_ids = exercise_ids.to(DEVICE)
        preds        = model(skels, exercise_ids)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# Plots
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR,    test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR,          test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR,         test_pcc=final_te['pcc'])
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
# Cell 16.5 — Per-exercise test metrics  (unchanged logic)
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

per_ex_df = pd.DataFrame([
    {'exercise': f'E{ex}', **vals} for ex, vals in per_ex_results.items()])
per_ex_csv = os.path.join(LOGS_DIR, 'per_exercise_metrics.csv')
per_ex_df.to_csv(per_ex_csv, index=False)
print(f'\n  ✓ Per-exercise CSV → {per_ex_csv}')

# Per-exercise scatter grid
n_ex   = len(unique_exercises)
n_cols = 3
n_rows = int(np.ceil(n_ex / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
axes = axes.flatten()
fig.suptitle('Per-Exercise: True vs Predicted Quality (Test Set)',
             fontsize=14, fontweight='bold')

for i, ex_id in enumerate(unique_exercises):
    ax   = axes[i]
    mask = all_ex_arr == ex_id
    qt   = all_true_q_arr[mask]; qp = all_pred_q_arr[mask]
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

np.savez(os.path.join(LOGS_DIR, 'per_exercise_metrics.npz'),
         exercise_ids = np.array(unique_exercises),
         n    = np.array([per_ex_results[e]['n']    for e in unique_exercises]),
         mae  = np.array([per_ex_results[e]['mae']  for e in unique_exercises]),
         rmse = np.array([per_ex_results[e]['rmse'] for e in unique_exercises]),
         r2   = np.array([per_ex_results[e]['r2']   for e in unique_exercises]),
         pcc  = np.array([per_ex_results[e]['pcc']  for e in unique_exercises]))


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════
bv            = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 60)
print('  TRAINING SUMMARY — ST-GCN Early Fusion Multi-View')
print(f'  Fusion      : {FUSION_MODE.upper()}  |  Cameras: {ALL_CAMERAS}')
print(f'  in_features : {IN_FEATURES}  ({N_CAMERAS} × 6)')
print('=' * 60)
print(f'  Best Epoch   : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE : {best_val_mae:.4f}')
print(f'  Best Val RMSE: {best_val_rmse:.4f}')
print(f'  Best Val R²  : {best_val_r2:.4f}')
print(f'  Best Val PCC : {best_val_pcc:.4f}')
print('─' * 60)
print(f'  Test MAE     : {final_te["mae"]:.4f}')
print(f'  Test RMSE    : {final_te["rmse"]:.4f}')
print(f'  Test R²      : {final_te["r2"]:.4f}')
print(f'  Test PCC     : {final_te["pcc"]:.4f}')
print('=' * 60)

log.info(f'[EarlyFusion] Best Epoch={best_epoch}  stopped={stopped_epoch}')
log.info(f'[EarlyFusion] Test MAE={final_te["mae"]:.4f}')
log.info(f'[EarlyFusion] Test RMSE={final_te["rmse"]:.4f}')
log.info(f'[EarlyFusion] Test R²={final_te["r2"]:.4f}')
log.info(f'[EarlyFusion] Test PCC={final_te["pcc"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [{'split': 'test_overall', 'exercise': 'all',
         'fusion': FUSION_MODE, 'cameras': str(ALL_CAMERAS),
         'in_features': IN_FEATURES,
         'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
         'val_mae': best_val_mae,  'val_rmse': best_val_rmse,
         'val_r2':  best_val_r2,   'val_pcc':  best_val_pcc,
         'test_mae': final_te['mae'], 'test_rmse': final_te['rmse'],
         'test_r2':  final_te['r2'], 'test_pcc':  final_te['pcc']}]

for ex, vals in per_ex_results.items():
    rows.append({'split': 'test_per_exercise', 'exercise': f'E{ex}',
                 'fusion': FUSION_MODE,
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
