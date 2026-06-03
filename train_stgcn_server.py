# ══════════════════════════════════════════════════════════════════════════
# ST-GCN Multi-View EARLY FUSION — Regression
#
# Early Fusion Strategy:
#   For each trial, all 3 camera views (C0, C1, C2) are loaded and their
#   skeleton sequences are concatenated along the feature dimension BEFORE
#   entering the network:
#
#     Single view  : (B, T, J, 6)           ← 6 = 3 pos + 3 vel
#     Early Fusion : (B, T, J, 6 × 3=18)    ← concat C0, C1, C2 features
#
#   The ST-GCN then operates on the 18-channel fused input.
#   The adjacency graph structure (17 joints) stays identical.
#
# Key differences from single-view script:
#   1. build_multiview_index()  → groups files by trial_key + camera
#   2. MultiViewDataset         → loads 3 npz files, fuses them
#   3. STGCN_Regression         → in_features=18 instead of 6
#   4. filter_complete_camera_groups() is now MANDATORY (not optional)
# ══════════════════════════════════════════════════════════════════════════


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
BATCH_SIZE    = 32          # ← reduced: each sample now loads 3× npz files
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_MultiView_EarlyFusion_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Exercise Filter ───────────────────────────────────────────────────────
EXCLUDED_EXERCISES = {3, 7, 9}
EXERCISE_REMAP     = {}

# ── Multi-view specific ───────────────────────────────────────────────────
ALL_CAMERAS    = [0, 1, 2]      # cameras required for every trial
NUM_VIEWS      = len(ALL_CAMERAS)
IN_FEATURES    = 6 * NUM_VIEWS  # 18 after early fusion

print('✓ Configuration loaded  [MULTI-VIEW EARLY FUSION]')
print(f'  DATASET_DIR  : {DATASET_DIR}')
print(f'  SPLIT_DIR    : {SPLIT_DIR}')
print(f'  ALL_CAMERAS  : {ALL_CAMERAS}')
print(f'  IN_FEATURES  : {IN_FEATURES}  (6 per camera × {NUM_VIEWS} cameras)')
print(f'  EXISTS       : {os.path.exists(DATASET_DIR)}')


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
print(f'✓ Run directory: {RUN_DIR}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 3 — Filename parser & skeleton loader  (unchanged from single-view)
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
# Cell 3.5 — Explore dataset folder & one NPZ file
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
    raise FileNotFoundError(f'❌ CSV not loaded from: {CSV_PATH}')

print(f'Columns : {df_csv.columns.tolist()}')
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

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger("STGCN-MultiView-EarlyFusion")
log.info("=" * 70)
log.info("ST-GCN Multi-View Early Fusion Regression | BZU Physiotherapy")
log.info(f"Cameras: {ALL_CAMERAS}  |  IN_FEATURES={IN_FEATURES}  |  Epochs={EPOCHS}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Skeleton visualisation helpers  (identical to single-view)
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
# Cell 7 — Build MULTI-VIEW index from pre-split directories
#
# Key change vs single-view:
#   ─ We keep camera_id=None (load ALL cameras)
#   ─ filter_complete_camera_groups() is REQUIRED, not optional
#   ─ build_multiview_trial_index() collapses per-file rows into
#     per-trial rows, storing one filepath per camera:
#       { trial_key, exercise, person, quality, cam0, cam1, cam2 }
# ══════════════════════════════════════════════════════════════════════════

def build_index_from_split(split_name, df_csv, camera_id=None):
    """Same as single-view: one row per file."""
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
        return pd.DataFrame()

    df = pd.DataFrame(records)

    correct_mean   = df.loc[df['trial_num'] <= 2, 'quality'].mean()
    erroneous_mean = df.loc[df['trial_num'] >= 3, 'quality'].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5

    df.loc[df['quality'].isna() & (df['trial_num'] <= 2), 'quality'] = correct_mean
    df.loc[df['quality'].isna() & (df['trial_num'] >= 3), 'quality'] = erroneous_mean

    print(f'  Samples : {len(df)}  |  Unique trials : {df["trial_key"].nunique()}')
    return df


def filter_complete_camera_groups(df, label=''):
    """Keep only trials that have ALL cameras. For multi-view this is mandatory."""
    REQUIRED_CAMERAS = set(ALL_CAMERAS)
    coverage   = df.groupby('trial_key')['camera'].apply(set)
    complete   = coverage[coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))].index
    incomplete = coverage[~coverage.apply(lambda s: REQUIRED_CAMERAS.issubset(s))]

    n_before = len(df)
    df = df[df['trial_key'].isin(complete)].reset_index(drop=True)

    if len(incomplete):
        print(f'  [{label}] ⚠️  Dropped {n_before - len(df)} files '
              f'from {len(incomplete)} incomplete trials')
    else:
        print(f'  [{label}] ✓ All trials have complete {ALL_CAMERAS} coverage')
    return df


def remove_corrupted_multiview(df, label=''):
    bad = []
    for fpath in df['filepath']:
        if load_skeleton(fpath) is None:
            bad.append(fpath)
    if bad:
        # Mark entire trials as bad
        bad_keys = df[df['filepath'].isin(bad)]['trial_key'].unique()
        print(f'  [{label}] Removing {len(bad_keys)} trials with corrupted files')
        df = df[~df['trial_key'].isin(bad_keys)].reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


def exclude_exercises(df, excluded=EXCLUDED_EXERCISES, label=''):
    before = len(df)
    df = df[~df['exercise'].isin(excluded)].reset_index(drop=True)
    print(f'  [{label}] Excluded E{sorted(excluded)} → '
          f'{before - len(df)} rows dropped, {len(df)} remain')
    return df


def build_multiview_trial_index(df):
    """
    Collapse per-file rows into per-trial rows.

    Output columns:
        trial_key, exercise, person, trial_num, quality, split,
        filepath_c0, filepath_c1, filepath_c2
    """
    records = []
    for trial_key, group in df.groupby('trial_key'):
        cam_map = {row['camera']: row['filepath']
                   for _, row in group.iterrows()}
        if not all(c in cam_map for c in ALL_CAMERAS):
            continue   # safety check (should already be filtered)

        row0 = group.iloc[0]
        rec  = {
            'trial_key' : trial_key,
            'exercise'  : row0['exercise'],
            'person'    : row0['person'],
            'trial_num' : row0['trial_num'],
            'trial_id'  : row0['trial_id'],
            'quality'   : row0['quality'],
            'split'     : row0['split'],
        }
        for c in ALL_CAMERAS:
            rec[f'filepath_c{c}'] = cam_map[c]
        records.append(rec)

    mv_df = pd.DataFrame(records).reset_index(drop=True)
    print(f'  → Multi-view trial index: {len(mv_df)} trials '
          f'(each covers cameras {ALL_CAMERAS})')
    return mv_df


# ── Build raw file-level splits ───────────────────────────────────────────
train_df_files = build_index_from_split('train', df_csv)
val_df_files   = build_index_from_split('valid', df_csv)
test_df_files  = build_index_from_split('test',  df_csv)

# ── Filter: only trials with all cameras ─────────────────────────────────
train_df_files = filter_complete_camera_groups(train_df_files, 'TRAIN')
val_df_files   = filter_complete_camera_groups(val_df_files,   'VALID')
test_df_files  = filter_complete_camera_groups(test_df_files,  'TEST')

# ── Remove corrupted ──────────────────────────────────────────────────────
print('\nChecking for corrupted files...')
train_df_files = remove_corrupted_multiview(train_df_files, 'TRAIN')
val_df_files   = remove_corrupted_multiview(val_df_files,   'VALID')
test_df_files  = remove_corrupted_multiview(test_df_files,  'TEST')

# ── Exclude exercises ─────────────────────────────────────────────────────
print('\nExcluding exercises...')
train_df_files = exclude_exercises(train_df_files, label='TRAIN')
val_df_files   = exclude_exercises(val_df_files,   label='VALID')
test_df_files  = exclude_exercises(test_df_files,  label='TEST')

# ── Remap exercise IDs ────────────────────────────────────────────────────
remaining_exercises = sorted(
    set(train_df_files['exercise'].unique()) |
    set(val_df_files['exercise'].unique())   |
    set(test_df_files['exercise'].unique())
)
EXERCISE_REMAP = {orig: new for new, orig in enumerate(remaining_exercises)}
print(f'\n  Exercise ID remap : {EXERCISE_REMAP}')

for df_ in [train_df_files, val_df_files, test_df_files]:
    df_['exercise'] = df_['exercise'].map(EXERCISE_REMAP)

# ── Collapse to trial-level multi-view index ──────────────────────────────
print('\nBuilding multi-view trial index...')
train_df = build_multiview_trial_index(train_df_files)
val_df   = build_multiview_trial_index(val_df_files)
test_df  = build_multiview_trial_index(test_df_files)

# ── Leakage check ─────────────────────────────────────────────────────────
tr_keys = set(train_df['trial_key'])
vl_keys = set(val_df['trial_key'])
te_keys = set(test_df['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('✓ No data-leakage detected across splits')

# ── Summary ───────────────────────────────────────────────────────────────
print(f'\n{"═"*68}')
print(f'  {"Split":<8} {"Trials":>8} {"Correct":>9} {"Erroneous":>11} '
      f'{"Q mean":>8} {"Q std":>7}')
print(f'  {"─"*66}')
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    cor = (d['trial_num'] <= 2).sum()
    err = (d['trial_num'] >= 3).sum()
    q   = d['quality']
    print(f'  {name:<8} {len(d):>8} {cor:>9} {err:>11} '
          f'{q.mean():>8.3f} {q.std():>7.3f}')
print(f'{"═"*68}')
print('\n✓ Multi-view trial index ready')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.5 — Camera Distribution Check
# ══════════════════════════════════════════════════════════════════════════

df_index = pd.concat([train_df_files, val_df_files, test_df_files], ignore_index=True)
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

lengths      = []
# sample from file-level index (before trial collapse)
sample_files = df_index['filepath'].sample(min(500, len(df_index)), random_state=42)

for fpath in sample_files:
    skel = load_skeleton(fpath)
    if skel is not None:
        lengths.append(skel.shape[0])

lengths = np.array(lengths)
print(f"Frame length distribution (sample of {len(lengths)} files):")
print(f"  min={lengths.min()}  max={lengths.max()}  "
      f"mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  std={lengths.std():.1f}")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p:3d}th = {np.percentile(lengths, p):.0f}")

print(f"\nValue counts (top 10):")
unique, counts = np.unique(lengths, return_counts=True)
for cnt, val in sorted(zip(counts, unique), reverse=True)[:10]:
    print(f"  {int(val):4d} frames → {cnt:4d} files ({cnt/len(lengths)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.8 — Split quality distribution audit
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Quality Score Distributions per Split (Multi-View)",
             fontsize=14, fontweight='bold')

splits_list  = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
trial_types  = [('Correct (T≤2)',   lambda d: d[d['trial_num'] <= 2]),
                ('Erroneous (T≥3)', lambda d: d[d['trial_num'] >= 3])]

for col, (split_name, split_df) in enumerate(splits_list):
    for row, (type_name, filter_fn) in enumerate(trial_types):
        ax  = axes[row][col]
        sub = filter_fn(split_df)
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{split_name} — {type_name}')
            continue
        q = sub['quality']
        ax.hist(q, bins=30,
                color='steelblue' if row == 0 else 'tomato',
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

print(f"\n{'Split':<8} {'Type':<12} {'n':>6} {'mean':>7} {'std':>7} "
      f"{'min':>6} {'25%':>6} {'50%':>6} {'75%':>6} {'max':>6}")
print("-" * 70)
for split_name, split_df in splits_list:
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
for split_name, split_df in splits_list:
    print(f"  {split_name:<6}: {sorted(split_df['person'].unique())}")

print("\nPer-person quality mean per split:")
for split_name, split_df in splits_list:
    print(f"\n  {split_name}:")
    print(split_df.groupby('person')['quality']
          .agg(['mean','std','count']).round(3).to_string())

print("\n" + "=" * 65)
print("Exercise balance across splits")
print("=" * 65)
ex_counts = pd.DataFrame({
    name: split_df['exercise'].value_counts().sort_index()
    for name, split_df in splits_list
})
ex_counts.index   = [f'E{i}' for i in ex_counts.index]
ex_counts.columns = ['Train', 'Val', 'Test']
ex_counts['Train%'] = (ex_counts['Train'] / ex_counts['Train'].sum() * 100).round(1)
ex_counts['Val%']   = (ex_counts['Val']   / ex_counts['Val'].sum()   * 100).round(1)
ex_counts['Test%']  = (ex_counts['Test']  / ex_counts['Test'].sum()  * 100).round(1)
print(ex_counts.to_string())


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.9 — Skeleton visualisation (sample from trial index)
# ══════════════════════════════════════════════════════════════════════════

SKELETON_EDGES_VIZ = SKELETON_EDGES
JOINT_COLORS_VIZ   = JOINT_COLORS
PART_COLOR_VIZ     = PART_COLOR


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
    for ax, hx, hy, view_title, xlabel, ylabel, invert_y in views:
        for (i, j) in SKELETON_EDGES_VIZ:
            ax.plot([hx[i], hx[j]], [hy[i], hy[j]], color='dimgray', lw=2, zorder=1)
        for part, idxs in JOINT_COLORS_VIZ.items():
            ax.scatter(hx[idxs], hy[idxs], c=PART_COLOR_VIZ[part], s=80, zorder=3,
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
        for (i, j) in SKELETON_EDGES_VIZ:
            ax.plot([x[i], x[j]], [y[i], y[j]], color='dimgray', lw=2)
        for part, pidxs in JOINT_COLORS_VIZ.items():
            ax.scatter(x[pidxs], y[pidxs], c=PART_COLOR_VIZ[part],
                       s=60, edgecolors='black', linewidths=0.4, zorder=3)
        ax.set_title(f'Frame {fi}', fontsize=9)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Motion frames saved → {save_path}')
    plt.close()


# ── Visualise one sample (C0 of first trial) ─────────────────────────────
sample_row  = train_df.iloc[10]
sample_skel = load_skeleton(sample_row['filepath_c0'])

print(f"Sample trial : {sample_row['trial_key']}")
print(f"Exercise     : E{sample_row['exercise']}")
print(f"Skeleton     : {sample_skel.shape}")
print(f"X range: [{sample_skel[:,:,0].min():.3f}, {sample_skel[:,:,0].max():.3f}]")
print(f"Y range: [{sample_skel[:,:,1].min():.3f}, {sample_skel[:,:,1].max():.3f}]")
print(f"Z range: [{sample_skel[:,:,2].min():.3f}, {sample_skel[:,:,2].max():.3f}]")

hip  = sample_skel[:, 0, :]
head = sample_skel[:, 10, :]
print(f"\nHip  mean XYZ: {hip.mean(axis=0)}")
print(f"Head mean XYZ: {head.mean(axis=0)}")
print(f"Difference  : {head.mean(axis=0) - hip.mean(axis=0)}")

plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton (C0) · {sample_row['trial_key']} · E{sample_row['exercise']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
)
plot_skeleton_frames(
    sample_skel, n_frames=5,
    title=f"Motion Sequence (C0) · {sample_row['trial_key']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_motion.png'),
)

# ── Camera angle check across 3 views ─────────────────────────────────────
print("\nCamera angle check (same trial, 3 cameras):")
for c in ALL_CAMERAS:
    skel = load_skeleton(sample_row[f'filepath_c{c}'])
    if skel is not None:
        print(f"  C{c}  Hip XYZ:{skel[:,0,:].mean(axis=0).round(3)}"
              f"  Head XYZ:{skel[:,10,:].mean(axis=0).round(3)}")
    else:
        print(f"  C{c}  [file not found]")


#
# Each __getitem__ loads 3 skeleton sequences (one per camera) and applies:
#   1. Temporal normalisation to TARGET_FRAMES  (per camera)
#   2. Velocity computation                     (per camera)
#   3. Centring & scaling                       (per camera, independently)
#   4. EARLY FUSION: concatenate along feature axis
#        C0: (T, J, 6) ──┐
#        C1: (T, J, 6) ──┼──cat(axis=-1)──→  (T, J, 18)
#        C2: (T, J, 6) ──┘
#
# The network receives a single (T, J, 18) tensor — no view dimension.
# ══════════════════════════════════════════════════════════════════════════

def normalise_length(skel, target_frames=TARGET_FRAMES):
    T = skel.shape[0]
    if T == target_frames:
        return skel
    old_idx = np.linspace(0, 1, T)
    new_idx = np.linspace(0, 1, target_frames)
    out = np.zeros((target_frames, skel.shape[1], skel.shape[2]), dtype=np.float32)
    for j in range(skel.shape[1]):
        for ax in range(skel.shape[2]):
            out[:, j, ax] = np.interp(new_idx, old_idx, skel[:, j, ax])
    return out


def add_velocity(skel):
    """skel: (T, J, 3) → (T, J, 6) with zero-padded first frame."""
    velocity    = np.zeros_like(skel)
    velocity[1:] = skel[1:] - skel[:-1]
    return np.concatenate([skel, velocity], axis=-1)   # (T, J, 6)


def augment_skel(skel, target_frames=TARGET_FRAMES):
    T = skel.shape[0]
    speed    = np.random.uniform(0.75, 1.25)
    n_new    = max(10, int(T * speed))
    idxs     = np.linspace(0, T - 1, n_new).astype(int)
    skel     = normalise_length(skel[idxs], target_frames)

    keep_ratio = np.random.uniform(0.80, 1.0)
    n_keep     = max(10, int(target_frames * keep_ratio))
    keep_idxs  = np.sort(np.random.choice(target_frames, n_keep, replace=False))
    skel       = normalise_length(skel[keep_idxs], target_frames)
    return skel


class MultiViewDataset(Dataset):
    """
    Multi-view early fusion dataset.

    Each row in df represents ONE TRIAL with filepaths for all cameras.
    Returns:
        fused   : torch.Tensor (T, J, 6×V=18)  — early-fused features
        quality : torch.Tensor scalar
        ex_id   : torch.Tensor long
    """
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        view_feats = []

        for c in ALL_CAMERAS:
            fpath = row[f'filepath_c{c}']
            skel  = load_skeleton(fpath)

            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3), dtype=np.float32)

            skel = normalise_length(skel, self.target_frames)

            if self.augment:
                skel = augment_skel(skel, self.target_frames)

            skel_vel = add_velocity(skel)         # (T, J, 6)
            view_feats.append(skel_vel)

        # ── Early Fusion: stack along feature axis ────────────────────────
        # Each view: (T, J, 6) → stacked: (T, J, 18)
        fused = np.concatenate(view_feats, axis=-1)   # (T, J, 6*V)

        fused_t  = torch.tensor(fused,          dtype=torch.float32)
        quality  = torch.tensor(row['quality'], dtype=torch.float32)
        ex_id    = torch.tensor(row['exercise'],dtype=torch.long)
        return fused_t, quality, ex_id

print('✓ MultiViewDataset defined  [early fusion: (T,J,18)]')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — Normalisation: centre_and_scale for multi-view fused input
#
# For each view's position channels, centre on mid-hip and scale by torso.
# Channels layout:  [pos0(3), vel0(3), pos1(3), vel1(3), pos2(3), vel2(3)]
#                    view-0 (0:6)     view-1 (6:12)    view-2 (12:18)
# ══════════════════════════════════════════════════════════════════════════

def centre_and_scale_multiview(x):
    """
    x: (B, T, J, 18)  ← 3 views × 6 channels (pos+vel)

    Applies hip-centring and torso-height scaling independently
    to each view's position channels. Velocity channels are
    scaled by the same torso height.
    """
    out = x.clone()
    for v in range(NUM_VIEWS):
        start = v * 6          # e.g. 0, 6, 12
        pos   = out[:, :, :, start:start+3]
        vel   = out[:, :, :, start+3:start+6]

        hip     = (pos[:, :, 1:2, :] + pos[:, :, 4:5, :]) / 2.0
        pos     = pos - hip
        shoulder = (pos[:, :, 11:12, :] + pos[:, :, 14:15, :]) / 2.0
        torso_h  = shoulder[:, :, :, 1:2].abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        pos      = pos / torso_h
        vel      = vel / torso_h

        out[:, :, :, start:start+3]   = pos
        out[:, :, :, start+3:start+6] = vel
    return out

print('✓ centre_and_scale_multiview defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — True ST-GCN adjacency  (identical to single-view)
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
        if dist[j] < dist[i]:  A[1, i, j] = 1.0
        elif dist[j] > dist[i]: A[2, i, j] = 1.0
        else:                   A[1, i, j] = 1.0

        if dist[i] < dist[j]:  A[1, j, i] = 1.0
        elif dist[i] > dist[j]: A[2, j, i] = 1.0
        else:                   A[1, j, i] = 1.0

    for k in range(3):
        row_sum  = A[k].sum(axis=1)
        d_inv_sq = np.where(row_sum > 0, np.power(row_sum, -0.5), 0.0)
        D_inv_sq = np.diag(d_inv_sq)
        A[k]     = D_inv_sq @ A[k] @ D_inv_sq

    return torch.tensor(A, dtype=torch.float32)

print('✓ build_stgcn_adjacency defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN model  (only in_features changes: 6 → 18)
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
        return self.relu(self.bn(out))


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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x   = self.spatial(x, A)
        x   = self.temporal(x)
        return self.relu(x + res)


NUM_EXERCISES = len(EXERCISE_REMAP)

class STGCN_MultiView_EarlyFusion(nn.Module):
    """
    ST-GCN with early fusion of multi-view skeletons.

    Input:  (B, T, J, IN_FEATURES=18)
                        ↑ 6 channels per view × 3 views, fused before the network

    The rest of the architecture is identical to single-view ST-GCN:
        9 × STGCNBlock → GAP → Exercise Embedding → Regression Head

    The only modification: data_bn and the first STGCNBlock accept
    in_features=18 instead of 6.
    """
    def __init__(self, in_features=IN_FEATURES, K=3, dropout=0.5):
        super().__init__()

        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer('A', A)

        # Input normalisation over all fused channels
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
        x           : (B, T, J, C)   C = IN_FEATURES = 18
        exercise_id : (B,)
        returns     : (B,) quality scores
        """
        B, T, J, C = x.shape

        # Data BN: (B,T,J,C) → (B,C*J,T) → BN → (B,C,T,J)
        x = x.permute(0, 3, 2, 1).reshape(B, C * J, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, J, T).permute(0, 1, 3, 2)   # (B, C, T, J)

        for block in self.blocks:
            x = block(x, self.A)

        x   = self.gap(x).squeeze(-1).squeeze(-1)        # (B, 256)
        ex  = self.ex_embed(exercise_id)                 # (B, 32)
        h   = torch.cat([x, ex], dim=1)                  # (B, 288)
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# ── Sanity check ──────────────────────────────────────────────────────────
_dummy_x  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_ex = torch.zeros(2, dtype=torch.long)
_model    = STGCN_MultiView_EarlyFusion()
_out      = _model(_dummy_x, _dummy_ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
print(f'\n✓ STGCN_MultiView_EarlyFusion sanity check passed — output: {_out.shape}')

total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters: {total_params:,}')
del _dummy_x, _dummy_ex, _model, _out


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — run_epoch  (unchanged logic, uses centre_and_scale_multiview)
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def run_epoch(model, loader, reg_fn, is_train=True, optimiser=None):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for fused, qualities, exercise_ids in loader:
            fused        = centre_and_scale_multiview(fused.to(DEVICE))
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)

            preds = model(fused, exercise_ids)
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

print('✓ run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Plotting helpers  (identical to single-view)
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
    ax.set_title('Regression Loss — Multi-View Early Fusion', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curve.png'))

def plot_rmse_mae(history, save_dir, test_rmse=None, test_mae=None):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1)
    ax.set_title('R² Score — Multi-View Early Fusion', fontsize=13, fontweight='bold')
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
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1)
    ax.set_title('Pearson Correlation — Multi-View Early Fusion', fontsize=13, fontweight='bold')
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
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True Quality'); ax.set_ylabel('Predicted Quality')
    ax.set_title(f'{split_name} — True vs Predicted (Multi-View Early Fusion)',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f'R²={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}',
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
               label=f'Best ({best_epoch})')
    ax.axvline(stopped_epoch, color='red',    linestyle='--', linewidth=2,
               label=f'Stopped ({stopped_epoch})')
    ax.set_title('MAE + Early Stopping', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))

print('✓ Plotting helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Early Stopping
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
# Cell 15 — Training loop
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
    MultiViewDataset(train_df, augment=True),
    batch_size  = BATCH_SIZE,
    sampler     = train_sampler,
    num_workers = 0,
    pin_memory  = (DEVICE == 'cuda'),
)
val_loader  = DataLoader(
    MultiViewDataset(val_df,  augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE=='cuda'),
)
test_loader = DataLoader(
    MultiViewDataset(test_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE=='cuda'),
)

model = STGCN_MultiView_EarlyFusion(
    in_features=IN_FEATURES, K=3, dropout=0.5
).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=50, min_lr=1e-6, verbose=True)

early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
SPLITS        = ['train', 'val']
METRICS       = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history       = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info('STARTING MULTI-VIEW EARLY FUSION TRAINING')
log.info(f'train={len(train_df)} trials | val={len(val_df)} | test={len(test_df)}')
log.info(f'IN_FEATURES={IN_FEATURES}  BATCH={BATCH_SIZE}  PATIENCE={PATIENCE}')
log.info('=' * 70)

print(f'\n{"═"*68}')
print(f'  Multi-View Early Fusion  |  '
      f'Train:{len(train_df)}  Val:{len(val_df)}  Test:{len(test_df)}')
print(f'  IN_FEATURES={IN_FEATURES} ({NUM_VIEWS} cameras × 6)  '
      f'|  Patience:{PATIENCE}  |  Batch:{BATCH_SIZE}')
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
        break

print('\n✓ Training complete!')

# ── Restore best weights & evaluate test ──────────────────────────────────
model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch
final_te = run_epoch(model, test_loader, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')
print(f'  PCC  : {final_te["pcc"]:.4f}')

# ── Collect predictions ───────────────────────────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for fused, qualities, exercise_ids in test_loader:
        fused        = centre_and_scale_multiview(fused.to(DEVICE))
        exercise_ids = exercise_ids.to(DEVICE)
        preds        = model(fused, exercise_ids)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# ── Plots ─────────────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR, test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR, test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR, test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History JSON → {json_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Per-exercise test metrics  (identical to single-view)
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

per_ex_df = pd.DataFrame([
    {'exercise': f'E{ex}', **vals} for ex, vals in per_ex_results.items()
])
per_ex_df.to_csv(os.path.join(LOGS_DIR, 'per_exercise_metrics.csv'), index=False)

# ── Per-exercise scatter grid ─────────────────────────────────────────────
n_ex   = len(unique_exercises)
n_cols = 3
n_rows = int(np.ceil(n_ex / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle('Per-Exercise: True vs Predicted Quality (Multi-View Early Fusion)',
             fontsize=13, fontweight='bold')

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
    ax.set_xlabel('True Quality'); ax.set_ylabel('Predicted Quality')
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.97,
            f'MAE={res["mae"]:.3f}\nRMSE={res["rmse"]:.3f}\n'
            f'R²={res["r2"]:.3f}\nPCC={res["pcc"]:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'per_exercise_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise scatter grid saved')

np.savez(os.path.join(LOGS_DIR, 'test_predictions.npz'),
         q_true=np.array(all_true_q),
         q_pred=np.array(all_pred_q),
         exercise_ids=np.array(all_exercise_ids))
np.savez(os.path.join(LOGS_DIR, 'training_history.npz'),
         **{k: np.array(v) for k, v in history.items()})


# ══════════════════════════════════════════════════════════════════════════
# Cell 16.5 — Diagnostic: per-exercise skeleton variance + quality dist
# ══════════════════════════════════════════════════════════════════════════

from collections import defaultdict

print("=" * 60)
print("DIAGNOSTIC 1: Per-exercise skeleton variance (C0)")
print("=" * 60)

axis_var    = defaultdict(list)
sample_rows = train_df.sample(min(300, len(train_df)), random_state=0)

for _, row in sample_rows.iterrows():
    skel = load_skeleton(row['filepath_c0'])
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


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv            = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 60)
print('  TRAINING SUMMARY — ST-GCN Multi-View Early Fusion')
print('=' * 60)
print(f'  Cameras          : {ALL_CAMERAS}  ({IN_FEATURES} input features)')
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

log.info(f'[EARLY FUSION] Best Epoch={best_epoch}  stopped={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f}  RMSE={final_te["rmse"]:.4f}')
log.info(f'Test R²={final_te["r2"]:.4f}   PCC={final_te["pcc"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [{'split': 'test_overall', 'exercise': 'all',
         'fusion': 'early', 'cameras': str(ALL_CAMERAS),
         'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
         'val_mae': best_val_mae, 'val_rmse': best_val_rmse,
         'val_r2': best_val_r2,   'val_pcc': best_val_pcc,
         'test_mae': final_te['mae'], 'test_rmse': final_te['rmse'],
         'test_r2':  final_te['r2'],  'test_pcc':  final_te['pcc']}]
for ex, vals in per_ex_results.items():
    rows.append({'split': 'test_per_exercise', 'exercise': f'E{ex}',
                 'fusion': 'early', 'cameras': str(ALL_CAMERAS),
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed.')
