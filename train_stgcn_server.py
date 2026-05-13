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
FUSED_JOINTS  = NUM_JOINTS * NUM_VIEWS   # 51

TARGET_FRAMES = 100

# ── Training config ────────────────────────────────────────────────────────
# FIX 1: Increased regularisation vs single-view to fight overfitting
EPOCHS        = 300
LR            = 1e-4        # lower than single-view (was 1e-4)
BATCH_SIZE    = 32
WEIGHT_DECAY  = 1e-3          # stronger than single-view (was 5e-4)
OUT_DIR       = "/mvdlph/masa/GCN_MultiView_EarlyFusion_Results"

# ── Early Stopping ─────────────────────────────────────────────────────────
# FIX 2: Longer patience to avoid stopping at epoch 5 like single-view
PATIENCE      = 80            # shorter than 80 but with better regularisation
MIN_DELTA     = 1e-4

# ── Single-view baseline (from your results) ───────────────────────────────
SV_TEST_MAE  = 0.4076
SV_TEST_RMSE = 0.5569
SV_TEST_R2   = 0.4859
SV_TEST_PCC  = 0.7443

print('✓ Configuration loaded')
print(f'  DATASET_DIR  : {DATASET_DIR}')
print(f'  SPLIT_DIR    : {SPLIT_DIR}')
print(f'  ALL_CAMERAS  : {ALL_CAMERAS}  (NUM_VIEWS={NUM_VIEWS})')
print(f'  FUSED_JOINTS : {FUSED_JOINTS}  ({NUM_VIEWS} × {NUM_JOINTS})')
print(f'  EXISTS       : {os.path.exists(DATASET_DIR)}')
print(f'  SPLIT EXISTS : {os.path.exists(SPLIT_DIR)}')
print(f'  LR           : {LR}  (reduced from single-view)')
print(f'  WEIGHT_DECAY : {WEIGHT_DECAY}  (stronger regularisation)')
print(f'  PATIENCE     : {PATIENCE} epochs')
print(f'\n  Single-View Baseline:')
print(f'    MAE={SV_TEST_MAE}  RMSE={SV_TEST_RMSE}  R²={SV_TEST_R2}  PCC={SV_TEST_PCC}')


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

run_name  = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
RUN_DIR   = os.path.join(OUT_DIR, run_name)
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
LOGS_DIR  = os.path.join(RUN_DIR, "logs")

for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Libraries imported")
print("✓ Run directory:", RUN_DIR)


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
log = logging.getLogger("GCN-MultiView-EarlyFusion")
log.info("=" * 70)
log.info(f"ST-GCN Multi-View Early Fusion | Cameras={ALL_CAMERAS} | FusedJoints={FUSED_JOINTS}")
log.info(f"Epochs={EPOCHS} | Patience={PATIENCE} | Batch={BATCH_SIZE} | LR={LR}")
log.info(f"Single-View Baseline: MAE={SV_TEST_MAE} RMSE={SV_TEST_RMSE} R²={SV_TEST_R2} PCC={SV_TEST_PCC}")
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
    """Load NPZ skeleton, returns (T, 17, 3) float32 or None on failure."""
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
# Cell 7 — Build multi-view dataset index FROM PRE-SPLIT DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════

def build_multiview_index(split_name, df_csv, cameras=ALL_CAMERAS):
    """
    Reads pre-split folders (same folders used by single-view experiment).
    Returns one row per (exercise, person, trial, segment) with one
    filepath column per camera: filepath_C0, filepath_C1, filepath_C2.
    """
    split_path = os.path.join(SPLIT_DIR, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    all_files = sorted(glob.glob(
        os.path.join(split_path, '**/*.npz'), recursive=True))
    print(f'\n[{split_name.upper()}] NPZ files found : {len(all_files)}')

    # key: (exercise, person, trial_id, segment) → {camera: fpath}
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

    # Fill NaN quality
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
    """Drop rows where ALL cameras are missing or corrupted."""
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


# ── Build the three splits (same pre-split folders as single-view) ─────────
train_df = build_multiview_index('train', df_csv)
val_df   = build_multiview_index('valid', df_csv)
test_df  = build_multiview_index('test',  df_csv)

print('\nChecking for corrupted files...')
train_df = remove_all_corrupted(train_df, label='TRAIN')
val_df   = remove_all_corrupted(val_df,   label='VALID')
test_df  = remove_all_corrupted(test_df,  label='TEST')

df_index = pd.concat([train_df, val_df, test_df], ignore_index=True)

# ── Sanity: no trial_key leak across splits ────────────────────────────────
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
print('\n✓ Multi-view index ready  →  train / val / test DataFrames built')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7.5 — Camera availability audit
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Camera availability per split:")
print("=" * 60)
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"\n  {name}:")
    for cam in ALL_CAMERAS:
        col = f'filepath_C{cam}'
        if col in d.columns:
            avail = d[col].notna().sum()
            print(f"    Camera {cam}: {avail}/{len(d)} segments ({avail/len(d)*100:.1f}%)")

print("\n" + "=" * 60)
print("View completeness per split:")
print("=" * 60)
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"  {name}: " + str(d['n_views_available'].value_counts().sort_index().to_dict()))


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


def plot_skeleton_3d(skel, frame_idx=0, title='Skeleton Sanity Check', save_path=None):
    pts = skel[frame_idx]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    views = [
        (axes[0], x,  y,  'Front View  (X–Y)', 'X', 'Y',  False),
        (axes[1], z,  y,  'Side View   (Z–Y)', 'Z', 'Y',  False),
        (axes[2], x, -z,  'Top View    (X–Z)', 'X', '-Z', False),
    ]
    for ax, hx, hy, view_title, xlabel, ylabel, invert_y in views:
        for (i, j) in SKELETON_EDGES:
            ax.plot([hx[i], hx[j]], [hy[i], hy[j]], color='dimgray', lw=2, zorder=1)
        for part, idxs in JOINT_COLORS.items():
            ax.scatter(hx[idxs], hy[idxs], c=PART_COLOR[part], s=80, zorder=3,
                       edgecolors='black', linewidths=0.5, label=part)
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
# Cell 9 — Visualise a sample skeleton (first available camera)
# ══════════════════════════════════════════════════════════════════════════

idx = 10
row = df_index.iloc[idx]
sample_fp = None
for cam in ALL_CAMERAS:
    fp = row.get(f'filepath_C{cam}')
    if fp is not None:
        sample_fp = fp
        break

if sample_fp:
    sample_skel = load_skeleton(sample_fp)
    print(row[['person', 'exercise', 'trial_id', 'segment']])
    print(f'Skeleton shape : {sample_skel.shape}')
    plot_skeleton_3d(
        sample_skel, frame_idx=0,
        title=f"Skeleton · {row['trial_key']} · E{row['exercise']}",
        save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
    )


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════

def _normalise_length_np(skel, target_frames):
    """
    Resample skel (T, J, C) → (target_frames, J, C).
    FIX: vectorised over J*C instead of nested loops — much faster.
    """
    T = skel.shape[0]
    if T == target_frames:
        return skel
    old_idx  = np.linspace(0, 1, T)
    new_idx  = np.linspace(0, 1, target_frames)
    flat     = skel.reshape(T, -1)                          # (T, J*C)
    out_flat = np.zeros((target_frames, flat.shape[1]), dtype=np.float32)
    for k in range(flat.shape[1]):
        out_flat[:, k] = np.interp(new_idx, old_idx, flat[:, k])
    return out_flat.reshape(target_frames, skel.shape[1], skel.shape[2])


def _centre_scale_np(skel):
    """
    Centre on mid-hip, scale by torso height.
    skel: (T, J, 3) → (T, J, 3)
    """
    hip     = (skel[:, 1:2, :] + skel[:, 4:5, :]) / 2.0
    skel    = skel - hip
    shoulder = (skel[:, 11:12, :] + skel[:, 14:15, :]) / 2.0
    torso_h  = np.abs(shoulder[:, :, 1:2]).mean(axis=0, keepdims=True).clip(min=1e-6)
    return skel / torso_h


def _add_velocity_np(skel):
    """
    skel: (T, J, 3) → (T, J, 6)  position + velocity
    """
    vel     = np.zeros_like(skel)
    vel[1:] = skel[1:] - skel[:-1]
    return np.concatenate([skel, vel], axis=-1)


print('✓ Preprocessing helpers defined (vectorised normalise_length)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — BZUMultiViewDataset (EARLY FUSION)
#
#  Per sample: load each camera → (T, J, 3)
#              centre+scale per view independently
#              add velocity → (T, J, 6)
#              concatenate along joint axis → (T, V*J, 6) = (T, 51, 6)
#  Missing cameras → zero-filled (T, J, 6) block
# ══════════════════════════════════════════════════════════════════════════

class BZUMultiViewDataset(Dataset):
    def __init__(self, df, cameras=ALL_CAMERAS,
                 target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.cameras       = cameras
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        views = []
        view_mask = []   # ← NEW

        for cam in self.cameras:
            fp   = row.get(f'filepath_C{cam}')
            skel = load_skeleton(fp) if fp is not None else None

            if skel is None:
                # Zero-pad for missing/corrupted camera
                block = np.zeros((self.target_frames, NUM_JOINTS, 6),
                                 dtype=np.float32)
                views.append(block)
                view_mask.append(False)   # ← NEW: camera missing
                continue

            skel = _normalise_length_np(skel, self.target_frames)  # (T,J,3)
            if self.augment:
                skel = self._augment(skel)
            skel = _centre_scale_np(skel)                           # (T,J,3)
            skel = _add_velocity_np(skel)                           # (T,J,6)
            views.append(skel)
            view_mask.append(True)        # ← NEW: camera available

        # Early fusion: stack along joint axis → (T, V*J, 6)
        fused = np.concatenate(views, axis=1)   # (T, 51, 6)

        skel_tensor = torch.tensor(fused,           dtype=torch.float32)
        quality     = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)
        mask_tensor = torch.tensor(view_mask,          dtype=torch.bool)  # ← NEW (V,)
        return skel_tensor, quality, exercise_id, mask_tensor

    def _augment(self, skel):
        """Speed jitter + small Gaussian noise."""
        T     = skel.shape[0]
        speed = np.random.uniform(0.85, 1.15)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = _normalise_length_np(skel[idxs], self.target_frames)
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.003
        if np.random.rand() < 0.5:          # ← add this
         skel[:, :, 0] *= -1.0           # ← and this (mirror left-right)
        return skel


print('✓ BZUMultiViewDataset defined (early fusion, cameras=%s)' % ALL_CAMERAS)

# Quick shape check
_ds   = BZUMultiViewDataset(train_df)
_s, _q, _e, _m = _ds[0]          # ← unpack all 4 values
print(f'  Sample shape: {_s.shape}  '
      f'(expected T={TARGET_FRAMES}, V*J={FUSED_JOINTS}, C=6)')
assert _s.shape == (TARGET_FRAMES, FUSED_JOINTS, 6), \
    f"Shape mismatch! Got {_s.shape}"
del _ds, _s, _q, _e, _m          # ← also del _m
print('✓ Dataset shape check passed')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Multi-view Adjacency + GCN Model
# ══════════════════════════════════════════════════════════════════════════

def build_multiview_adj(num_joints, intra_edges, num_views, cross_view=True):
    """
    Build Kipf-normalised adjacency for a multi-view fused graph.
    Intra-view: skeleton edges per camera block.
    Cross-view: same joint j across cameras u↔v.
    """
    N = num_joints * num_views
    A = np.zeros((N, N), dtype=np.float32)

    for v in range(num_views):
        offset = v * num_joints
        for (i, j) in intra_edges:
            A[offset + i, offset + j] = 1.0
            A[offset + j, offset + i] = 1.0

    if cross_view:
        for u in range(num_views):
            for v in range(num_views):
                if u == v:
                    continue
                for j in range(num_joints):
                    A[u * num_joints + j, v * num_joints + j] = 1.0

    # Kipf: Â = D̃^(-½)(A+I)D̃^(-½)
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
    """
    Bidirectional GRU over T frames.
    FIX: uses max pooling over joints instead of mean — preserves
    peak motion information better.
    """
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
        # h: (B, T, J, C)
        # FIX: max pooling over joints instead of mean
        h = h.max(dim=2)[0]                                 # (B, T, C)
        _, hidden = self.gru(h)                              # (2*L, B, H)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)       # (B, 2H)
        return self.drop(h)


NUM_EXERCISES = 10

class GCN_MultiView_Regression(nn.Module):
    """
    Multi-view early-fusion GCN.
    Input : (B, T, V*J, 6)
    Output: (B,) quality score in range (1, 5)

    FIX: output uses wider tanh range to handle quality labels near 1 or 5.
    FIX: stronger dropout than single-view to fight overfitting.
    FIX: smaller hidden dims — single-view already had enough capacity.
    """
    def __init__(self, num_joints=FUSED_JOINTS, in_features=6,
                 hidden_dims=None, dropout=0.5, cross_view=True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        A_hat = build_multiview_adj(
            NUM_JOINTS, SKELETON_EDGES, NUM_VIEWS, cross_view=cross_view)
        self.register_buffer('A_hat', A_hat)

        self.data_bn  = nn.BatchNorm1d(in_features)
        # Learnable token for missing cameras — shape (NUM_JOINTS, 6)
        self.missing_view_token = nn.Parameter(torch.zeros(NUM_JOINTS, 6))  # ← NEW
        self.gcn      = GCNBackbone(in_features, hidden_dims, dropout=dropout)

        self.temporal = TemporalEncoder(
            feat_dim   = hidden_dims[-1],
            hidden_dim = 128,
            num_layers = 2,
            dropout    = 0.4,           # stronger than original 0.3
        )
        combined_dim = self.temporal.out_dim  # 256


        # # FIX: stronger dropout in regression head
        # self.reg_head = nn.Sequential(
        #     nn.Linear(combined_dim + 32, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),            # was 0.3
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),          # added LayerNorm
        #     nn.ReLU(),
        #     nn.Dropout(0.3),            # was 0.2
        #     nn.Linear(128, 1),
        # )

        self.heads = nn.ModuleDict({
            f'E{i}': nn.Sequential(
                nn.Linear(combined_dim, 128),
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

    def forward(self, x, exercise_id,  view_mask):
        """
        x           : (B, T, V*J, 6)
        exercise_id : (B,) long
        """
        B, T, J, C = x.shape

         # ── Replace missing view blocks with learnable token ──────────────
        filled_views = []
        for v in range(NUM_VIEWS):
            start     = v * NUM_JOINTS
            end       = (v + 1) * NUM_JOINTS
            view_data = x[:, :, start:end, :]          # (B, T, J, 6)
            avail     = view_mask[:, v].float().view(B, 1, 1, 1)  # 1=available, 0=missing
            token     = self.missing_view_token.view(1, 1, NUM_JOINTS, 6)
            filled    = avail * view_data + (1.0 - avail) * token
            filled_views.append(filled)
        x = torch.cat(filled_views, dim=2)             # (B, T, V*J, 6)
        # ──────────────────────────────────────────────────────────────────

        # rest unchanged from here
        xbn = x.permute(0, 3, 1, 2).reshape(B, C, T * x.shape[2])
        xbn = self.data_bn(xbn)
        x   = xbn.reshape(B, C, T, J).permute(0, 2, 3, 1)

        h  = self.gcn(x, self.A_hat)       # (B, T, V*J, 256)
        h  = self.temporal(h)              # (B, 256)

        # ex = self.ex_embed(exercise_id)    # (B, 32)
        # h  = torch.cat([h, ex], dim=1)     # (B, 288)

        # # FIX: wider output range 3±2.4 → (0.6, 5.4) instead of 3±2 → (1, 5)
        # # Handles labels near 1.0 (min=1.34 in your data, so 0.6 gives headroom)
        # # and near 5.0 (max=5.0 in your data)
        # out = 3.0 + 2.4 * torch.tanh(self.reg_head(h).squeeze(1))
        # واضيفي هاي:
        out = torch.cat([
            self.heads[f'E{exercise_id[b].item()}'](h[b].unsqueeze(0))
            for b in range(h.size(0))
        ], dim=0).squeeze(-1)
        out = 3.0 + 2.4 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# ── Sanity check ──────────────────────────────────────────────────────────
# AFTER (fixed)
_dummy_x    = torch.zeros(2, TARGET_FRAMES, FUSED_JOINTS, 6)
_dummy_ex   = torch.zeros(2, dtype=torch.long)
_dummy_mask = torch.ones(2, NUM_VIEWS, dtype=torch.bool)
_model      = GCN_MultiView_Regression()
_out        = _model(_dummy_x, _dummy_ex, _dummy_mask)    # ← correct order
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
print(f'✓ GCN_MultiView_Regression sanity check passed — output shape: {_out.shape}')
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters: {total_params:,}')

A_hat_check = build_multiview_adj(NUM_JOINTS, SKELETON_EDGES, NUM_VIEWS)
intra_e = (A_hat_check[:NUM_JOINTS, :NUM_JOINTS] > 0).sum().item()
cross_e = (A_hat_check[:NUM_JOINTS, NUM_JOINTS:2*NUM_JOINTS] > 0).sum().item()
print(f'✓ Multi-view adjacency: intra-view={int(intra_e)}, cross-view per block={int(cross_e)}')

del _dummy_x, _dummy_ex, _model, _out


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Device & run_epoch
#
# FIX: optimiser is optional — pass None for val/test to make intent clear.
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def run_epoch(model, loader, reg_fn, optimiser=None, is_train=True):
    """
    FIX: optimiser is now optional (None for val/test).
    Skeletons arrive pre-normalised from the Dataset.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, qualities, exercise_ids, view_mask in loader:
            skels        = skels.to(DEVICE)
            qualities    = qualities.to(DEVICE)
            exercise_ids = exercise_ids.to(DEVICE)
            view_mask    = view_mask.to(DEVICE) 
            
            preds = model(skels, exercise_ids, view_mask)          # ← pass mask
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


print('✓ run_epoch defined (optimiser=None for val/test)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13.5 — Split quality distribution audit
# ══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Quality score distribution audit")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Quality Score Distributions per Split", fontsize=14, fontweight='bold')

splits_audit = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
trial_types  = [
    ('Correct (T≤2)',   lambda d: d[d['trial_num'] <= 2]),
    ('Erroneous (T≥3)', lambda d: d[d['trial_num'] >= 3]),
]

for col, (split_name, split_df) in enumerate(splits_audit):
    for row_idx, (type_name, filter_fn) in enumerate(trial_types):
        ax  = axes[row_idx][col]
        sub = filter_fn(split_df)
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{split_name} — {type_name}')
            continue
        q = sub['quality']
        ax.hist(q, bins=30, color='steelblue' if row_idx == 0 else 'tomato',
                edgecolor='black', alpha=0.8)
        ax.axvline(q.mean(), color='red',  linestyle='--', linewidth=2,
                   label=f'mean={q.mean():.3f}')
        ax.axvline(q.median(), color='gold', linestyle=':', linewidth=2,
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
# Cell 14 — Plotting helpers
# FIX: all plots now include single-view baseline reference lines
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def _add_test_line(ax, val, label, color='green'):
    ax.axhline(val, color=color, linestyle='-.', linewidth=1.5,
               label=f'Test {label}={val:.4f}')


def _add_baseline_line(ax, val, label, color='purple'):
    """Add single-view baseline reference line."""
    ax.axhline(val, color=color, linestyle=':', linewidth=1.5,
               label=f'SV baseline {label}={val:.4f}')


def plot_loss_curves(history, save_dir, test_loss=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    if test_loss is not None:
        _add_test_line(ax, test_loss, 'Loss')
    ax.set_title('Regression Loss (MSE) — Multi-View Early Fusion',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curve.png'))


def plot_rmse_mae(history, save_dir, test_rmse=None, test_mae=None):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('RMSE & MAE — Multi-View Early Fusion vs Single-View Baseline',
                 fontsize=13, fontweight='bold')
    for ax, metric, title, test_val, sv_val in [
        (axes[0], 'rmse', 'RMSE', test_rmse, SV_TEST_RMSE),
        (axes[1], 'mae',  'MAE',  test_mae,  SV_TEST_MAE),
    ]:
        ax.plot(epochs, history[f'train_{metric}'], label='Train',      color='steelblue')
        ax.plot(epochs, history[f'val_{metric}'],   label='Validation', color='darkorange')
        if test_val is not None:
            _add_test_line(ax, test_val, title)
        _add_baseline_line(ax, sv_val, f'SV {title}')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'rmse_mae.png'))


def plot_r2(history, save_dir, test_r2=None):
    epochs = range(1, len(history['val_r2']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    if test_r2 is not None:
        _add_test_line(ax, test_r2, 'R²')
    _add_baseline_line(ax, SV_TEST_R2, 'SV R²')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (R²=1)')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='Baseline (R²=0)')
    ax.set_title('R² Score — Multi-View Early Fusion vs Single-View',
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
    _add_baseline_line(ax, SV_TEST_PCC, 'SV PCC')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (PCC=1)')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='No correlation')
    ax.set_title('Pearson Correlation — Multi-View Early Fusion vs Single-View',
                 fontsize=13, fontweight='bold')
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
    ax.set_title(f'{split_name} Set — True vs Predicted (Multi-View Early Fusion)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.text(0.05, 0.95,
            f'R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}\n'
            f'——— Single-View ———\n'
            f'R²   = {SV_TEST_R2:.4f}\nMAE  = {SV_TEST_MAE:.4f}\nRMSE = {SV_TEST_RMSE:.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
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
    _add_baseline_line(ax, SV_TEST_MAE, 'SV MAE')
    ax.set_title('MAE + Early Stopping — Multi-View Early Fusion',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))


def plot_comparison_bar(final_te, save_dir):
    """
    FIX: new plot — side-by-side bar chart comparing
    single-view baseline vs multi-view early fusion.
    """
    metrics = ['MAE', 'RMSE', 'R²', 'PCC']
    sv_vals = [SV_TEST_MAE,  SV_TEST_RMSE,  SV_TEST_R2,  SV_TEST_PCC]
    mv_vals = [final_te['mae'], final_te['rmse'], final_te['r2'], final_te['pcc']]

    x    = np.arange(len(metrics))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, sv_vals, w, label='Single-View',
                color='steelblue',  edgecolor='black', alpha=0.85)
    b2 = ax.bar(x + w/2, mv_vals, w, label='Multi-View Early Fusion',
                color='darkorange', edgecolor='black', alpha=0.85)

    for bar, v in zip(b1, sv_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    for bar, v in zip(b2, mv_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_title('Single-View vs Multi-View Early Fusion — Test Set',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'comparison_sv_vs_mv_early_fusion.png'))


print('✓ Plotting helpers defined (with single-view baseline lines)')


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


print('✓ EarlyStopping defined (monitoring val MAE)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Training Loop
#
# FIX: num_workers > 0 to avoid DataLoader bottleneck with 3 cameras.
# FIX: optimiser=None passed to run_epoch for val/test.
# FIX: Huber loss (SmoothL1) instead of MSE — more robust to outliers,
#      which matters given exercises like E3/E7/E9 that had bad R² in SV.
# ══════════════════════════════════════════════════════════════════════════

# FIX: Huber loss (delta=1.0) is less sensitive to outlier quality scores
reg_fn = nn.SmoothL1Loss(beta=1.0)

# FIX: num_workers for faster loading (3 NPZ files per sample)
num_workers = min(4, os.cpu_count() or 1)
print(f'DataLoader num_workers = {num_workers}')

train_loader = DataLoader(
    BZUMultiViewDataset(train_df, augment=True),
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = num_workers,
    pin_memory  = (DEVICE == 'cuda'),
    persistent_workers = (num_workers > 0),
)

val_loader = DataLoader(
    BZUMultiViewDataset(val_df, augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = num_workers,
    pin_memory  = (DEVICE == 'cuda'),
    persistent_workers = (num_workers > 0),
)

test_loader = DataLoader(
    BZUMultiViewDataset(test_df, augment=False),
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = num_workers,
    pin_memory  = (DEVICE == 'cuda'),
    persistent_workers = (num_workers > 0),
)

model = GCN_MultiView_Regression(
    num_joints   = FUSED_JOINTS,
    in_features  = 6,
    hidden_dims  = [64, 128, 256],
    dropout      = 0.5,
    cross_view   = True,
).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# FIX: longer patience for scheduler — single-view reduced LR at epoch 36
# which was too early given the overfitting pattern
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=20, min_lr=1e-6, verbose=True)

early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val']
METRICS = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info('STARTING MULTI-VIEW EARLY FUSION TRAINING')
log.info(f'Cameras={ALL_CAMERAS}  FusedJoints={FUSED_JOINTS}  CrossViewEdges=True  Heads=per-exercise')
log.info(f'Loss=SmoothL1(Huber)  LR={LR}  WD={WEIGHT_DECAY}  Batch={BATCH_SIZE}')
log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')
log.info('=' * 70)

print(f'\n{"═"*72}')
print(f'  Multi-View Early Fusion  |  Cameras: {ALL_CAMERAS}  '
      f'|  Fused joints: {FUSED_JOINTS}')
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Loss : SmoothL1 (Huber)  |  LR: {LR}  |  WD: {WEIGHT_DECAY}')
print(f'  Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}  |  Workers: {num_workers}')
print(f'  Single-View Baseline: MAE={SV_TEST_MAE} RMSE={SV_TEST_RMSE} '
      f'R²={SV_TEST_R2} PCC={SV_TEST_PCC}')
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

# ── Final test evaluation (optimiser=None) ────────────────────────────────
final_te = run_epoch(model, test_loader, reg_fn, optimiser=None, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  {"Metric":<8}  {"Multi-View EF":>14}  {"Single-View":>12}  {"Δ":>8}')
print(f'  {"─"*50}')
for metric, mv_val, sv_val in [
    ('Loss',  final_te["loss"], None),
    ('RMSE',  final_te["rmse"], SV_TEST_RMSE),
    ('MAE',   final_te["mae"],  SV_TEST_MAE),
    ('R²',    final_te["r2"],   SV_TEST_R2),
    ('PCC',   final_te["pcc"],  SV_TEST_PCC),
]:
    if sv_val is not None:
        delta = mv_val - sv_val
        arrow = '↑' if (metric in ['R²', 'PCC'] and delta > 0) or \
                       (metric in ['RMSE', 'MAE'] and delta < 0) else '↓'
        print(f'  {metric:<8}  {mv_val:>14.4f}  {sv_val:>12.4f}  {delta:>+7.4f} {arrow}')
    else:
        print(f'  {metric:<8}  {mv_val:>14.4f}')

# ── Collect predictions ───────────────────────────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for skels, qualities, exercise_ids, view_mask in test_loader:
        skels        = skels.to(DEVICE)
        exercise_ids = exercise_ids.to(DEVICE)
        view_mask    = view_mask.to(DEVICE)                        # ← NEW
        preds        = model(skels, exercise_ids, view_mask)       # ← pass mask

        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# ── Save all plots ────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history, PLOTS_DIR, test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history, PLOTS_DIR,       test_r2=final_te['r2'])
plot_pcc(history, PLOTS_DIR,      test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)
plot_comparison_bar(final_te, PLOTS_DIR)          # NEW: SV vs MV comparison

json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')


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
fig.suptitle('Per-Exercise: True vs Predicted — Multi-View Early Fusion (Test)',
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
    ax.set_title(f'Exercise E{ex_id}  (n={res["n"]})',
                 fontsize=10, fontweight='bold')
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

# ── Per-exercise bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Per-Exercise Test Metrics — Multi-View Early Fusion',
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
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)

for ax, metric in zip(axes, ['mae', 'rmse', 'r2', 'pcc']):
    ax.axhline(final_te[metric], color='red', linestyle='--',
               linewidth=1.5, label=f'MV Overall={final_te[metric]:.3f}')
    ax.legend(fontsize=7)

plt.tight_layout()
bar_path = os.path.join(PLOTS_DIR, 'per_exercise_bar.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  ✓ Per-exercise bar chart → {bar_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv           = best_epoch - 1
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]
best_val_pcc  = history['val_pcc'][bv]

print('=' * 65)
print('  TRAINING SUMMARY — Multi-View Early Fusion')
print(f'  Cameras: {ALL_CAMERAS}  |  Fused joints: {FUSED_JOINTS}')
print('=' * 65)
print(f'  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')
print(f'  Best Val PCC     : {best_val_pcc:.4f}')
print('─' * 65)
print(f'  {"Metric":<8}  {"Multi-View EF":>14}  {"Single-View":>12}  {"Δ":>8}')
print(f'  {"─"*48}')
for metric, mv_val, sv_val in [
    ('MAE',  final_te["mae"],  SV_TEST_MAE),
    ('RMSE', final_te["rmse"], SV_TEST_RMSE),
    ('R²',   final_te["r2"],   SV_TEST_R2),
    ('PCC',  final_te["pcc"],  SV_TEST_PCC),
]:
    delta = mv_val - sv_val
    arrow = '↑' if (metric in ['R²', 'PCC'] and delta > 0) or \
                   (metric in ['RMSE', 'MAE'] and delta < 0) else '↓'
    print(f'  {metric:<8}  {mv_val:>14.4f}  {sv_val:>12.4f}  {delta:>+7.4f} {arrow}')
print('=' * 65)

log.info(f'Multi-View Early Fusion Summary: Best={best_epoch} stopped={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f} RMSE={final_te["rmse"]:.4f} '
         f'R²={final_te["r2"]:.4f} PCC={final_te["pcc"]:.4f}')
log.info(f'vs SV: ΔMAE={final_te["mae"]-SV_TEST_MAE:+.4f} '
         f'ΔR²={final_te["r2"]-SV_TEST_R2:+.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [{'model': 'single_view',
         'test_mae': SV_TEST_MAE, 'test_rmse': SV_TEST_RMSE,
         'test_r2':  SV_TEST_R2,  'test_pcc':  SV_TEST_PCC},
        {'model': 'multiview_early_fusion_per_ex_heads',
         'cameras': str(ALL_CAMERAS), 'fused_joints': FUSED_JOINTS,
         'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
         'val_mae': best_val_mae,  'val_rmse': best_val_rmse,
         'val_r2':  best_val_r2,   'val_pcc':  best_val_pcc,
         'test_mae': final_te['mae'], 'test_rmse': final_te['rmse'],
         'test_r2':  final_te['r2'],  'test_pcc':  final_te['pcc']}]

for ex, vals in per_ex_results.items():
    rows.append({'model': 'multiview_early_fusion_per_exercise',
                 'exercise': f'E{ex}',
                 'test_mae': vals['mae'], 'test_rmse': vals['rmse'],
                 'test_r2':  vals['r2'],  'test_pcc':  vals['pcc'],
                 'n': vals['n']})

pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV (includes single-view baseline) → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
