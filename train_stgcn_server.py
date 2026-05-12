# ══════════════════════════════════════════════════════════════════════════
# MULTI-VIEW EARLY FUSION — full replacement for the single-view notebook
#
# WHAT CHANGED vs single-view
# ────────────────────────────────────────────────────────────────────────
#  Cell 1   : CAMERA_ID removed; ALL_CAMERAS & NUM_VIEWS added
#  Cell 7   : build_index_from_split  → one row per (exercise,person,trial,segment)
#             with cam0_path / cam1_path / cam2_path columns
#  Cell 10  : BZUDataset loads all views, stacks pos+vel → (T, J, 6*V)
#  Cell 11  : GCN_Regression  in_features = 6 * NUM_VIEWS  (= 18)
#  Cell 12  : centre_and_scale handles 6*V channels
#  Cell 16  : Training loop — camera label removed from logs
#  Everything else is unchanged.
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration  ★ CHANGED
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR     = os.path.join(DATASET_DIR, "by_person")
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"

# ── Multi-view settings ───────────────────────────────────────────────────
ALL_CAMERAS   = [0, 1, 2]          # cameras to fuse
NUM_VIEWS     = len(ALL_CAMERAS)   # 3
# CAMERA_ID is intentionally removed — we use all cameras

NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
TARGET_FRAMES = 100
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 300
LR            = 3e-4
BATCH_SIZE    = 48
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_MultiView_EarlyFusion_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 80
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 10

print('✓ Configuration loaded (MULTI-VIEW EARLY FUSION)')
print(f'  DATASET_DIR : {DATASET_DIR}')
print(f'  SPLIT_DIR   : {SPLIT_DIR}')
print(f'  ALL_CAMERAS : {ALL_CAMERAS}  (NUM_VIEWS={NUM_VIEWS})')
print(f'  INPUT DIM   : {6 * NUM_VIEWS}  (6 features × {NUM_VIEWS} views)')
print(f'  EXISTS      : {os.path.exists(DATASET_DIR)}')
print(f'  SPLIT EXISTS: {os.path.exists(SPLIT_DIR)}')
print(f'  PATIENCE    : {PATIENCE} epochs')


# ══════════════════════════════════════════════════════════════════════════
# Cell 2 — Imports & output folders  (UNCHANGED)
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


# ══════════════════════════════════════════════════════════════════════════
# Cell 3 — Explore dataset folder & one NPZ file  (UNCHANGED)
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
# Cell 4 — Load CSV labels  (UNCHANGED)
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
# Cell 4.5 — Person-level data audit  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Quality stats per person:")
print("=" * 60)
print(df_csv.groupby('person')['mean'].agg(['mean','std','min','max','count']).round(3))


# ══════════════════════════════════════════════════════════════════════════
# Cell 5 — Logging setup  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(LOGS_DIR, f"training_{timestamp}.log")

class Tee:
    def __init__(self, console, filepath):
        self.console  = console
        self._logfile = open(filepath, 'a', encoding='utf-8', buffering=1)
    def write(self, msg):
        self.console.write(msg); self._logfile.write(msg)
    def flush(self):
        self.console.flush(); self._logfile.flush()
    def restore(self):
        sys.stdout = self.console; self._logfile.close()

if not isinstance(sys.stdout, Tee):
    sys.stdout = Tee(sys.stdout, log_file)
print(f'✓ stdout → also writing to {log_file}')

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger("GCN-MultiView")
log.info("=" * 70)
log.info("ST-GCN Multi-View EARLY FUSION Regression | BZU Physiotherapy Dataset")
log.info(f"Cameras : {ALL_CAMERAS}  |  NUM_VIEWS={NUM_VIEWS}  |  "
         f"in_features={6*NUM_VIEWS}  |  Epochs={EPOCHS}  |  Patience={PATIENCE}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader  (UNCHANGED)
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
        if arr.ndim == 1:  return None
        if arr.ndim == 2:  arr = arr.reshape(arr.shape[0], 17, 3)
        elif arr.ndim == 4: arr = arr.squeeze(0)
        if arr.shape[1] != 17 or arr.shape[2] != 3: return None
        return arr
    except Exception:
        return None

print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build multi-view index from pre-split directories  ★ CHANGED
#
# New structure: one row per (exercise, person, trial, segment) combination.
# Camera filepaths are stored in separate columns: cam0_path, cam1_path, ...
# A view that is absent for a sample gets NaN in its column — the dataset
# will substitute an all-zero skeleton for that view at load time.
# ══════════════════════════════════════════════════════════════════════════
def build_index_from_split(split_name, df_csv, all_cameras=ALL_CAMERAS):
    """
    Scan a pre-split folder (train | valid | test), parse every NPZ filename,
    then PIVOT cameras so that each row represents one
    (exercise, person, trial, segment) sample with one column per camera view.

    Parameters
    ----------
    split_name  : str             — 'train', 'valid', or 'test'
    df_csv      : pd.DataFrame    — CSV quality labels
    all_cameras : list[int]       — camera IDs to include (default ALL_CAMERAS)

    Returns
    -------
    pd.DataFrame  columns:
        exercise, person, trial_num, trial_id, segment, quality, trial_key,
        cam0_path, cam1_path, cam2_path   (NaN if that view is absent)
    """
    split_path = os.path.join(SPLIT_DIR, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    all_files = sorted(glob.glob(
        os.path.join(split_path, '**/*.npz'), recursive=True))
    print(f'\n[{split_name.upper()}] NPZ files found : {len(all_files)}')

    df_csv = df_csv.copy()
    df_csv.columns = df_csv.columns.str.strip()

    # ── 1. Collect all individual-file records ────────────────────────────
    records = []
    for fpath in all_files:
        meta = parse_filename(fpath)
        if meta is None:
            continue
        if meta['camera'] not in all_cameras:
            continue
        records.append(meta)

    if not records:
        print(f'  ⚠️  No records for split="{split_name}"')
        return pd.DataFrame()

    df_raw = pd.DataFrame(records)

    # ── 2. Pivot cameras into columns ─────────────────────────────────────
    # Group key = (exercise, person, trial_num, trial_id, segment)
    GROUP_KEYS = ['exercise', 'person', 'trial_num', 'trial_id', 'segment']

    pivot = (
        df_raw
        .pivot_table(
            index   = GROUP_KEYS,
            columns = 'camera',
            values  = 'filepath',
            aggfunc = 'first',   # take first if duplicates exist
        )
        .reset_index()
    )
    # Rename camera columns → cam0_path, cam1_path, ...
    pivot.columns.name = None
    cam_col_map = {cam: f'cam{cam}_path' for cam in all_cameras}
    pivot.rename(columns=cam_col_map, inplace=True)
    # Ensure all expected cam columns exist (fill with NaN if absent)
    for cam in all_cameras:
        col = f'cam{cam}_path'
        if col not in pivot.columns:
            pivot[col] = np.nan

    # ── 3. Merge quality labels from CSV ─────────────────────────────────
    def lookup_quality(row):
        match = df_csv[
            (df_csv['exercise'] == f"E{row['exercise']}") &
            (df_csv['person']   == row['person'])          &
            (df_csv['trial']    == row['trial_id'])
        ]
        return float(match.iloc[0]['mean']) if len(match) > 0 else np.nan

    pivot['quality']   = pivot.apply(lookup_quality, axis=1)
    pivot['trial_key'] = (pivot['exercise'].apply(lambda e: f'E{e}') + '_'
                          + pivot['person'] + '_' + pivot['trial_id'])
    pivot['split']     = split_name

    # ── 4. Fill NaN quality with split-conditional means ──────────────────
    correct_mean   = pivot.loc[pivot['trial_num'] <= 2, 'quality'].mean()
    erroneous_mean = pivot.loc[pivot['trial_num'] >= 3, 'quality'].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5

    pivot.loc[pivot['quality'].isna() & (pivot['trial_num'] <= 2),
              'quality'] = correct_mean
    pivot.loc[pivot['quality'].isna() & (pivot['trial_num'] >= 3),
              'quality'] = erroneous_mean

    # ── 5. View coverage audit ─────────────────────────────────────────────
    cam_cols  = [f'cam{c}_path' for c in all_cameras]
    view_avail = pivot[cam_cols].notna().sum(axis=1)  # views available per sample
    full_views = (view_avail == NUM_VIEWS).sum()

    print(f'  Segments (samples)     : {len(pivot)}')
    print(f'  All {NUM_VIEWS} views present    : {full_views} '
          f'({100*full_views/len(pivot):.1f}%)')
    for cam in all_cameras:
        col = f'cam{cam}_path'
        n   = pivot[col].notna().sum()
        print(f'    cam{cam}: {n}/{len(pivot)} ({100*n/len(pivot):.1f}%)')
    print(f'  Unique trials          : {pivot["trial_key"].nunique()}')
    q = pivot['quality']
    print(f'  Quality mean±std       : {q.mean():.3f} ± {q.std():.3f}')

    return pivot.reset_index(drop=True)


def remove_corrupted(df, label=''):
    """
    Drop rows where ALL camera views fail to load
    (partial corruption is handled with zero-pad in the dataset).
    """
    cam_cols = [f'cam{c}_path' for c in ALL_CAMERAS]
    bad_rows = []
    for idx, row in df.iterrows():
        any_ok = False
        for col in cam_cols:
            if pd.notna(row[col]) and load_skeleton(row[col]) is not None:
                any_ok = True
                break
        if not any_ok:
            bad_rows.append(idx)
    if bad_rows:
        print(f'  [{label}] Removing {len(bad_rows)} fully-corrupted sample(s)')
        df = df.drop(index=bad_rows).reset_index(drop=True)
    print(f'  [{label}] Clean samples : {len(df)}')
    return df


# ── Build the three splits ────────────────────────────────────────────────
train_df = build_index_from_split('train', df_csv)
val_df   = build_index_from_split('valid', df_csv)
test_df  = build_index_from_split('test',  df_csv)

# ── Remove fully-corrupted samples ────────────────────────────────────────
print('\nChecking for corrupted files...')
train_df = remove_corrupted(train_df, 'TRAIN')
val_df   = remove_corrupted(val_df,   'VALID')
test_df  = remove_corrupted(test_df,  'TEST')

# ── Combine for shared analysis ───────────────────────────────────────────
df_index = pd.concat([train_df, val_df, test_df], ignore_index=True)

# ── Sanity: no trial_key appears in more than one split ───────────────────
tr_keys = set(train_df['trial_key'])
vl_keys = set(val_df['trial_key'])
te_keys = set(test_df['trial_key'])
assert tr_keys.isdisjoint(vl_keys), 'LEAK: train ∩ val'
assert tr_keys.isdisjoint(te_keys), 'LEAK: train ∩ test'
assert vl_keys.isdisjoint(te_keys), 'LEAK: val ∩ test'
print('\n✓ No data-leakage detected across splits')

# ── Summary table ─────────────────────────────────────────────────────────
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

print('\n✓ Multi-view index ready  →  train / val / test DataFrames built')


# ══════════════════════════════════════════════════════════════════════════
# Cells 7.5 / 7.6 / 7.7 — audits  (largely UNCHANGED, minor label tweaks)
# ══════════════════════════════════════════════════════════════════════════

# View coverage per split
print("\nView coverage per split:")
for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    cam_cols = [f'cam{c}_path' for c in ALL_CAMERAS]
    for col in cam_cols:
        n = d[col].notna().sum() if col in d.columns else 0
        print(f"  {name} {col}: {n}/{len(d)}")

# Frame length distribution
lengths = []
sample_files = []
for _, row in df_index.sample(min(300, len(df_index)), random_state=42).iterrows():
    for cam in ALL_CAMERAS:
        p = row.get(f'cam{cam}_path')
        if pd.notna(p):
            sample_files.append(p)
            break  # one file per sample is enough for length stats

for fpath in sample_files:
    skel = load_skeleton(fpath)
    if skel is not None:
        lengths.append(skel.shape[0])

lengths = np.array(lengths)
print(f"\nFrame length distribution (sample of {len(lengths)} files):")
print(f"  min={lengths.min()}  max={lengths.max()}  "
      f"mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  std={lengths.std():.1f}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Skeleton visualisation helpers  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════
SKELETON_EDGES = [
    (0, 1), (1, 2),  (2, 3),
    (0, 4), (4, 5),  (5, 6),
    (0, 7), (7, 8),  (8, 9),
    (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
JOINT_NAMES  = [
    'Hip','R-Hip','R-Knee','R-Ankle','L-Hip','L-Knee','L-Ankle',
    'Spine','Thorax','Neck','Head',
    'L-Shoulder','L-Elbow','L-Wrist','R-Shoulder','R-Elbow','R-Wrist',
]
JOINT_COLORS = {
    'head' : [9, 10], 'arms': [11,12,13,14,15,16],
    'torso': [0,7,8], 'legs': [1,2,3,4,5,6],
}
PART_COLOR = {'head':'gold','arms':'dodgerblue','torso':'limegreen','legs':'tomato'}

def plot_skeleton_3d(skel, frame_idx=0, title='Skeleton', save_path=None):
    pts = skel[frame_idx]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    views = [
        (axes[0], x, y,  'Front (X–Y)', 'X','Y', False),
        (axes[1], z, y,  'Side  (Z–Y)', 'Z','Y', False),
        (axes[2], x, -z, 'Top   (X–Z)', 'X','-Z',False),
    ]
    for ax, hx, hy, vtitle, xl, yl, inv in views:
        for i,j in SKELETON_EDGES:
            ax.plot([hx[i],hx[j]],[hy[i],hy[j]],color='dimgray',lw=2,zorder=1)
        for part, idxs in JOINT_COLORS.items():
            ax.scatter(hx[idxs],hy[idxs],c=PART_COLOR[part],s=80,zorder=3,
                       edgecolors='black',linewidths=0.5,label=part)
        for ji in range(len(hx)):
            ax.annotate(str(ji),(hx[ji],hy[ji]),textcoords='offset points',
                        xytext=(5,5),fontsize=7,fontweight='bold',color='black',
                        bbox=dict(boxstyle='round,pad=0.1',facecolor='white',
                                  alpha=0.6,edgecolor='none'))
        ax.set_title(vtitle,fontweight='bold',fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        if inv: ax.invert_yaxis()
    axes[0].legend(loc='lower right',fontsize=7,framealpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Skeleton plot saved → {save_path}')
    plt.close()

print('✓ Skeleton visualisation helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — Visualise a sample skeleton  (UNCHANGED logic, paths adapted)
# ══════════════════════════════════════════════════════════════════════════
idx      = 10
row_demo = df_index.iloc[idx]

# Use the first available camera for the sanity visualisation
demo_path = None
for cam in ALL_CAMERAS:
    p = row_demo.get(f'cam{cam}_path')
    if pd.notna(p):
        demo_path = p
        break

sample_skel = load_skeleton(demo_path)
print(row_demo[['person','exercise','trial_id','segment']])
print(f'Skeleton shape : {sample_skel.shape}')
plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton · {row_demo['trial_key']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
)


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset — EARLY FUSION  ★ CHANGED
#
# For each sample the dataset:
#  1. Loads each available camera view (missing → zero skeleton)
#  2. Normalises length to TARGET_FRAMES independently per view
#  3. Computes velocity for each view independently
#  4. Concatenates: [pos_v0, vel_v0, pos_v1, vel_v1, pos_v2, vel_v2]
#     → shape  (T, J, 6 * NUM_VIEWS) = (100, 17, 18)
# ══════════════════════════════════════════════════════════════════════════
IN_FEATURES = 6 * NUM_VIEWS   # 18

class BZUDataset(Dataset):
    """
    Returns (skeleton_multiview, quality_score, exercise_id).
    skeleton_multiview : (T, J, 6*NUM_VIEWS)  — early fusion of all camera views
    """
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment
        self.cam_cols      = [f'cam{c}_path' for c in ALL_CAMERAS]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        view_feat = []   # will hold one (T, J, 6) tensor per view

        for cam_col in self.cam_cols:
            fpath = row.get(cam_col)

            # ── Load skeleton (zero-pad if missing / corrupted) ───────────
            skel = None
            if pd.notna(fpath):
                skel = load_skeleton(fpath)
            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3), dtype=np.float32)

            # ── Augment before normalising length ─────────────────────────
            if self.augment:
                skel = self._augment(skel)

            # ── Normalise to TARGET_FRAMES ─────────────────────────────────
            skel = self._normalise_length(skel)   # (T, J, 3)

            # ── Velocity: finite difference, zero-pad first frame ─────────
            velocity    = np.zeros_like(skel)
            velocity[1:] = skel[1:] - skel[:-1]

            # ── pos + vel → (T, J, 6) ─────────────────────────────────────
            feat = np.concatenate([skel, velocity], axis=-1)   # (T, J, 6)
            view_feat.append(feat)

        # ── Early fusion: concatenate all views along feature axis ────────
        # Result: (T, J, 6 * NUM_VIEWS) = (100, 17, 18)
        fused = np.concatenate(view_feat, axis=-1).astype(np.float32)

        skel_tensor = torch.tensor(fused,           dtype=torch.float32)
        quality     = torch.tensor(row['quality'],  dtype=torch.float32)
        exercise_id = torch.tensor(row['exercise'], dtype=torch.long)
        return skel_tensor, quality, exercise_id

    # ── helpers ───────────────────────────────────────────────────────────
    def _normalise_length(self, skel):
        T = skel.shape[0]
        if T == self.target_frames:
            return skel
        old_idx = np.linspace(0, 1, T)
        new_idx = np.linspace(0, 1, self.target_frames)
        out     = np.zeros((self.target_frames, skel.shape[1], skel.shape[2]),
                           dtype=np.float32)
        for j in range(skel.shape[1]):
            for ax in range(skel.shape[2]):
                out[:, j, ax] = np.interp(new_idx, old_idx, skel[:, j, ax])
        return out

    def _augment(self, skel):
        T     = skel.shape[0]
        speed = np.random.uniform(0.8, 1.2)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = self._normalise_length(skel[idxs])
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.005
        return skel


# ── Sanity check ─────────────────────────────────────────────────────────
_ds  = BZUDataset(train_df.head(4), augment=False)
_s, _q, _e = _ds[0]
assert _s.shape == (TARGET_FRAMES, NUM_JOINTS, IN_FEATURES), \
    f"Expected ({TARGET_FRAMES},{NUM_JOINTS},{IN_FEATURES}), got {_s.shape}"
print(f'✓ BZUDataset multi-view sanity check passed')
print(f'  Fused skeleton shape : {_s.shape}   '
      f'(T={TARGET_FRAMES}, J={NUM_JOINTS}, C={IN_FEATURES}={6}×{NUM_VIEWS} views)')
del _ds, _s, _q, _e


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — GCN Model  ★ CHANGED  (in_features = 6 * NUM_VIEWS = 18)
#           Architecture is identical to single-view; only the input
#           feature dimension widens to absorb all camera views.
# ══════════════════════════════════════════════════════════════════════════

def build_adj_kipf(num_joints, edges):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0; A[j, i] = 1.0
    A_tilde = A + np.eye(num_joints, dtype=np.float32)
    deg      = A_tilde.sum(axis=1)
    d_inv_sq = np.diag(np.power(deg, -0.5))
    A_hat    = d_inv_sq @ A_tilde @ d_inv_sq
    return torch.tensor(A_hat, dtype=torch.float32)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)
        if self.W.bias is not None: nn.init.zeros_(self.W.bias)

    def forward(self, H, A_hat):
        support = self.W(H)
        A_exp   = A_hat.unsqueeze(0).expand(H.size(0), -1, -1)
        return torch.bmm(A_exp, support)


class GCNBackbone(nn.Module):
    def __init__(self, in_features, hidden_dims, dropout=0.5):
        super().__init__()
        dims   = [in_features] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(GraphConvolution(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
        self.layers  = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.n_gcn   = len(hidden_dims)

    def forward(self, x, A_hat):
        B, T, J, C = x.shape
        h = x.reshape(B * T, J, C)
        gcn_idx = 0
        for i in range(0, len(self.layers), 2):
            h = self.layers[i](h, A_hat)
            N, J2, Co = h.shape
            h = self.layers[i+1](h.reshape(N*J2, Co)).reshape(N, J2, Co)
            gcn_idx += 1
            if gcn_idx < self.n_gcn:
                h = F.relu(h)
                h = self.dropout(h)
        return h.reshape(B, T, J, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.out_dim = hidden_dim * 2
        self.drop    = nn.Dropout(dropout)

    def forward(self, h):
        h = h.mean(dim=2)            # (B, T, C)
        _, hidden = self.gru(h)      # hidden: (2*layers, B, hidden)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.drop(h)


NUM_EXERCISES = 10

class GCN_Regression(nn.Module):
    """
    ST-GCN regression with multi-view early fusion.
    in_features = 6 * NUM_VIEWS  (position + velocity for every camera).
    """
    def __init__(self, in_features=IN_FEATURES, hidden_dims=None, dropout=0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        A_hat = build_adj_kipf(NUM_JOINTS, SKELETON_EDGES)
        self.register_buffer('A_hat', A_hat)

        # ── Input BN over IN_FEATURES channels ────────────────────────────
        self.data_bn  = nn.BatchNorm1d(in_features)   # ★ was 6, now 18
        self.gcn      = GCNBackbone(in_features, hidden_dims, dropout=dropout)

        self.temporal = TemporalEncoder(
            feat_dim=hidden_dims[-1], hidden_dim=128,
            num_layers=2, dropout=0.3,
        )
        combined_dim  = self.temporal.out_dim   # 256

        self.ex_embed = nn.Embedding(NUM_EXERCISES, 32)

        self.reg_head = nn.Sequential(
            nn.Linear(combined_dim + 32, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(p)
                    elif 'bias' in name: nn.init.zeros_(p)

    def forward(self, x, exercise_id):
        B, T, J, C = x.shape   # C = IN_FEATURES = 18

        # Input BN
        xbn = x.permute(0, 3, 1, 2).reshape(B, C, T * J)
        xbn = self.data_bn(xbn)
        x   = xbn.reshape(B, C, T, J).permute(0, 2, 3, 1)

        h   = self.gcn(x, self.A_hat)           # (B, T, J, 256)
        h   = self.temporal(h)                  # (B, 256)
        ex  = self.ex_embed(exercise_id)         # (B, 32)
        h   = torch.cat([h, ex], dim=1)          # (B, 288)
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# ── Sanity check ─────────────────────────────────────────────────────────
_dummy_x  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_ex = torch.zeros(2, dtype=torch.long)
_model    = GCN_Regression()
_out      = _model(_dummy_x, _dummy_ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
print(f'✓ Multi-view GCN sanity check passed — output shape: {_out.shape}')
print(f'  in_features = {IN_FEATURES}  ({6} × {NUM_VIEWS} views)')
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f'✓ Total trainable parameters: {total_params:,}')
del _dummy_x, _dummy_ex, _model, _out


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch  ★ CHANGED (centre_and_scale)
#
# centre_and_scale now normalises each view's position block independently.
# Layout of the 18 channels:
#   [0:3]  = cam0 position   [3:6]  = cam0 velocity
#   [6:9]  = cam1 position   [9:12] = cam1 velocity
#   [12:15]= cam2 position   [15:18]= cam2 velocity
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    Normalise each camera view's position channels independently,
    then re-scale its corresponding velocity channels by the same factor.

    x : (B, T, J, 6*NUM_VIEWS)
    Returns tensor of same shape with every view hip-centred & torso-scaled.
    """
    out = x.clone()
    for v in range(NUM_VIEWS):
        pos_s = v * 6        # start of position block for view v
        vel_s = pos_s + 3    # start of velocity block for view v

        pos = out[:, :, :, pos_s:pos_s+3]   # (B, T, J, 3)
        vel = out[:, :, :, vel_s:vel_s+3]   # (B, T, J, 3)

        # Hip-centre: midpoint of R-Hip (j=1) and L-Hip (j=4)
        hip = (pos[:, :, 1:2, :] + pos[:, :, 4:5, :]) / 2.0
        pos = pos - hip

        # Torso-scale: mean height of shoulder midpoint
        shoulder = (pos[:, :, 11:12, :] + pos[:, :, 14:15, :]) / 2.0
        torso_h  = shoulder[:, :, :, 1:2].abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        pos = pos / torso_h
        vel = vel / torso_h   # same scale for velocity

        out[:, :, :, pos_s:pos_s+3] = pos
        out[:, :, :, vel_s:vel_s+3] = vel

    return out


def run_epoch(model, loader, optimiser, reg_fn, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, qualities, exercise_ids in loader:
            skels        = centre_and_scale(skels.to(DEVICE))
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

print('✓ centre_and_scale (multi-view) and run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cells 13.5 / 14 / 15 — distribution audit, plotting helpers, EarlyStopping
#                         (UNCHANGED — copy from single-view notebook)
# ══════════════════════════════════════════════════════════════════════════

# … (paste your unchanged Cell 13.5, Cell 14, Cell 15 here) …


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Training Loop  ★ CHANGED  (camera label removed from logs)
# ══════════════════════════════════════════════════════════════════════════

reg_fn = nn.MSELoss()

def make_ds(df, aug):
    return BZUDataset(df, augment=aug)

train_loader = DataLoader(make_ds(train_df, True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))
val_loader   = DataLoader(make_ds(val_df,   False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))
test_loader  = DataLoader(make_ds(test_df,  False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))

model = GCN_Regression(
    in_features  = IN_FEATURES,   # ★ 18 instead of 6
    hidden_dims  = [64, 128, 256],
    dropout      = 0.5,
).to(DEVICE)

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5,
    patience=30, min_lr=1e-6, verbose=True,
)
early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
SPLITS        = ['train', 'val']
METRICS       = ['loss', 'rmse', 'mae', 'r2', 'pcc']
history       = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info('STARTING MULTI-VIEW EARLY FUSION REGRESSION TRAINING')
log.info(f'Cameras={ALL_CAMERAS}  in_features={IN_FEATURES}  '
         f'train={len(train_df)}  val={len(val_df)}  test={len(test_df)}')
log.info('=' * 70)

print(f'\n{"═"*68}')
print(f'  MULTI-VIEW EARLY FUSION  |  Views: {ALL_CAMERAS}  |  '
      f'in_features={IN_FEATURES}')
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Patience: {PATIENCE}  |  LR: 1e-4  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, optimiser, reg_fn, is_train=True)
    vl = run_epoch(model, val_loader,   optimiser, reg_fn, is_train=False)
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

# ── Restore best weights & final test eval ───────────────────────────────
model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch
final_te   = run_epoch(model, test_loader, optimiser, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')
print(f'  PCC  : {final_te["pcc"]:.4f}')

# ── Collect predictions for scatter / per-exercise plots ─────────────────
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

# ── Save plots (unchanged helper calls) ──────────────────────────────────
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te['loss'])
plot_rmse_mae(history,   PLOTS_DIR, test_rmse=final_te['rmse'], test_mae=final_te['mae'])
plot_r2(history,         PLOTS_DIR, test_r2=final_te['r2'])
plot_pcc(history,        PLOTS_DIR, test_pcc=final_te['pcc'])
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary  (UNCHANGED except title)
# ══════════════════════════════════════════════════════════════════════════
bv = best_epoch - 1
print('=' * 60)
print('  TRAINING SUMMARY — ST-GCN Multi-View Early Fusion')
print(f'  Cameras fused     : {ALL_CAMERAS}  (in_features={IN_FEATURES})')
print('=' * 60)
print(f'  Best Epoch        : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE      : {history["val_mae"][bv]:.4f}')
print(f'  Best Val RMSE     : {history["val_rmse"][bv]:.4f}')
print(f'  Best Val R²       : {history["val_r2"][bv]:.4f}')
print(f'  Best Val PCC      : {history["val_pcc"][bv]:.4f}')
print('─' * 60)
print(f'  Test MAE          : {final_te["mae"]:.4f}')
print(f'  Test RMSE         : {final_te["rmse"]:.4f}')
print(f'  Test R²           : {final_te["r2"]:.4f}')
print(f'  Test PCC          : {final_te["pcc"]:.4f}')
print('=' * 60)

log.info(f'MULTI-VIEW EARLY FUSION  cameras={ALL_CAMERAS}')
log.info(f'Best Epoch={best_epoch}  stopped_epoch={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f}  RMSE={final_te["rmse"]:.4f}  '
         f'R²={final_te["r2"]:.4f}  PCC={final_te["pcc"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
rows = [{'split': 'test_overall', 'exercise': 'all',
         'cameras': str(ALL_CAMERAS), 'num_views': NUM_VIEWS,
         'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch,
         'val_mae':  history['val_mae'][bv],  'val_rmse':  history['val_rmse'][bv],
         'val_r2':   history['val_r2'][bv],   'val_pcc':   history['val_pcc'][bv],
         'test_mae': final_te['mae'], 'test_rmse': final_te['rmse'],
         'test_r2':  final_te['r2'],  'test_pcc':  final_te['pcc']}]
pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV → {summary_path}')

sys.stdout.restore()
print('✓ Log file closed and saved.')
