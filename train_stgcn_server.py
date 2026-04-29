# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"
CAMERA_ID     = 1
NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
TARGET_FRAMES = 100
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 300
LR            = 3e-4
BATCH_SIZE    = 48
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_SingleView_Regression_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE  = 50
MIN_DELTA = 1e-4
WARMUP_EPOCHS = 10

print('✓ Configuration loaded')
print(f'  DATASET_DIR : {DATASET_DIR}')
print(f'  NPZ_KEY     : {NPZ_KEY}')
print(f'  CAMERA_ID   : C{CAMERA_ID}')
print(f'  EXISTS      : {os.path.exists(DATASET_DIR)}')
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
from sklearn.metrics import r2_score   # no confusion_matrix needed

PLOTS_DIR = os.path.join(OUT_DIR, "plots")
LOGS_DIR  = os.path.join(OUT_DIR, "logs")
for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Libraries imported")
print("✓ Output folders ready:")
for d in [PLOTS_DIR, LOGS_DIR]:
    print("  ", d)


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
# Cell 5 — Logging setup
# ══════════════════════════════════════════════════════════════════════════

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(LOGS_DIR, f"training_{timestamp}.log")


class Tee:
    """Mirrors stdout to a log file simultaneously."""
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
log = logging.getLogger("GCN-Regression")
log.info("=" * 70)
log.info("ST-GCN Single-View Regression | BZU Physiotherapy Dataset")
log.info(f"Camera : C{CAMERA_ID}  |  Split : {int(TRAIN_RATIO*100)}/"
         f"{int(VAL_RATIO*100)}/{int((1-TRAIN_RATIO-VAL_RATIO)*100)}"
         f"  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
log.info(f"Log file : {log_file}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader
# ══════════════════════════════════════════════════════════════════════════

# Filename pattern: E4_P10_T6_C2_seg9_MMPose_human3d_motionbert_3D.npz

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

        # ── FIX: swap axes so Y=height, Z=depth ──────────────────
        # Original: X=left/right, Y=depth(tiny), Z=height
        # Target:   X=left/right, Y=height,      Z=depth
        x = arr[:, :, 0].copy()   # left/right  → keep as X
        y = arr[:, :, 1].copy()   # depth        → becomes Z
        z = arr[:, :, 2].copy()   # height       → becomes Y
        arr[:, :, 0] = x
        arr[:, :, 1] = z          # Y = height (was Z)
        arr[:, :, 2] = y          # Z = depth  (was Y)
        # ──────────────────────────────────────────────────────────

        return arr
    except Exception:
        return None

print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build dataset index
# ══════════════════════════════════════════════════════════════════════════

def build_index(dataset_dir, camera_id, df_csv):
    """
    Scan all NPZ files for the chosen camera and merge quality labels from CSV.
    trial_key = 'Pxx_Tyy' used as grouping key to prevent data leakage.
    Missing quality scores are filled with split-conditional means.
    """
    df_csv         = df_csv.copy()
    df_csv.columns = df_csv.columns.str.strip()

    all_files = sorted(glob.glob(
        os.path.join(dataset_dir, '**/*.npz'), recursive=True))
    print(f'NPZ files found (all cameras) : {len(all_files)}')

    if not all_files:
        print('❌ No NPZ files found — check DATASET_DIR in Cell 1')
        return pd.DataFrame()

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
        meta['trial_key'] = f"{meta['person']}_{meta['trial_id']}"
        records.append(meta)

    if not records:
        print(f'❌ No records for camera C{camera_id}. Try CAMERA_ID=1 or 2.')
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Fill NaN quality scores with split-conditional means
    correct_mean   = df.loc[df['trial_num'] <= 2, 'quality'].mean()
    erroneous_mean = df.loc[df['trial_num'] >= 3, 'quality'].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5

    mask_correct   = df['quality'].isna() & (df['trial_num'] <= 2)
    mask_erroneous = df['quality'].isna() & (df['trial_num'] >= 3)
    df.loc[mask_correct,   'quality'] = correct_mean
    df.loc[mask_erroneous, 'quality'] = erroneous_mean

    print(f'\n✓ Total samples  (C{camera_id}) : {len(df)}')
    print(f'✓ Unique trials              : {df["trial_key"].nunique()}')
    print(f'✓ Quality mean ± std         : {df["quality"].mean():.3f} ± {df["quality"].std():.3f}')
    print(f'\n✓ Exercise distribution:')
    print(df['exercise'].value_counts().sort_index())
    return df


df_index = build_index(DATASET_DIR, CAMERA_ID, df_csv)

# Filter out corrupted NPZ files
print("Checking for corrupted files...")
bad_files = []
for fpath in df_index['filepath']:
    skel = load_skeleton(fpath)
    if skel is None:
        bad_files.append(fpath)

if bad_files:
    print(f"  Removing {len(bad_files)} corrupted files from index")
    for f in bad_files:
        print(f"    BAD: {f}")
    df_index = df_index[~df_index['filepath'].isin(bad_files)].reset_index(drop=True)

print(f"  Clean samples remaining: {len(df_index)}")

if len(df_index) > 0:
    print('\n── Sample rows ──')
    print(df_index[['exercise', 'person', 'trial_id', 'segment', 'quality']].head(15))

print(df_index['camera'].value_counts().sort_index())
print(f'\nTotal samples: {len(df_index)}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Skeleton visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

# Human3.6M joint ordering (MotionBERT)
SKELETON_EDGES = [
    (0, 1), (1, 2),  (2, 3),           # right leg:  Hip→RHip→RKnee→RAnkle
    (0, 4), (4, 5),  (5, 6),           # left leg:   Hip→LHip→LKnee→LAnkle
    (0, 7), (7, 8),  (8, 9),           # spine:      Hip→Spine→Thorax→Neck
    (9, 10),                            # Neck→Head
    (8, 11), (11, 12), (12, 13),       # left arm:   Thorax→LShoulder→LElbow→LWrist
    (8, 14), (14, 15), (15, 16),       # right arm:  Thorax→RShoulder→RElbow→RWrist
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
        (axes[0], x,  y,  'Front View  (X–Y)', 'X (left/right)', 'Y (up/down)',   True),
        (axes[1], z,  y,  'Side View   (Z–Y)', 'Z (depth)',       'Y (up/down)',   True),
        (axes[2], x, -z,  'Top View    (X–Z)', 'X (left/right)', '-Z (forward)',  False),
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

sample_skel = load_skeleton(df_index.iloc[10]['filepath'])

print("Axis ranges across all frames:")
print(f"X: [{sample_skel[:,:,0].min():.3f}, {sample_skel[:,:,0].max():.3f}] — likely left/right")
print(f"Y: [{sample_skel[:,:,1].min():.3f}, {sample_skel[:,:,1].max():.3f}] — likely ???")
print(f"Z: [{sample_skel[:,:,2].min():.3f}, {sample_skel[:,:,2].max():.3f}] — likely height")

# Check hip (joint 0) vs head (joint 10) on each axis
hip  = sample_skel[:, 0, :]   # shape (T, 3)
head = sample_skel[:, 10, :]

print("\nHip  mean XYZ:", hip.mean(axis=0))
print("Head mean XYZ:", head.mean(axis=0))
print("\nDifference (head - hip):", head.mean(axis=0) - hip.mean(axis=0))

# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset (regression only)
# ══════════════════════════════════════════════════════════════════════════

class BZUDataset(Dataset):
    """Returns (skeleton, quality_score) — no exercise ID."""
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        skel    = load_skeleton(row['filepath'])
        if skel is None:
            skel = np.zeros((TARGET_FRAMES, 17, 3), dtype=np.float32)
        skel    = self._normalise_length(skel)
        if self.augment:
            skel = self._augment(skel)
        skel    = torch.tensor(skel,            dtype=torch.float32)
        quality = torch.tensor(row['quality'],  dtype=torch.float32)
        return skel, quality                      # regression only

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
        speed = np.random.uniform(0.8, 1.2)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = self._normalise_length(skel[idxs])
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.005
        if np.random.rand() < 0.5:
            skel[:, :, 0] *= -1.0
        return skel


print('✓ BZUDataset defined (regression only)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN Model (Regression only)
# ══════════════════════════════════════════════════════════════════════════

def build_adj(num_joints, edges):
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg      = A.sum(axis=1)
    d_inv_sq = np.diag(np.power(deg, -0.5))
    return torch.tensor(d_inv_sq @ A @ d_inv_sq, dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, t_kernel=9, stride=1, dropout=0.0):
        super().__init__()
        self.register_buffer('A', A)
        pad = (t_kernel - 1) // 2

        self.W_s    = nn.Linear(in_ch, out_ch, bias=False)
        self.bn_s   = nn.BatchNorm2d(out_ch)
        self.t_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=(t_kernel, 1),
                      padding=(pad, 0),
                      stride=(stride, 1),
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.res = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        B, C, T, J = x.shape
        xs = x.permute(0, 2, 3, 1).reshape(B * T, J, C)
        xs = self.W_s(xs)
        A_exp = self.A.unsqueeze(0).expand(B * T, -1, -1).contiguous()
        xs = torch.bmm(A_exp, xs)
        xs = xs.reshape(B, T, J, -1).permute(0, 3, 1, 2)
        xs = F.relu(self.bn_s(xs))
        return F.relu(self.t_conv(xs) + self.res(x))


class STGCN_Regression(nn.Module):
    """Predicts quality score directly — no classification head."""
    def __init__(self, dropout=0.3):
        super().__init__()
        A = build_adj(NUM_JOINTS, SKELETON_EDGES)

        self.data_bn = nn.BatchNorm1d(3)   # BN over 3 coordinate channels

        cfg = [
            (3,   64,  1, 3),
            (64,  64,  1, 9),
            (64,  128, 2, 9),
            (128, 128, 1, 9),
            (128, 256, 2, 9),
            (256, 256, 1, 9),
        ]
        self.blocks = nn.ModuleList(
            [STGCNBlock(ic, oc, A, t_kernel=tk, stride=s, dropout=dropout)
            for ic, oc, s, tk in cfg]
        )

        self.drop     = nn.Dropout(dropout)
        # Regression head only
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64),
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
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, J, C = x.shape   # (B, 100, 17, 3)

        xbn = x.permute(0, 3, 1, 2).reshape(B, C, T * J)
        xbn = self.data_bn(xbn)
        x   = xbn.reshape(B, C, T, J).permute(0, 2, 3, 1)
        x   = x.permute(0, 3, 1, 2)          # (B, 3, T, J)

        for blk in self.blocks:
            x = blk(x)

        x = x.mean(dim=[2, 3])               # (B, 256)
        x = self.drop(x)

        # Produce a score in (1,5) range — adjust if your quality scale differs
        qua = 3.0 + 2.0 * torch.tanh(self.reg_head(x).squeeze(1))
        return qua

print('✓ ST-GCN regression model defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch (regression only)
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    Root-relative normalisation + torso-height scaling.
    x: (B, T, J, 3)  — after axis swap: dim1=X, dim2=Y(height), dim3=Z(depth)
    """
    hip  = (x[:, :, 1:2, :] + x[:, :, 4:5, :]) / 2.0
    x    = x - hip

    # Now Y (index 1) is height — torso_h will be meaningful
    shoulder = (x[:, :, 11:12, :] + x[:, :, 14:15, :]) / 2.0
    torso_h  = shoulder[:, :, :, 1:2].abs()   # index 1 = Y = height ✓
    torso_h  = torso_h.mean(dim=1, keepdim=True).clamp(min=1e-6)
    return x / torso_h


def run_epoch(model, loader, optimiser, reg_fn, is_train=True):
    """
    One epoch over loader. Returns dict with loss, rmse, mae, r2.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, qualities in loader:
            skels     = centre_and_scale(skels.to(DEVICE))
            qualities = qualities.to(DEVICE)

            preds = model(skels)                       # (B,)
            loss  = reg_fn(preds, qualities)

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            total_loss += loss.item()
            q_true.extend(qualities.cpu().numpy())
            q_pred.extend(preds.detach().cpu().numpy())

    n   = max(1, len(loader))
    qt  = np.array(q_true)
    qp  = np.array(q_pred)

    return {
        'loss': total_loss / n,
        'rmse': float(np.sqrt(np.mean((qt - qp) ** 2))),
        'mae' : float(np.mean(np.abs(qt - qp))),
        'r2'  : float(r2_score(qt, qp)) if len(qt) > 1 else 0.0,
    }

print('✓ centre_and_scale and run_epoch (regression) defined')


# ══════════════════════════════════════════════════════════════════
# Cell 13-NEW — LOPO-CV Training Loop
# ══════════════════════════════════════════════════════════════════

from sklearn.model_selection import LeaveOneGroupOut
import copy, json

persons     = df_index['person'].unique()       # 16 persons
groups      = df_index['person'].values         # group label per sample
logo        = LeaveOneGroupOut()

fold_results = []   # collect per-fold test metrics

for fold_idx, (train_val_idx, test_idx) in enumerate(
        logo.split(df_index, groups=groups)):

    test_person = df_index.iloc[test_idx]['person'].iloc[0]
    print(f'\n{"═"*68}')
    print(f'  FOLD {fold_idx+1:2d}/16 — held-out person: {test_person}')
    print(f'{"═"*68}')

    # ── Split train_val further into train/val (use one more person as val)
    train_val_df = df_index.iloc[train_val_idx].reset_index(drop=True)
    test_df      = df_index.iloc[test_idx].reset_index(drop=True)

    # Use a second held-out person for early stopping validation
    # Pick the person whose mean quality is closest to overall mean
    remaining_persons = train_val_df['person'].unique()
    overall_mean      = train_val_df['quality'].mean()
    val_person        = min(
        remaining_persons,
        key=lambda p: abs(
            train_val_df[train_val_df['person']==p]['quality'].mean()
            - overall_mean
        )
    )

    train_df = train_val_df[train_val_df['person'] != val_person].reset_index(drop=True)
    val_df   = train_val_df[train_val_df['person'] == val_person].reset_index(drop=True)

    print(f'  Train: {len(train_df)} samples ({len(remaining_persons)-1} persons)')
    print(f'  Val  : {len(val_df)} samples  (person {val_person})')
    print(f'  Test : {len(test_df)} samples  (person {test_person})')

    # ── DataLoaders ───────────────────────────────────────────────
    train_loader = DataLoader(BZUDataset(train_df, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE=='cuda'))
    val_loader   = DataLoader(BZUDataset(val_df, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE=='cuda'))
    test_loader  = DataLoader(BZUDataset(test_df, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE=='cuda'))

    # ── Fresh model + optimizer per fold ─────────────────────────
    model     = STGCN_Regression(dropout=0.3).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    reg_fn    = nn.SmoothL1Loss()

    fold_history = {f'{s}_{m}': []
                    for s in ['train','val','test']
                    for m in ['loss','rmse','mae','r2']}
    stopped_epoch = EPOCHS

    # ── Per-fold training loop ────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimiser, reg_fn, is_train=True)
        vl = run_epoch(model, val_loader,   optimiser, reg_fn, is_train=False)
        te = run_epoch(model, test_loader,  optimiser, reg_fn, is_train=False)
        scheduler.step()

        for split, res in [('train',tr),('val',vl),('test',te)]:
            for m in ['loss','rmse','mae','r2']:
                fold_history[f'{split}_{m}'].append(res[m])

        stop, improved = early_stop.step(vl['mae'], model, epoch)
        star = ' ★' if improved else ''
        msg  = (f'  Ep {epoch:3d}/{EPOCHS} | '
                f'Tr mae={tr["mae"]:.3f} r2={tr["r2"]:.3f} | '
                f'Vl mae={vl["mae"]:.3f} r2={vl["r2"]:.3f} | '
                f'Te mae={te["mae"]:.3f} r2={te["r2"]:.3f} | '
                f'ES {early_stop.counter}/{PATIENCE}{star}')
        print(msg)
        log.info(msg)
        if stop:
            stopped_epoch = epoch
            print(f'  ⏹ Early stop at epoch {epoch} (best={early_stop.best_epoch})')
            break

    # ── Evaluate with best weights ────────────────────────────────
    model.load_state_dict(early_stop.best_wts)
    final_te = run_epoch(model, test_loader, optimiser, reg_fn, is_train=False)

    fold_results.append({
        'fold'         : fold_idx + 1,
        'test_person'  : test_person,
        'val_person'   : val_person,
        'best_epoch'   : early_stop.best_epoch,
        'stopped_epoch': stopped_epoch,
        'test_mae'     : final_te['mae'],
        'test_rmse'    : final_te['rmse'],
        'test_r2'      : final_te['r2'],
    })

    print(f'\n  ── Fold {fold_idx+1} Test Results ──')
    print(f'  MAE={final_te["mae"]:.4f}  RMSE={final_te["rmse"]:.4f}  R²={final_te["r2"]:.4f}')

    # Optional: save per-fold scatter plot
    model.eval()
    all_true_q, all_pred_q = [], []
    with torch.no_grad():
        for skels, qualities in test_loader:
            skels = centre_and_scale(skels.to(DEVICE))
            preds = model(skels)
            all_true_q.extend(qualities.numpy())
            all_pred_q.extend(preds.cpu().numpy())

    plot_regression_scatter(all_true_q, all_pred_q,
                            split_name=f'Fold{fold_idx+1}_{test_person}',
                            save_dir=PLOTS_DIR)


# ══════════════════════════════════════════════════════════════════
# Cell 14-NEW — Aggregate LOPO Results
# ══════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(fold_results)
print(results_df.to_string(index=False))

mean_mae  = results_df['test_mae'].mean()
std_mae   = results_df['test_mae'].std()
mean_rmse = results_df['test_rmse'].mean()
std_rmse  = results_df['test_rmse'].std()
mean_r2   = results_df['test_r2'].mean()
std_r2    = results_df['test_r2'].std()

print('\n' + '='*60)
print('  LOPO-CV FINAL SUMMARY')
print('='*60)
print(f'  MAE  : {mean_mae:.4f} ± {std_mae:.4f}')
print(f'  RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}')
print(f'  R²   : {mean_r2:.4f} ± {std_r2:.4f}')
print('='*60)

log.info(f'LOPO-CV  MAE={mean_mae:.4f}±{std_mae:.4f}  '
         f'RMSE={mean_rmse:.4f}±{std_rmse:.4f}  '
         f'R²={mean_r2:.4f}±{std_r2:.4f}')

# Save results
results_df.to_csv(os.path.join(OUT_DIR, 'lopo_cv_results.csv'), index=False)
print(f'✓ Results saved → {os.path.join(OUT_DIR, "lopo_cv_results.csv")}')
