import os 

# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"
CAMERA_ID     = 1
NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
TARGET_FRAMES = 100
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 200
LR            = 3e-4
BATCH_SIZE    = 48
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_SingleView_Regression_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE  = 30
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

        # لا تغيير في المحاور — استخدم البيانات كما هي من MotionBERT
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

frame = sample_skel[0]  # أول frame

print("Joint 0 (Root/Hip):")
print(f"  X={frame[0,0]:.3f}  Y={frame[0,1]:.3f}  Z={frame[0,2]:.3f}")

print("\nJoint 1 (R-Hip) vs Joint 4 (L-Hip):")
print(f"  R-Hip X={frame[1,0]:.3f}  |  L-Hip X={frame[4,0]:.3f}")
# لو الداتا صح → R-Hip و L-Hip لازم يكونوا على جانبين مختلفين (X مختلف الإشارة)

print("\nJoint 10 (Head) vs Joint 0 (Root):")
print(f"  Head Y={frame[10,1]:.3f}  |  Root Y={frame[0,1]:.3f}")
# لو Y للأسفل → Head أصغر من Root (أكثر سالب)
# لو Y للأعلى → Head أكبر
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
    x: (B, T, J, 3)
    """
    hip  = (x[:, :, 1:2, :] + x[:, :, 4:5, :]) / 2.0
    x    = x - hip

    shoulder = (x[:, :, 11:12, :] + x[:, :, 14:15, :]) / 2.0
    torso_h  = shoulder[:, :, :, 1:2].abs()
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


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Trial-based train / val / test split
# ══════════════════════════════════════════════════════════════════════════

def get_trial_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=42):
    rng = np.random.default_rng(random_state)

    correct_keys   = df[df['trial_num'] <= 2]['trial_key'].unique()
    erroneous_keys = df[df['trial_num'] >= 3]['trial_key'].unique()
    rng.shuffle(correct_keys)
    rng.shuffle(erroneous_keys)

    def split_keys(keys):
        n = len(keys)
        n_test  = max(1, int((1.0 - train_ratio - val_ratio) * n))
        n_val   = max(1, int(val_ratio * n))
        n_train = n - n_val - n_test
        return keys[:n_train], keys[n_train:n_train+n_val], keys[n_train+n_val:]

    tr_c, vl_c, te_c = split_keys(correct_keys)
    tr_e, vl_e, te_e = split_keys(erroneous_keys)

    train_keys = np.concatenate([tr_c, tr_e])
    val_keys   = np.concatenate([vl_c, vl_e])
    test_keys  = np.concatenate([te_c, te_e])

    train_df = df[df['trial_key'].isin(train_keys)].reset_index(drop=True)
    val_df   = df[df['trial_key'].isin(val_keys)].reset_index(drop=True)
    test_df  = df[df['trial_key'].isin(test_keys)].reset_index(drop=True)

    print(f'Correct trials:   {len(correct_keys)} → train={len(tr_c)}, val={len(vl_c)}, test={len(te_c)}')
    print(f'Erroneous trials: {len(erroneous_keys)} → train={len(tr_e)}, val={len(vl_e)}, test={len(te_e)}')
    print(f'  Train: {len(train_df)} samples | Val: {len(val_df)} | Test: {len(test_df)}')
    return train_df, val_df, test_df

train_df, val_df, test_df = get_trial_split(df_index)
print('\n✓ Train / Val / Test split ready')


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers (regression only)
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def plot_loss_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    ax.plot(epochs, history['test_loss'],  label='Test',       color='green', linestyle='--')
    ax.set_title('Regression Loss (SmoothL1)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curve.png'))


def plot_rmse_mae(history, save_dir):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('RMSE & MAE — Train / Val / Test', fontsize=14, fontweight='bold')

    axes[0].plot(epochs, history['train_rmse'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_rmse'],   label='Validation', color='darkorange')
    axes[0].plot(epochs, history['test_rmse'],  label='Test',       color='green', linestyle='--')
    axes[0].set_title('RMSE')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('RMSE')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_mae'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_mae'],   label='Validation', color='darkorange')
    axes[1].plot(epochs, history['test_mae'],  label='Test',       color='green', linestyle='--')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'rmse_mae.png'))


def plot_r2(history, save_dir):
    epochs = range(1, len(history['val_r2']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    ax.plot(epochs, history['test_r2'],  label='Test',       color='green', linestyle='--')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (R²=1)')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='Baseline (R²=0)')
    ax.set_title('R² Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('R²')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'r2_curve.png'))


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
    """MAE curves annotated with best epoch and early-stop epoch."""
    epochs = range(1, len(history['val_mae']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_mae'], label='Train MAE', color='steelblue')
    ax.plot(epochs, history['val_mae'],   label='Val MAE',   color='darkorange')
    ax.plot(epochs, history['test_mae'],  label='Test MAE',  color='green', linestyle='--')
    ax.axvline(best_epoch,    color='purple', linestyle=':',  linewidth=2,
               label=f'Best epoch ({best_epoch})')
    ax.axvline(stopped_epoch, color='red',    linestyle='--', linewidth=2,
               label=f'Early stop ({stopped_epoch})')
    ax.set_title('MAE + Early Stopping', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))


print('✓ Plotting helpers (regression) defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Early Stopping (monitors validation MAE)
# ══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Monitors val_mae (lower = better).
    """
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
# Cell 16 — Training Loop (Regression only)
# ══════════════════════════════════════════════════════════════════════════

reg_fn = nn.SmoothL1Loss()   # could also use nn.MSELoss()

def make_ds(df, aug):
    return BZUDataset(df, augment=aug)

train_loader = DataLoader(make_ds(train_df, True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))
val_loader   = DataLoader(make_ds(val_df, False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))
test_loader  = DataLoader(make_ds(test_df, False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))

model      = STGCN_Regression(dropout=0.3).to(DEVICE)
optimiser  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Warmup + cosine schedule
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

SPLITS  = ['train', 'val', 'test']
METRICS = ['loss', 'rmse', 'mae', 'r2']
history = {f'{s}_{m}': [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info('=' * 70)
log.info('STARTING REGRESSION TRAINING  (with Early Stopping on MAE)')
log.info('=' * 70)
log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')

print(f'\n{"═"*68}')
print(f'  Camera C{CAMERA_ID}  |  Train: {len(train_df)}  '
      f'Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Patience: {PATIENCE}  |  LR: {LR}  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, optimiser, reg_fn, is_train=True)
    vl = run_epoch(model, val_loader,   optimiser, reg_fn, is_train=False)
    te = run_epoch(model, test_loader,  optimiser, reg_fn, is_train=False)
    scheduler.step()

    for split, res in [('train', tr), ('val', vl), ('test', te)]:
        history[f'{split}_loss'].append(res['loss'])
        history[f'{split}_rmse'].append(res['rmse'])
        history[f'{split}_mae'].append(res['mae'])
        history[f'{split}_r2'].append(res['r2'])

    stop, improved = early_stop.step(vl['mae'], model, epoch)

    star = ' ★' if improved else ''
    msg  = (f'  Ep {epoch:3d}/{EPOCHS} | '
            f'Tr loss={tr["loss"]:.4f} mae={tr["mae"]:.3f} r2={tr["r2"]:.3f} | '
            f'Vl loss={vl["loss"]:.4f} mae={vl["mae"]:.3f} r2={vl["r2"]:.3f} | '
            f'Te loss={te["loss"]:.4f} mae={te["mae"]:.3f} r2={te["r2"]:.3f} | '
            f'ES {early_stop.counter}/{PATIENCE}{star}')
    print(msg)
    log.info(msg)

    if improved:
        print(f'    ★ val_mae={early_stop.best_mae:.4f}  '
              f'rmse={vl["rmse"]:.4f}  r2={vl["r2"]:.4f}')

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

# ── Final evaluation with best weights ───────────────────────────────────
final_te = run_epoch(model, test_loader, optimiser, reg_fn, is_train=False)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────')
print(f'  Loss : {final_te["loss"]:.4f}')
print(f'  RMSE : {final_te["rmse"]:.4f}')
print(f'  MAE  : {final_te["mae"]:.4f}')
print(f'  R²   : {final_te["r2"]:.4f}')

# ── Collect predictions for scatter plot ──────────────────────────────────
model.eval()
all_true_q, all_pred_q = [], []
with torch.no_grad():
    for skels, qualities in test_loader:
        skels = centre_and_scale(skels.to(DEVICE))
        preds = model(skels)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())

# ── Save all plots ────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR)
plot_rmse_mae(history, PLOTS_DIR)
plot_r2(history, PLOTS_DIR)
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

# ── Save history JSON ─────────────────────────────────────────────────────
json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')





# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC CELL — Data analysis, no classification metrics
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
from collections import defaultdict

print("=" * 60)
print("DIAGNOSTIC 1: Per-exercise skeleton variance (C0)")
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


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv           = best_epoch - 1   # 0-based index
best_val_mae  = history['val_mae'][bv]
best_val_rmse = history['val_rmse'][bv]
best_val_r2   = history['val_r2'][bv]

print('=' * 60)
print('  TRAINING SUMMARY — ST-GCN Regression (Single View)')
print('=' * 60)
print(f'  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')
print('─' * 60)
print(f'  Test MAE         : {final_te["mae"]:.4f}')
print(f'  Test RMSE        : {final_te["rmse"]:.4f}')
print(f'  Test R²          : {final_te["r2"]:.4f}')
print('=' * 60)

log.info(f'Best Epoch={best_epoch}  stopped_epoch={stopped_epoch}')
log.info(f'Test MAE={final_te["mae"]:.4f}')
log.info(f'Test RMSE={final_te["rmse"]:.4f}')
log.info(f'Test R²={final_te["r2"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
pd.DataFrame([{
    'best_epoch'   : best_epoch,
    'stopped_epoch': stopped_epoch,
    'val_mae'      : best_val_mae,
    'val_rmse'     : best_val_rmse,
    'val_r2'       : best_val_r2,
    'test_mae'     : final_te['mae'],
    'test_rmse'    : final_te['rmse'],
    'test_r2'      : final_te['r2'],
}]).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV saved → {summary_path}')
log.info('✓ All done!')

sys.stdout.restore()
print('✓ Log file closed and saved.')
