import os

# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════

# ── Dataset path ──────────────────────────────────────────────────────────
DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"

# ── Single-view: camera to use (0=front, 1=side, 2=oblique) ──────────────
CAMERA_ID     = 0

# ── NPZ key — confirmed from Cell 4 output: 'keypoints_3d' ───────────────
NPZ_KEY       = "keypoints_3d"

# ── Skeleton ──────────────────────────────────────────────────────────────
NUM_JOINTS    = 17
NUM_CLASSES   = 10   # E0–E9

# ── Sequence ──────────────────────────────────────────────────────────────
TARGET_FRAMES = 100

# ── Training ──────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.70   # 70% train
VAL_RATIO     = 0.15   # 15% val
# remaining 15% → test
EPOCHS        = 200
BATCH_SIZE    = 16
LR            = 1e-3
WEIGHT_DECAY  = 1e-4

# ── Output folder (saved to masa) ─────────────────────────────────────────
OUT_DIR       = "/mvdlph/masa/GCN_SingleView_Results"

print('✓ Configuration loaded')
print(f'  DATASET_DIR : {DATASET_DIR}')
print(f'  NPZ_KEY     : {NPZ_KEY}')
print(f'  CAMERA_ID   : C{CAMERA_ID}')
print(f'  EXISTS      : {os.path.exists(DATASET_DIR)}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 2 — Imports & output folders
# ══════════════════════════════════════════════════════════════════════════

import os, re, glob, json, logging, datetime, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # ← no display on server, must be before pyplot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score   # ← r2_score added

# ── Create output folders ─────────────────────────────────────────────────
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
LOGS_DIR  = os.path.join(OUT_DIR, "logs")
for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Libraries imported")
print("✓ Output folders created:")
for d in [PLOTS_DIR, LOGS_DIR]:
    print("  ", d)


# ══════════════════════════════════════════════════════════════════════════
# Cell 3 — Explore dataset folder & one NPZ file
# ══════════════════════════════════════════════════════════════════════════

all_npz = sorted(glob.glob(os.path.join(DATASET_DIR, '**/*.npz'), recursive=True))
print(f'Total NPZ files found : {len(all_npz)}')

if len(all_npz) == 0:
    print('\n❌ No NPZ files found! Checking folder contents...')
    print('Folder contents:')
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
    print('\n▶ Update NPZ_KEY in Cell 1 if the key name differs from "keypoints_3d"')


# ══════════════════════════════════════════════════════════════════════════
# Cell 4 — Load CSV labels
# ══════════════════════════════════════════════════════════════════════════

import io, pandas as pd

df_csv = None

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'rb') as f:
        raw = f.read()
    print(f'File size     : {len(raw)} bytes')
    print(f'First 8 bytes : {raw[:8]}')
    for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'cp1252']:
        try:
            text = raw.decode(enc)
            tmp  = pd.read_csv(io.StringIO(text))
            tmp.columns = tmp.columns.str.strip()
            if 'exercise' in tmp.columns:
                df_csv = tmp
                print(f'\n✓ CSV loaded with encoding: {enc}')
                break
            else:
                print(f'  {enc}: wrong columns → {tmp.columns.tolist()[:4]}')
        except Exception as e:
            print(f'  {enc}: {e}')
else:
    print(f'⚠️  CSV not found at: {CSV_PATH}')

if df_csv is None:
    raise FileNotFoundError(
        f'\n❌ CSV not found or could not be loaded from: {CSV_PATH}\n'
        'Please check CSV_PATH in Cell 1 and re-run.'
    )

print(f'\nColumns : {df_csv.columns.tolist()}')
print(f'Shape   : {df_csv.shape}')
print(df_csv.to_string())


# ══════════════════════════════════════════════════════════════════════════
# Cell 5 — Logging setup
# ══════════════════════════════════════════════════════════════════════════

import sys

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(LOGS_DIR, f"training_{timestamp}.log")

class Tee:
    """
    Redirects sys.stdout so that every print() goes to BOTH the
    console (original stdout) and the log file simultaneously.
    Call Tee.restore() at the end to close the file handle cleanly.
    """
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
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers= [
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("GCN-SingleView")
log.info("=" * 70)
log.info("ST-GCN Single-View Baseline | BZU Physiotherapy Dataset")
log.info(f"Camera : C{CAMERA_ID}  |  Split : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int((1-TRAIN_RATIO-VAL_RATIO)*100)}  |  Epochs : {EPOCHS}")
log.info(f"Log file : {log_file}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader
# ══════════════════════════════════════════════════════════════════════════

# Filename pattern:
#   E4_P10_T6_C2_seg9_MMPose_human3d_motionbert_3D.npz
#    ↑   ↑   ↑  ↑   ↑
#    E   P   T  C   seg

def parse_filename(fpath):
    base = os.path.basename(fpath)
    m = re.match(r"E(\d+)_P(\d+)_T(\d+)_C(\d+)_seg(\d+)", base)
    if m is None:
        return None
    return {
        "exercise"   : int(m.group(1)),
        "person"     : f"P{m.group(2)}",
        "trial_num"  : int(m.group(3)),
        "trial_id"   : f"T{m.group(3)}",
        "camera"     : int(m.group(4)),
        "segment"    : int(m.group(5)),
        "filepath"   : fpath,
    }


def load_skeleton(fpath, key=NPZ_KEY):
    """
    Load a skeleton array from an NPZ file.
    Always returns shape (T, 17, 3) float32.

    Human3.6M joint ordering (MotionBERT):
      0:Hip  1:RHip  2:RKnee  3:RAnkle  4:LHip  5:LKnee  6:LAnkle
      7:Spine  8:Thorax  9:Neck  10:Head
      11:LShoulder  12:LElbow  13:LWrist
      14:RShoulder  15:RElbow  16:RWrist
    """
    data = np.load(fpath, allow_pickle=True)
    arr  = data[key] if key in data else data[list(data.keys())[0]]
    arr  = arr.astype(np.float32)

    # Handle shape variants from different MMPose export versions
    if arr.ndim == 2:               # (T, 51) → (T, 17, 3)
        arr = arr.reshape(arr.shape[0], 17, 3)
    elif arr.ndim == 4:             # (1, T, 17, 3) → (T, 17, 3)
        arr = arr.squeeze(0)
    elif arr.ndim == 3:             # already (T, 17, 3)
        pass

    # Fix MotionBERT axis convention (Z=up → Y=up)
    arr = arr[:, :, [0, 2, 1]]   # swap Y↔Z
    arr[:, :, 1] *= -1            # flip Y so up is positive
    return arr


print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build dataset index
# ══════════════════════════════════════════════════════════════════════════

def build_index(dataset_dir, camera_id, df_csv):
    """
    Scan all NPZ files for the chosen camera and merge quality labels from CSV.
    Split key = (person, trial_id) to avoid leakage between folds.
    """
    df_csv = df_csv.copy()
    df_csv.columns = df_csv.columns.str.strip()

    all_files = sorted(glob.glob(
        os.path.join(dataset_dir, '**/*.npz'), recursive=True))
    print(f'NPZ files found (all cameras) : {len(all_files)}')

    if len(all_files) == 0:
        print('❌ No NPZ files found — check DATASET_DIR in Cell 1')
        return pd.DataFrame()

    records = []
    for fpath in all_files:
        meta = parse_filename(fpath)
        if meta is None:
            continue
        if camera_id is not None and meta['camera'] != camera_id:
            continue

        ex_str = f"E{meta['exercise']}"
        tr_str = meta['trial_id']
        pr_str = meta['person']

        row = df_csv[
            (df_csv['exercise'] == ex_str) &
            (df_csv['person']   == pr_str) &
            (df_csv['trial']    == tr_str)
        ]
        quality = float(row.iloc[0]['mean']) if len(row) > 0 else np.nan

        meta['quality']   = quality
        meta['trial_key'] = f"{pr_str}_{tr_str}"
        records.append(meta)

    if len(records) == 0:
        print(f'❌ No records for camera C{camera_id}. Try CAMERA_ID=1 or 2.')
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Fill missing quality scores with split mean
    correct_mean   = df[df['trial_num'] <= 2]['quality'].mean()
    erroneous_mean = df[df['trial_num'] >= 3]['quality'].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5
    df.loc[(df['quality'].isna()) & (df['trial_num'] <= 2),  'quality'] = correct_mean
    df.loc[(df['quality'].isna()) & (df['trial_num'] >= 3), 'quality'] = erroneous_mean

    print(f'\n✓ Total samples  (C{camera_id}) : {len(df)}')
    print(f'✓ Unique trials              : {df["trial_key"].nunique()}')
    print(f'✓ Quality mean ± std         : {df["quality"].mean():.3f} ± {df["quality"].std():.3f}')
    print(f'\n✓ Exercise distribution:')
    print(df['exercise'].value_counts().sort_index())
    return df


df_index = build_index(DATASET_DIR, CAMERA_ID, df_csv)

if len(df_index) > 0:
    print('\n── Sample rows ──')
    print(df_index[['exercise','person','trial_id','segment','quality']].head(15))

print(df_index['camera'].value_counts().sort_index())
print(f'\nTotal samples: {len(df_index)}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 8 — Skeleton visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),       # right leg
    (0, 4), (4, 5), (5, 6),       # left leg
    (0, 7), (7, 8), (8, 9),       # spine → neck
    (9, 10),                       # neck → head
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
]

JOINT_NAMES = [
    'Hip', 'R-Hip', 'R-Knee', 'R-Ankle',
    'L-Hip', 'L-Knee', 'L-Ankle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'L-Shoulder', 'L-Elbow', 'L-Wrist',
    'R-Shoulder', 'R-Elbow', 'R-Wrist',
]

JOINT_COLORS = {
    'head'  : [9, 10],
    'arms'  : [11, 12, 13, 14, 15, 16],
    'torso' : [0, 7, 8],
    'legs'  : [1, 2, 3, 4, 5, 6],
}
PART_COLOR = {
    'head':'gold','arms':'dodgerblue','torso':'limegreen','legs':'tomato'
}


def plot_skeleton_3d(skel, frame_idx=0, title='Skeleton Sanity Check', save_path=None):
    pts = skel[frame_idx]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    views = [
        (axes[0], x,  y,  'Front View  (X–Y)', 'X (left/right)', 'Y (up/down)', True),
        (axes[1], z,  y,  'Side View   (Z–Y)', 'Z (depth)',       'Y (up/down)', True),
        (axes[2], x, -z,  'Top View    (X–Z)', 'X (left/right)', '-Z (forward)', False),
    ]
    for ax, hx, hy, view_title, xlabel, ylabel, do_invert_y in views:
        for (i, j) in SKELETON_EDGES:
            ax.plot([hx[i], hx[j]], [hy[i], hy[j]], color='dimgray', lw=2, zorder=1)
        for part, idxs in JOINT_COLORS.items():
            ax.scatter(hx[idxs], hy[idxs], c=PART_COLOR[part], s=80, zorder=3,
                       edgecolors='black', linewidths=0.5, label=part)
        ax.set_title(view_title, fontweight='bold', fontsize=10)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        if do_invert_y:
            ax.invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Skeleton plot saved → {save_path}')
    plt.close()


def plot_skeleton_frames(skel, n_frames=5, title='Skeleton Motion', save_path=None):
    T    = skel.shape[0]
    idxs = np.linspace(0, T-1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(4*n_frames, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    for col, fi in enumerate(idxs):
        ax  = axes[col]
        pts = skel[fi]
        x, y = pts[:,0], pts[:,1]
        for (i, j) in SKELETON_EDGES:
            ax.plot([x[i],x[j]], [y[i],y[j]], color='dimgray', lw=2)
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

idx = 10

sample_skel = load_skeleton(df_index.iloc[idx]['filepath'])

print(df_index.iloc[idx][['person', 'exercise', 'trial_id', 'segment', 'filepath']])
print(f'Skeleton shape : {sample_skel.shape}')
print(f'X range : [{sample_skel[:,:,0].min():.3f}, {sample_skel[:,:,0].max():.3f}]')
print(f'Y range : [{sample_skel[:,:,1].min():.3f}, {sample_skel[:,:,1].max():.3f}]')
print(f'Z range : [{sample_skel[:,:,2].min():.3f}, {sample_skel[:,:,2].max():.3f}]')

plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton · {df_index.iloc[idx]['trial_key']} · Exercise E{df_index.iloc[idx]['exercise']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png')
)

plot_skeleton_frames(
    sample_skel, n_frames=5,
    title=f"Motion Sequence · {df_index.iloc[idx]['trial_key']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_motion.png')
)


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset
# ══════════════════════════════════════════════════════════════════════════

class BZUDataset(Dataset):
    def __init__(self, df, target_frames=TARGET_FRAMES, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        skel    = load_skeleton(row["filepath"])
        skel    = self._normalise_length(skel)
        if self.augment:
            skel = self._augment(skel)
        skel    = torch.tensor(skel,                dtype=torch.float32)
        ex_id   = torch.tensor(row["exercise"],     dtype=torch.long)
        quality = torch.tensor(row["quality"],      dtype=torch.float32)
        return skel, ex_id, quality

    def _normalise_length(self, skel):
        T = skel.shape[0]
        if T == self.target_frames:
            return skel
        old_indices = np.linspace(0, 1, T)
        new_indices = np.linspace(0, 1, self.target_frames)
        new_skel = np.zeros((self.target_frames, skel.shape[1], skel.shape[2]), dtype=np.float32)
        for joint in range(skel.shape[1]):
            for axis in range(skel.shape[2]):
                new_skel[:, joint, axis] = np.interp(new_indices, old_indices, skel[:, joint, axis])
        return new_skel

    def _augment(self, skel):
        T     = skel.shape[0]
        speed = np.random.uniform(0.8, 1.2)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = self._normalise_length(skel[idxs])
        skel  += np.random.randn(*skel.shape).astype(np.float32) * 0.005
        if np.random.rand() < 0.5:
            skel[:, :, 0] *= -1.0
        return skel


print('✓ BZUDataset defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN Model
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
                      kernel_size=(t_kernel, 1), padding=(pad, 0),
                      stride=(stride, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.res = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride,1), bias=False),
                nn.BatchNorm2d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        B, C, T, J = x.shape
        xs = x.permute(0,2,3,1).reshape(B*T, J, C)
        xs = torch.bmm(self.A.unsqueeze(0).expand(B*T,-1,-1), xs)
        xs = self.W_s(xs).reshape(B, T, J, -1).permute(0,3,1,2)
        xs = F.relu(self.bn_s(xs))
        return F.relu(self.t_conv(xs) + self.res(x))


class STGCN_SingleView(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        A = build_adj(NUM_JOINTS, SKELETON_EDGES)
        self.data_bn = nn.BatchNorm1d(NUM_JOINTS * 3)
        cfg = [(3,64,1),(64,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1)]
        self.blocks  = nn.ModuleList(
            [STGCNBlock(ic, oc, A, stride=s, dropout=dropout) for ic,oc,s in cfg])
        self.drop    = nn.Dropout(dropout)
        self.cls_head = nn.Linear(256, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(256,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, J, C = x.shape
        xbn = x.reshape(B*T, J*C)
        xbn = self.data_bn(xbn)
        x   = xbn.reshape(B, T, J, C)
        x   = x.permute(0,3,1,2)
        for blk in self.blocks:
            x = blk(x)
        x   = x.mean(dim=[2,3])
        x   = self.drop(x)
        cls = self.cls_head(x)
        qua = 1.0 + 4.0*torch.sigmoid(self.reg_head(x))
        return cls, qua


print('✓ ST-GCN model defined (single clean forward method)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Skeleton normalisation & training helpers
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    hip      = (x[:,:,1:2,:] + x[:,:,4:5,:]) / 2.0
    x        = x - hip
    shoulder = (x[:,:,11:12,:] + x[:,:,14:15,:]) / 2.0
    torso_h  = shoulder.norm(dim=-1, keepdim=True)
    torso_h  = torso_h.mean(dim=1, keepdim=True).clamp(min=1e-6)
    return x / torso_h


def run_epoch(model, loader, optimiser, cls_fn, reg_fn,
              is_train=True, cls_w=1.0, reg_w=0.5):
    model.train() if is_train else model.eval()
    tot = dict(loss=0, cls=0, reg=0, correct=0, total=0)
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, ex_ids, qualities in loader:
            skels     = centre_and_scale(skels.to(DEVICE))
            ex_ids    = ex_ids.to(DEVICE)
            qualities = qualities.to(DEVICE)

            cls_logits, qpred = model(skels)
            l_cls = cls_fn(cls_logits, ex_ids)
            l_reg = reg_fn(qpred.squeeze(1), qualities)
            loss  = cls_w * l_cls + reg_w * l_reg

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            tot['loss']    += loss.item()
            tot['cls']     += l_cls.item()
            tot['reg']     += l_reg.item()
            tot['correct'] += (cls_logits.argmax(1) == ex_ids).sum().item()
            tot['total']   += ex_ids.size(0)
            q_true.extend(qualities.cpu().numpy())
            q_pred.extend(qpred.squeeze(1).detach().cpu().numpy())

    n    = max(1, len(loader))
    qt   = np.array(q_true)
    qp   = np.array(q_pred)
    rmse = float(np.sqrt(np.mean((qt - qp)**2)))
    mae  = float(np.mean(np.abs(qt - qp)))
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0   # ← R² added

    return {
        'loss'    : tot['loss'] / n,
        'cls_loss': tot['cls']  / n,
        'reg_loss': tot['reg']  / n,
        'accuracy': tot['correct'] / max(1, tot['total']) * 100,
        'rmse'    : rmse,
        'mae'     : mae,
        'r2'      : r2,   # ← R² added
    }


print('✓ centre_and_scale and run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Trial-based single train / val / test split
# ══════════════════════════════════════════════════════════════════════════

def get_trial_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=42):
    trial_keys = df['trial_key'].unique()
    np.random.seed(random_state)
    np.random.shuffle(trial_keys)

    n_total = len(trial_keys)
    print(f'Total unique trials : {n_total}')

    n_test  = max(1, int((1.0 - train_ratio - val_ratio) * n_total))
    n_val   = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val - n_test

    if n_train < 1:
        raise ValueError(f'Not enough trials for the requested split ratios '
                         f'(train={n_train}, val={n_val}, test={n_test}).')

    train_keys = trial_keys[:n_train]
    val_keys   = trial_keys[n_train:n_train + n_val]
    test_keys  = trial_keys[n_train + n_val:]

    train_df = df[df['trial_key'].isin(train_keys)].reset_index(drop=True)
    val_df   = df[df['trial_key'].isin(val_keys)].reset_index(drop=True)
    test_df  = df[df['trial_key'].isin(test_keys)].reset_index(drop=True)

    print(f'  Train : {len(train_keys)} trials → {len(train_df)} samples')
    print(f'  Val   : {len(val_keys)}   trials → {len(val_df)} samples')
    print(f'  Test  : {len(test_keys)}  trials → {len(test_df)} samples')
    return train_df, val_df, test_df


train_df, val_df, test_df = get_trial_split(df_index)
print('\n✓ Single train/val/test split ready')


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    """Save figure to disk (no display on server)."""
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def plot_loss_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Loss Curves', fontsize=14, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    axes[0].plot(epochs, history['test_loss'],  label='Test',       color='green', linestyle='--')
    axes[0].set_title('Total Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_cls_loss'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_cls_loss'],   label='Validation', color='darkorange')
    axes[1].set_title('Classification Loss  (CrossEntropy)', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['train_reg_loss'], label='Train',      color='steelblue')
    axes[2].plot(epochs, history['val_reg_loss'],   label='Validation', color='darkorange')
    axes[2].set_title('Quality Score Loss  (SmoothL1)', fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curves.png'))


def plot_accuracy_rmse(history, save_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Accuracy & RMSE', fontsize=14, fontweight='bold')

    axes[0].plot(epochs, history['train_acc'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_acc'],   label='Validation', color='darkorange')
    axes[0].set_title('Exercise Classification Accuracy (%)', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_rmse'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_rmse'],   label='Validation', color='darkorange')
    axes[1].set_title('Quality Score RMSE', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('RMSE')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'accuracy_rmse.png'))


def plot_r2_mae(history, save_dir):
    """Plot R² and MAE curves for train and validation sets."""
    epochs = range(1, len(history['train_r2']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Quality Score Regression Metrics', fontsize=14, fontweight='bold')

    # ── R² ────────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    axes[0].axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (R²=1)')
    axes[0].axhline(0.0, color='red',  linestyle=':', linewidth=1, label='Baseline (R²=0)')
    axes[0].set_title('R² Score  (higher = better)', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('R²')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── MAE ───────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history['train_mae'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_mae'],   label='Validation', color='darkorange')
    axes[1].set_title('MAE  (lower = better)', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'r2_mae_curves.png'))


def plot_regression_scatter(q_true, q_pred, split_name='Test', save_dir=None):
    """
    Scatter plot of true vs predicted quality scores for the test set.
    Shows the identity line and annotates R², MAE, RMSE.
    """
    qt = np.array(q_true)
    qp = np.array(q_pred)
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp)**2)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(qt, qp, alpha=0.6, edgecolors='black', linewidths=0.4,
               color='steelblue', s=60, label='Samples')

    lo, hi = min(qt.min(), qp.min()) - 0.2, max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True Quality Score',      fontsize=12)
    ax.set_ylabel('Predicted Quality Score', fontsize=12)
    ax.set_title(f'{split_name} Set — True vs Predicted Quality',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    textstr = f'R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}'
    props   = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'regression_scatter_{split_name.lower()}.png')
        save_and_show(fig, path)
    else:
        plt.close(fig)


def plot_confusion_matrix(all_true, all_pred, save_dir):
    labels     = sorted(list(set(all_true) | set(all_pred)))
    label_names= [f'E{i}' for i in labels]
    cm         = confusion_matrix(all_true, all_pred, labels=labels)
    disp       = ConfusionMatrixDisplay(cm, display_labels=label_names)
    fig, ax    = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'confusion_matrix.png'))


print('✓ Plotting helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Training Loop (single run, no cross-validation)
# ══════════════════════════════════════════════════════════════════════════

actual_classes = sorted(df_index['exercise'].unique())
n_classes      = len(actual_classes)
ex_map         = {v: i for i, v in enumerate(actual_classes)}
rev_map        = {i: v for v, i in ex_map.items()}
print(f'Classes in data : E{actual_classes}  →  mapped 0..{n_classes-1}')
print(f'Model output size: {n_classes}')

log.info('='*70)
log.info('STARTING TRAINING (single train/val/test split)')
log.info('='*70)

cls_fn = nn.CrossEntropyLoss()
reg_fn = nn.SmoothL1Loss()


def make_ds(df, aug):
    d = df.copy()
    d['exercise'] = d['exercise'].map(ex_map)
    return BZUDataset(d, augment=aug)


train_loader = DataLoader(make_ds(train_df, True),
                          batch_size=min(BATCH_SIZE, len(train_df)),
                          shuffle=True, num_workers=0, pin_memory=(DEVICE=='cuda'))
val_loader   = DataLoader(make_ds(val_df, False),
                          batch_size=min(BATCH_SIZE, len(val_df)),
                          shuffle=False, num_workers=0, pin_memory=(DEVICE=='cuda'))
test_loader  = DataLoader(make_ds(test_df, False),
                          batch_size=min(BATCH_SIZE, len(test_df)),
                          shuffle=False, num_workers=0, pin_memory=(DEVICE=='cuda'))

model     = STGCN_SingleView(num_classes=n_classes, dropout=0.3).to(DEVICE)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=EPOCHS, eta_min=1e-5)

# ── History dict now includes r2 ──────────────────────────────────────────
history = {k: [] for k in [
    'train_loss','val_loss','test_loss',
    'train_cls_loss','val_cls_loss',
    'train_reg_loss','val_reg_loss',
    'train_acc','val_acc',
    'train_rmse','val_rmse',
    'train_mae','val_mae',
    'train_r2','val_r2',      # ← R² added
]}

best_val_acc   = 0.0
best_model_wts = None
best_epoch     = 1
best_val_rmse  = float('inf')
best_val_mae   = float('inf')
best_val_r2    = -float('inf')   # ← track best R² too

print(f'\n{"═"*60}')
print(f'  Train:{len(train_df)}  Val:{len(val_df)}  Test:{len(test_df)}')
print(f'{"═"*60}')
log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')

for epoch in range(1, EPOCHS+1):
    tr = run_epoch(model, train_loader, optimiser, cls_fn, reg_fn, is_train=True,  cls_w=1.0, reg_w=0.2)
    vl = run_epoch(model, val_loader,   optimiser, cls_fn, reg_fn, is_train=False, cls_w=1.0, reg_w=0.2)
    te = run_epoch(model, test_loader,  optimiser, cls_fn, reg_fn, is_train=False, cls_w=1.0, reg_w=0.2)
    scheduler.step()

    for k, v in [
        ('train_loss',    tr['loss']),     ('val_loss',    vl['loss']),
        ('test_loss',     te['loss']),     ('train_cls_loss', tr['cls_loss']),
        ('val_cls_loss',  vl['cls_loss']), ('train_reg_loss', tr['reg_loss']),
        ('val_reg_loss',  vl['reg_loss']), ('train_acc',   tr['accuracy']),
        ('val_acc',       vl['accuracy']), ('train_rmse',  tr['rmse']),
        ('val_rmse',      vl['rmse']),     ('train_mae',   tr['mae']),
        ('val_mae',       vl['mae']),      ('train_r2',    tr['r2']),   # ← R²
        ('val_r2',        vl['r2']),                                     # ← R²
    ]:
        history[k].append(v)

    msg = (f'  Ep {epoch:3d}/{EPOCHS} | '
           f'Train loss={tr["loss"]:.3f} acc={tr["accuracy"]:.1f}% '
           f'mae={tr["mae"]:.3f} r2={tr["r2"]:.3f} | '
           f'Val   loss={vl["loss"]:.3f} acc={vl["accuracy"]:.1f}% '
           f'mae={vl["mae"]:.3f} r2={vl["r2"]:.3f} | '
           f'Test  acc={te["accuracy"]:.1f}% mae={te["mae"]:.3f} r2={te["r2"]:.3f}')
    print(msg); log.info(msg)

    if vl['accuracy'] > best_val_acc:
        best_val_acc   = vl['accuracy']
        best_val_rmse  = vl['rmse']
        best_val_mae   = vl['mae']
        best_val_r2    = vl['r2']   # ← save R² at best checkpoint
        best_epoch     = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f'    ✓ Best weights updated  val_acc={best_val_acc:.1f}%  '
              f'val_mae={best_val_mae:.4f}  val_r2={best_val_r2:.4f}')

print('\n✓ Training complete!')

# ── Final evaluation with best weights ────────────────────────────────────
model.load_state_dict(best_model_wts)
final_te = run_epoch(model, test_loader, optimiser, cls_fn, reg_fn,
                     is_train=False, cls_w=1.0, reg_w=0.2)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────────────')
print(f'  Accuracy : {final_te["accuracy"]:.2f}%')
print(f'  RMSE     : {final_te["rmse"]:.4f}')
print(f'  MAE      : {final_te["mae"]:.4f}')
print(f'  R²       : {final_te["r2"]:.4f}')    # ← R² printed
print(f'  Loss     : {final_te["loss"]:.4f}')

# ── Confusion matrix + collect quality predictions for scatter ─────────────
model.eval()
all_true_cls, all_pred_cls = [], []
all_true_q,   all_pred_q   = [], []

with torch.no_grad():
    for skels, ex_ids, qualities in test_loader:
        skels = centre_and_scale(skels.to(DEVICE))
        cls_logits, qpred = model(skels)
        all_true_cls.extend([rev_map[e] for e in ex_ids.numpy()])
        all_pred_cls.extend([rev_map[p] for p in cls_logits.argmax(1).cpu().numpy()])
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(qpred.squeeze(1).cpu().numpy())

# ── Save all plots ─────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR)
plot_accuracy_rmse(history, PLOTS_DIR)
plot_r2_mae(history, PLOTS_DIR)                                      # ← new plot
plot_regression_scatter(all_true_q, all_pred_q,
                        split_name='Test', save_dir=PLOTS_DIR)       # ← new scatter
plot_confusion_matrix(all_true_cls, all_pred_cls, PLOTS_DIR)

# ── Save history JSON ──────────────────────────────────────────────────────
json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

print('='*60)
print('  TRAINING SUMMARY — ST-GCN Single View')
print('='*60)
print(f'  Best Epoch       : {best_epoch}')
print(f'  Best Val Acc     : {best_val_acc:.2f}%')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')    # ← R² added
print('─'*60)
print(f'  Test Accuracy    : {final_te["accuracy"]:.2f}%')
print(f'  Test RMSE        : {final_te["rmse"]:.4f}')
print(f'  Test MAE         : {final_te["mae"]:.4f}')
print(f'  Test R²          : {final_te["r2"]:.4f}')  # ← R² added
print('='*60)

log.info(f'Best Epoch    : {best_epoch}')
log.info(f'Test Accuracy : {final_te["accuracy"]:.2f}%')
log.info(f'Test RMSE     : {final_te["rmse"]:.4f}')
log.info(f'Test MAE      : {final_te["mae"]:.4f}')
log.info(f'Test R²       : {final_te["r2"]:.4f}')    # ← R² logged

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
pd.DataFrame([{
    'best_epoch'  : best_epoch,
    'val_acc'     : best_val_acc,
    'val_rmse'    : best_val_rmse,
    'val_mae'     : best_val_mae,
    'val_r2'      : best_val_r2,          # ← R² added
    'test_acc'    : final_te['accuracy'],
    'test_rmse'   : final_te['rmse'],
    'test_mae'    : final_te['mae'],
    'test_r2'     : final_te['r2'],        # ← R² added
}]).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV saved → {summary_path}')

log.info('\n✓ All done! Check /mvdlph/masa/ for plots and logs.')

sys.stdout.restore()
print('✓ Log file closed and saved.')
