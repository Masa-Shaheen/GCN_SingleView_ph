import os

# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"
CAMERA_ID     = 0
NPZ_KEY       = "keypoints_3d"
NUM_JOINTS    = 17
NUM_CLASSES   = 10
TARGET_FRAMES = 100
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EPOCHS        = 200
BATCH_SIZE    = 16
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_SingleView_Results"

# ══════════════════════ EARLY STOPPING CONFIG ═════════════════════════════
PATIENCE      = 20    # عدد epochs بدون تحسن قبل الوقوف
MIN_DELTA     = 1e-4  # أقل تحسن يُعتبر "تحسن حقيقي"
# ═════════════════════════════════════════════════════════════════════════

print('✓ Configuration loaded')
print(f'  DATASET_DIR : {DATASET_DIR}')
print(f'  NPZ_KEY     : {NPZ_KEY}')
print(f'  CAMERA_ID   : C{CAMERA_ID}')
print(f'  EXISTS      : {os.path.exists(DATASET_DIR)}')
print(f'  PATIENCE    : {PATIENCE} epochs')


# ══════════════════════════════════════════════════════════════════════════
# Cell 2 — Imports & output folders
# ══════════════════════════════════════════════════════════════════════════

import os, re, glob, json, logging, datetime, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score

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

import io

df_csv = None

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'rb') as f:
        raw = f.read()
    print(f'File size     : {len(raw)} bytes')
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
    raise FileNotFoundError(f'\n❌ CSV not found or could not be loaded from: {CSV_PATH}')

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
log = logging.getLogger("GCN-SingleView")
log.info("=" * 70)
log.info("ST-GCN Single-View Baseline | BZU Physiotherapy Dataset")
log.info(f"Camera : C{CAMERA_ID}  |  Split : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}"
         f"/{int((1-TRAIN_RATIO-VAL_RATIO)*100)}  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
log.info(f"Log file : {log_file}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader
# ══════════════════════════════════════════════════════════════════════════

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
    Load skeleton → always returns (T, 17, 3) float32.
    Handles shape variants: (T,51), (1,T,17,3), (T,17,3).
    Fixes MotionBERT axis convention: Z=up → Y=up.
    """
    data = np.load(fpath, allow_pickle=True)
    arr  = data[key] if key in data else data[list(data.keys())[0]]
    arr  = arr.astype(np.float32)

    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], 17, 3)
    elif arr.ndim == 4:
        arr = arr.squeeze(0)
    # ndim == 3 → already correct

    arr = arr[:, :, [0, 2, 1]]   # swap Y↔Z
    arr[:, :, 1] *= -1            # flip Y so up = positive
    return arr


print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build dataset index
# ══════════════════════════════════════════════════════════════════════════

def build_index(dataset_dir, camera_id, df_csv):
    df_csv = df_csv.copy()
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

    if not records:
        print(f'❌ No records for camera C{camera_id}. Try CAMERA_ID=1 or 2.')
        return pd.DataFrame()

    df = pd.DataFrame(records)

    correct_mean   = df[df['trial_num'] <= 2]['quality'].mean()
    erroneous_mean = df[df['trial_num'] >= 3]['quality'].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5
    df.loc[(df['quality'].isna()) & (df['trial_num'] <= 2), 'quality'] = correct_mean
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
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
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
    'head'  : [9, 10],
    'arms'  : [11, 12, 13, 14, 15, 16],
    'torso' : [0, 7, 8],
    'legs'  : [1, 2, 3, 4, 5, 6],
}
PART_COLOR = {
    'head':'gold', 'arms':'dodgerblue', 'torso':'limegreen', 'legs':'tomato'
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
        skel    = torch.tensor(skel,            dtype=torch.float32)
        ex_id   = torch.tensor(row["exercise"], dtype=torch.long)
        quality = torch.tensor(row["quality"],  dtype=torch.float32)
        return skel, ex_id, quality

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
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1), bias=False),
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
        self.data_bn  = nn.BatchNorm1d(NUM_JOINTS * 3)
        cfg = [(3,64,1),(64,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1)]
        self.blocks   = nn.ModuleList(
            [STGCNBlock(ic, oc, A, stride=s, dropout=dropout) for ic, oc, s in cfg])
        self.drop     = nn.Dropout(dropout)
        self.cls_head = nn.Linear(256, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
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
        x   = xbn.reshape(B, T, J, C).permute(0,3,1,2)   # (B,C,T,J)
        for blk in self.blocks:
            x = blk(x)
        x   = x.mean(dim=[2, 3])
        x   = self.drop(x)
        cls = self.cls_head(x)
        qua = 1.0 + 4.0 * torch.sigmoid(self.reg_head(x))   # → [1, 5]
        return cls, qua


print('✓ ST-GCN model defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Normalisation & run_epoch
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """Root-relative normalisation + torso-height scaling. x: (B,T,J,3)"""
    hip     = (x[:,:,1:2,:] + x[:,:,4:5,:]) / 2.0
    x       = x - hip
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
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0

    return {
        'loss'    : tot['loss']    / n,
        'cls_loss': tot['cls']     / n,
        'reg_loss': tot['reg']     / n,
        'accuracy': tot['correct'] / max(1, tot['total']) * 100,
        'rmse'    : rmse,
        'mae'     : mae,
        'r2'      : r2,
    }


print('✓ centre_and_scale and run_epoch defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Trial-based split
# ══════════════════════════════════════════════════════════════════════════

def get_trial_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=42):
    trial_keys = df['trial_key'].unique()
    np.random.seed(random_state)
    np.random.shuffle(trial_keys)

    n_total = len(trial_keys)
    n_test  = max(1, int((1.0 - train_ratio - val_ratio) * n_total))
    n_val   = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val - n_test

    if n_train < 1:
        raise ValueError(f'Not enough trials: train={n_train}, val={n_val}, test={n_test}')

    train_keys = trial_keys[:n_train]
    val_keys   = trial_keys[n_train:n_train + n_val]
    test_keys  = trial_keys[n_train + n_val:]

    train_df = df[df['trial_key'].isin(train_keys)].reset_index(drop=True)
    val_df   = df[df['trial_key'].isin(val_keys)].reset_index(drop=True)
    test_df  = df[df['trial_key'].isin(test_keys)].reset_index(drop=True)

    print(f'Total unique trials : {n_total}')
    print(f'  Train : {len(train_keys)} trials → {len(train_df)} samples')
    print(f'  Val   : {len(val_keys)}   trials → {len(val_df)} samples')
    print(f'  Test  : {len(test_keys)}  trials → {len(test_df)} samples')
    return train_df, val_df, test_df


train_df, val_df, test_df = get_trial_split(df_index)
print('\n✓ Single train/val/test split ready')


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers  ← TEST added to all curves
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def plot_loss_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Loss Curves', fontsize=14, fontweight='bold')

    # ── Total loss ────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    axes[0].plot(epochs, history['test_loss'],  label='Test',       color='green', linestyle='--')
    axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # ── Classification loss ───────────────────────────────────────────────
    axes[1].plot(epochs, history['train_cls_loss'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_cls_loss'],   label='Validation', color='darkorange')
    axes[1].plot(epochs, history['test_cls_loss'],  label='Test',       color='green', linestyle='--')
    axes[1].set_title('Classification Loss  (CrossEntropy)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # ── Regression loss ───────────────────────────────────────────────────
    axes[2].plot(epochs, history['train_reg_loss'], label='Train',      color='steelblue')
    axes[2].plot(epochs, history['val_reg_loss'],   label='Validation', color='darkorange')
    axes[2].plot(epochs, history['test_reg_loss'],  label='Test',       color='green', linestyle='--')
    axes[2].set_title('Quality Score Loss  (SmoothL1)')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'loss_curves.png'))


def plot_accuracy_rmse(history, save_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Accuracy & RMSE', fontsize=14, fontweight='bold')

    # ── Accuracy ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['train_acc'],  label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_acc'],    label='Validation', color='darkorange')
    axes[0].plot(epochs, history['test_acc'],   label='Test',       color='green', linestyle='--')
    axes[0].set_title('Exercise Classification Accuracy (%)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # ── RMSE ──────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history['train_rmse'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_rmse'],   label='Validation', color='darkorange')
    axes[1].plot(epochs, history['test_rmse'],  label='Test',       color='green', linestyle='--')
    axes[1].set_title('Quality Score RMSE')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('RMSE')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'accuracy_rmse.png'))


def plot_r2_mae(history, save_dir):
    epochs = range(1, len(history['train_r2']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Quality Score Regression Metrics', fontsize=14, fontweight='bold')

    # ── R² ────────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    axes[0].plot(epochs, history['test_r2'],  label='Test',       color='green', linestyle='--')
    axes[0].axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect (R²=1)')
    axes[0].axhline(0.0, color='red',  linestyle=':', linewidth=1, label='Baseline (R²=0)')
    axes[0].set_title('R² Score  (higher = better)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('R²')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # ── MAE ───────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history['train_mae'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_mae'],   label='Validation', color='darkorange')
    axes[1].plot(epochs, history['test_mae'],  label='Test',       color='green', linestyle='--')
    axes[1].set_title('MAE  (lower = better)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'r2_mae_curves.png'))


def plot_regression_scatter(q_true, q_pred, split_name='Test', save_dir=None):
    qt = np.array(q_true)
    qp = np.array(q_pred)
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp)**2)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(qt, qp, alpha=0.6, edgecolors='black', linewidths=0.4,
               color='steelblue', s=60)
    lo = min(qt.min(), qp.min()) - 0.2
    hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True Quality Score', fontsize=12)
    ax.set_ylabel('Predicted Quality Score', fontsize=12)
    ax.set_title(f'{split_name} Set — True vs Predicted Quality',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    textstr = f'R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    if save_dir:
        save_and_show(fig, os.path.join(save_dir, f'regression_scatter_{split_name.lower()}.png'))
    else:
        plt.close(fig)


def plot_confusion_matrix(all_true, all_pred, save_dir):
    labels      = sorted(set(all_true) | set(all_pred))
    label_names = [f'E{i}' for i in labels]
    cm          = confusion_matrix(all_true, all_pred, labels=labels)
    disp        = ConfusionMatrixDisplay(cm, display_labels=label_names)
    fig, ax     = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'confusion_matrix.png'))


def plot_early_stop(history, stopped_epoch, best_epoch, save_dir):
    """Visual showing when early stopping fired and where the best epoch was."""
    epochs = range(1, len(history['val_acc']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_acc'], label='Train Acc',
            color='steelblue')
    ax.plot(epochs, history['val_acc'],   label='Val Acc',
            color='darkorange')
    ax.plot(epochs, history['test_acc'],  label='Test Acc',
            color='green', linestyle='--')
    ax.axvline(best_epoch,    color='purple', linestyle=':',
               linewidth=2, label=f'Best epoch ({best_epoch})')
    ax.axvline(stopped_epoch, color='red',    linestyle='--',
               linewidth=2, label=f'Early stop ({stopped_epoch})')
    ax.set_title('Accuracy + Early Stopping', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'early_stopping.png'))


print('✓ Plotting helpers defined (all curves now include Test)')


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Early Stopping class
# ══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Monitors val_accuracy (higher = better).
    Saves the best model weights internally.
    Call .step(val_acc, model) each epoch.
    .should_stop → True when patience is exhausted.
    """
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = -float('inf')
        self.counter    = 0
        self.best_wts   = None
        self.best_epoch = 1

    def step(self, val_acc, model, epoch):
        if val_acc > self.best_acc + self.min_delta:
            # Genuine improvement
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_wts   = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False   # don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # stop!
            return False

    @property
    def should_stop(self):
        return self.counter >= self.patience


print('✓ EarlyStopping defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Training Loop
# ══════════════════════════════════════════════════════════════════════════

actual_classes = sorted(df_index['exercise'].unique())
n_classes      = len(actual_classes)
ex_map         = {v: i for i, v in enumerate(actual_classes)}
rev_map        = {i: v for v, i in ex_map.items()}
print(f'Classes in data : E{actual_classes}  →  mapped 0..{n_classes-1}')

log.info('='*70)
log.info('STARTING TRAINING  (with Early Stopping)')
log.info('='*70)

cls_fn = nn.CrossEntropyLoss()
reg_fn = nn.SmoothL1Loss()


def make_ds(df, aug):
    d = df.copy()
    d['exercise'] = d['exercise'].map(ex_map)
    return BZUDataset(d, augment=aug)


train_loader = DataLoader(make_ds(train_df, True),
                          batch_size=min(BATCH_SIZE, len(train_df)),
                          shuffle=True,  num_workers=0,
                          pin_memory=(DEVICE=='cuda'))
val_loader   = DataLoader(make_ds(val_df, False),
                          batch_size=min(BATCH_SIZE, len(val_df)),
                          shuffle=False, num_workers=0,
                          pin_memory=(DEVICE=='cuda'))
test_loader  = DataLoader(make_ds(test_df, False),
                          batch_size=min(BATCH_SIZE, len(test_df)),
                          shuffle=False, num_workers=0,
                          pin_memory=(DEVICE=='cuda'))

model     = STGCN_SingleView(num_classes=n_classes, dropout=0.3).to(DEVICE)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=EPOCHS, eta_min=1e-5)
early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

# ── History — all three splits tracked ────────────────────────────────────
history = {k: [] for k in [
    'train_loss', 'val_loss',  'test_loss',
    'train_cls_loss', 'val_cls_loss', 'test_cls_loss',
    'train_reg_loss', 'val_reg_loss', 'test_reg_loss',
    'train_acc',  'val_acc',   'test_acc',
    'train_rmse', 'val_rmse',  'test_rmse',
    'train_mae',  'val_mae',   'test_mae',
    'train_r2',   'val_r2',    'test_r2',
]}

stopped_epoch = EPOCHS   # updated if early stop fires

print(f'\n{"═"*60}')
print(f'  Train:{len(train_df)}  Val:{len(val_df)}  Test:{len(test_df)}')
print(f'  Patience : {PATIENCE} epochs | Min Δ : {MIN_DELTA}')
print(f'{"═"*60}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, optimiser, cls_fn, reg_fn,
                   is_train=True,  cls_w=1.0, reg_w=0.2)
    vl = run_epoch(model, val_loader,   optimiser, cls_fn, reg_fn,
                   is_train=False, cls_w=1.0, reg_w=0.2)
    te = run_epoch(model, test_loader,  optimiser, cls_fn, reg_fn,
                   is_train=False, cls_w=1.0, reg_w=0.2)
    scheduler.step()

    # ── Record all metrics for all three splits ───────────────────────────
    for split, res in [('train', tr), ('val', vl), ('test', te)]:
        history[f'{split}_loss'].append(res['loss'])
        history[f'{split}_cls_loss'].append(res['cls_loss'])
        history[f'{split}_reg_loss'].append(res['reg_loss'])
        history[f'{split}_acc'].append(res['accuracy'])
        history[f'{split}_rmse'].append(res['rmse'])
        history[f'{split}_mae'].append(res['mae'])
        history[f'{split}_r2'].append(res['r2'])

    msg = (f'  Ep {epoch:3d}/{EPOCHS} | '
           f'Tr loss={tr["loss"]:.3f} acc={tr["accuracy"]:.1f}% '
           f'mae={tr["mae"]:.3f} r2={tr["r2"]:.3f} | '
           f'Vl loss={vl["loss"]:.3f} acc={vl["accuracy"]:.1f}% '
           f'mae={vl["mae"]:.3f} r2={vl["r2"]:.3f} | '
           f'Te acc={te["accuracy"]:.1f}% mae={te["mae"]:.3f} r2={te["r2"]:.3f} | '
           f'ES {early_stop.counter}/{PATIENCE}')
    print(msg)
    log.info(msg)

    # ── Early stopping check ──────────────────────────────────────────────
    stop = early_stop.step(vl['accuracy'], model, epoch)
    if vl['accuracy'] >= early_stop.best_acc and early_stop.counter == 0:
        print(f'    ✓ Best weights updated  val_acc={early_stop.best_acc:.1f}%')
    if stop:
        stopped_epoch = epoch
        print(f'\n  ⏹  Early stopping fired at epoch {epoch}  '
              f'(best={early_stop.best_epoch}, patience={PATIENCE})')
        log.info(f'Early stopping at epoch {epoch}  best={early_stop.best_epoch}')
        break

print('\n✓ Training complete!')

# ── Restore best weights ──────────────────────────────────────────────────
model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch

# ── Final evaluation ──────────────────────────────────────────────────────
final_te = run_epoch(model, test_loader, optimiser, cls_fn, reg_fn,
                     is_train=False, cls_w=1.0, reg_w=0.2)

print(f'\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────────────')
print(f'  Accuracy : {final_te["accuracy"]:.2f}%')
print(f'  RMSE     : {final_te["rmse"]:.4f}')
print(f'  MAE      : {final_te["mae"]:.4f}')
print(f'  R²       : {final_te["r2"]:.4f}')
print(f'  Loss     : {final_te["loss"]:.4f}')

# ── Collect predictions for scatter + confusion matrix ────────────────────
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
plot_r2_mae(history, PLOTS_DIR)
plot_regression_scatter(all_true_q, all_pred_q, split_name='Test', save_dir=PLOTS_DIR)
plot_confusion_matrix(all_true_cls, all_pred_cls, PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

# ── Save history JSON ──────────────────────────────────────────────────────
json_path = os.path.join(LOGS_DIR, 'training_history.json')
with open(json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  ✓ History → {json_path}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

# ── Best val metrics at best_epoch ────────────────────────────────────────
bv_idx = best_epoch - 1   # 0-based index into history lists
best_val_acc  = history['val_acc'][bv_idx]
best_val_rmse = history['val_rmse'][bv_idx]
best_val_mae  = history['val_mae'][bv_idx]
best_val_r2   = history['val_r2'][bv_idx]

print('='*60)
print('  TRAINING SUMMARY — ST-GCN Single View')
print('='*60)
print(f'  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})')
print(f'  Best Val Acc     : {best_val_acc:.2f}%')
print(f'  Best Val RMSE    : {best_val_rmse:.4f}')
print(f'  Best Val MAE     : {best_val_mae:.4f}')
print(f'  Best Val R²      : {best_val_r2:.4f}')
print('─'*60)
print(f'  Test Accuracy    : {final_te["accuracy"]:.2f}%')
print(f'  Test RMSE        : {final_te["rmse"]:.4f}')
print(f'  Test MAE         : {final_te["mae"]:.4f}')
print(f'  Test R²          : {final_te["r2"]:.4f}')
print('='*60)

log.info(f'Best Epoch    : {best_epoch}  stopped_epoch={stopped_epoch}')
log.info(f'Test Accuracy : {final_te["accuracy"]:.2f}%')
log.info(f'Test RMSE     : {final_te["rmse"]:.4f}')
log.info(f'Test MAE      : {final_te["mae"]:.4f}')
log.info(f'Test R²       : {final_te["r2"]:.4f}')

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
pd.DataFrame([{
    'best_epoch'   : best_epoch,
    'stopped_epoch': stopped_epoch,
    'val_acc'      : best_val_acc,
    'val_rmse'     : best_val_rmse,
    'val_mae'      : best_val_mae,
    'val_r2'       : best_val_r2,
    'test_acc'     : final_te['accuracy'],
    'test_rmse'    : final_te['rmse'],
    'test_mae'     : final_te['mae'],
    'test_r2'      : final_te['r2'],
}]).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV saved → {summary_path}')
log.info('✓ All done!')

sys.stdout.restore()
print('✓ Log file closed and saved.')
