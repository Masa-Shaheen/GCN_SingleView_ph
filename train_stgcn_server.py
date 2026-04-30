# ══════════════════════════════════════════════════════════════════════════
# ST-GCN Single-View Regression — IMPROVED VERSION
# التحسينات:
#   1. Weighted Loss للدرجات النادرة
#   2. Stratification بـ 5 bins بدل 3
#   3. K-Fold Cross Validation (5-fold)
#   4. Data Augmentation أقوى (تدوير + occlusion)
#   5. Regression Head أعمق مع LayerNorm + GELU
#   6. centre_and_scale مصحّحة (joint 0 = Hip)
#   7. حفظ الـ Model بعد التدريب
#   8. num_workers=4 للسرعة
# ══════════════════════════════════════════════════════════════════════════


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
OUT_DIR       = "/mvdlph/masa/GCN_SingleView_Regression_Results_Improved"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 50
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 10

# ── [تحسين 3] K-Fold ──────────────────────────────────────────────────────
N_FOLDS       = 5          # عدد الـ folds
USE_KFOLD     = True       # غيّريه لـ False لو بدك تشغّلي بدون K-Fold

print('✓ Configuration loaded')
print(f'  DATASET_DIR : {DATASET_DIR}')
print(f'  NPZ_KEY     : {NPZ_KEY}')
print(f'  CAMERA_ID   : C{CAMERA_ID}')
print(f'  EXISTS      : {os.path.exists(DATASET_DIR)}')
print(f'  PATIENCE    : {PATIENCE} epochs')
print(f'  USE_KFOLD   : {USE_KFOLD}  (N_FOLDS={N_FOLDS})')


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
from sklearn.metrics  import r2_score
from sklearn.model_selection import KFold      # [تحسين 3]

PLOTS_DIR = os.path.join(OUT_DIR, "plots")
LOGS_DIR  = os.path.join(OUT_DIR, "logs")
for d in [PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✓ Libraries imported")
print("✓ Output folders ready")


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
log = logging.getLogger("GCN-Regression-Improved")
log.info("=" * 70)
log.info("ST-GCN Single-View Regression IMPROVED | BZU Physiotherapy Dataset")
log.info(f"Camera : C{CAMERA_ID}  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
log.info(f"K-Fold : {USE_KFOLD}  N_FOLDS={N_FOLDS}")
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

        # Swap axes: Y=height, Z=depth
        x = arr[:, :, 0].copy()
        y = arr[:, :, 1].copy()
        z = arr[:, :, 2].copy()
        arr[:, :, 0] = x
        arr[:, :, 1] = z
        arr[:, :, 2] = y

        return arr
    except Exception:
        return None

print('✓ parse_filename and load_skeleton defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build dataset index
# ══════════════════════════════════════════════════════════════════════════

def build_index(dataset_dir, camera_id, df_csv):
    df_csv         = df_csv.copy()
    df_csv.columns = df_csv.columns.str.strip()

    all_files = sorted(glob.glob(
        os.path.join(dataset_dir, '**/*.npz'), recursive=True))
    print(f'NPZ files found (all cameras) : {len(all_files)}')

    if not all_files:
        print('❌ No NPZ files found')
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
        print(f'❌ No records for camera C{camera_id}')
        return pd.DataFrame()

    df = pd.DataFrame(records)

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

print("Checking for corrupted files...")
bad_files = []
for fpath in df_index['filepath']:
    skel = load_skeleton(fpath)
    if skel is None:
        bad_files.append(fpath)

if bad_files:
    print(f"  Removing {len(bad_files)} corrupted files")
    df_index = df_index[~df_index['filepath'].isin(bad_files)].reset_index(drop=True)

print(f"  Clean samples remaining: {len(df_index)}")


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


def plot_skeleton_3d(skel, frame_idx=0, title='Skeleton', save_path=None):
    pts = skel[frame_idx]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    views = [
        (axes[0], x,  y,  'Front (X–Y)', 'X', 'Y', True),
        (axes[1], z,  y,  'Side  (Z–Y)', 'Z', 'Y', True),
        (axes[2], x, -z,  'Top   (X–Z)', 'X', '-Z', False),
    ]
    for ax, hx, hy, vt, xl, yl, inv in views:
        for (i, j) in SKELETON_EDGES:
            ax.plot([hx[i], hx[j]], [hy[i], hy[j]], color='dimgray', lw=2, zorder=1)
        for part, idxs in JOINT_COLORS.items():
            ax.scatter(hx[idxs], hy[idxs], c=PART_COLOR[part], s=80, zorder=3,
                       edgecolors='black', linewidths=0.5, label=part)
        ax.set_title(vt, fontweight='bold', fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        if inv:
            ax.invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Saved → {save_path}')
    plt.close()

print('✓ Skeleton visualisation helpers defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 9 — Visualise a sample
# ══════════════════════════════════════════════════════════════════════════

idx         = 10
sample_skel = load_skeleton(df_index.iloc[idx]['filepath'])
print(df_index.iloc[idx][['person', 'exercise', 'trial_id', 'segment']])
print(f'Skeleton shape : {sample_skel.shape}')

plot_skeleton_3d(
    sample_skel, frame_idx=0,
    title=f"Skeleton · {df_index.iloc[idx]['trial_key']}",
    save_path=os.path.join(PLOTS_DIR, 'sample_skeleton_3views.png'),
)


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset  [تحسين 4: Augmentation أقوى]
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
        skel    = load_skeleton(row['filepath'])
        if skel is None:
            skel = np.zeros((self.target_frames, 17, 3), dtype=np.float32)
        skel    = self._normalise_length(skel)
        if self.augment:
            skel = self._augment(skel)
        skel    = torch.tensor(skel,           dtype=torch.float32)
        quality = torch.tensor(row['quality'], dtype=torch.float32)
        return skel, quality

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
        T = skel.shape[0]

        # 1. تغيير السرعة
        speed = np.random.uniform(0.8, 1.2)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = self._normalise_length(skel[idxs])

        # 2. ضوضاء صغيرة
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.005

        # 3. قلب أفقي
        if np.random.rand() < 0.5:
            skel[:, :, 0] *= -1.0

        # [تحسين 4a] تدوير حول محور Y (±15 درجة)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([
                [ cos_a, 0, sin_a],
                [     0, 1,     0],
                [-sin_a, 0, cos_a],
            ], dtype=np.float32)
            # skel shape: (T, J, 3)  → reshape لـ matrix multiply
            orig_shape = skel.shape
            skel = skel.reshape(-1, 3) @ rot.T
            skel = skel.reshape(orig_shape)

        # [تحسين 4b] محاكاة Occlusion — استبدل إطارات عشوائية بالإطار السابق
        if np.random.rand() < 0.3:
            n_drop  = max(1, T // 10)
            drop_idx = np.random.choice(range(1, T), size=n_drop, replace=False)
            for di in drop_idx:
                skel[di] = skel[di - 1]

        return skel


print('✓ BZUDataset defined  [مع Augmentation محسّن]')


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN Model  [تحسين 5: Regression Head أعمق]
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
    def __init__(self, dropout=0.3):
        super().__init__()
        A = build_adj(NUM_JOINTS, SKELETON_EDGES)

        self.data_bn = nn.BatchNorm1d(3)

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

        self.drop = nn.Dropout(dropout)

        # [تحسين 5] Regression Head أعمق مع LayerNorm + GELU
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
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
        B, T, J, C = x.shape

        xbn = x.permute(0, 3, 1, 2).reshape(B, C, T * J)
        xbn = self.data_bn(xbn)
        x   = xbn.reshape(B, C, T, J).permute(0, 2, 3, 1)
        x   = x.permute(0, 3, 1, 2)

        for blk in self.blocks:
            x = blk(x)

        x   = x.mean(dim=[2, 3])
        x   = self.drop(x)
        qua = 3.0 + 2.0 * torch.tanh(self.reg_head(x).squeeze(1))
        return qua

print('✓ ST-GCN Regression model defined  [Head محسّن]')


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device, normalisation & run_epoch
# [تحسين 6]: centre_and_scale تستخدم joint 0 (Hip) مباشرة
# [تحسين 1]: Weighted Loss في run_epoch
# ══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')


def centre_and_scale(x):
    """
    [تحسين 6] Root-relative: يطرح joint 0 (Hip) مباشرة بدل متوسط joint 1+4
    x: (B, T, J, 3)
    """
    root = x[:, :, 0:1, :]            # joint 0 = Hip في Human3.6M
    x    = x - root

    shoulder = (x[:, :, 11:12, :] + x[:, :, 14:15, :]) / 2.0
    torso_h  = shoulder[:, :, :, 1:2].abs()
    torso_h  = torso_h.mean(dim=1, keepdim=True).clamp(min=1e-6)
    return x / torso_h


def compute_sample_weights(qualities_tensor):
    """
    [تحسين 1] يحسب وزن عكسي لكل عينة بحسب ندرة درجتها.
    الدرجات النادرة تأخذ وزناً أعلى.
    """
    q = qualities_tensor.cpu().numpy()
    # قسّمي النطاق 1-5 إلى 8 صناديق
    bins    = np.linspace(1.0, 5.0, 9)
    bin_idx = np.digitize(q, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 7)

    counts  = np.bincount(bin_idx, minlength=8).astype(float)
    counts  = np.maximum(counts, 1.0)
    weights = 1.0 / counts[bin_idx]
    weights = weights / weights.sum() * len(weights)   # normalize

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimiser, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    base_loss_fn = nn.SmoothL1Loss(reduction='none')

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, qualities in loader:
            skels     = centre_and_scale(skels.to(DEVICE))
            qualities = qualities.to(DEVICE)

            preds = model(skels)

            # [تحسين 1] Weighted Loss
            raw_loss = base_loss_fn(preds, qualities)         # (B,)
            weights  = compute_sample_weights(qualities).to(DEVICE)
            loss     = (raw_loss * weights).mean()

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

    return {
        'loss': total_loss / n,
        'rmse': float(np.sqrt(np.mean((qt - qp) ** 2))),
        'mae' : float(np.mean(np.abs(qt - qp))),
        'r2'  : float(r2_score(qt, qp)) if len(qt) > 1 else 0.0,
    }

print('✓ centre_and_scale [مصحّحة] و run_epoch [Weighted Loss] معرّفة')


# ══════════════════════════════════════════════════════════════════════════
# Cell 13 — Split functions
# [تحسين 2]: Stratification بـ 5 bins
# [تحسين 3]: K-Fold على الأشخاص
# ══════════════════════════════════════════════════════════════════════════

def get_trial_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=42):
    """
    [تحسين 2] تقسيم بحسب الأشخاص مع Stratification بـ 5 bins بدل 3.
    """
    rng = np.random.default_rng(random_state)

    person_quality = df.groupby('person')['quality'].mean()
    persons        = person_quality.index.values
    qualities      = person_quality.values

    # [تحسين 2] 5 bins بدل 3 لتمثيل أفضل للحالات النادرة
    bins   = np.percentile(qualities, [20, 40, 60, 80])
    strata = np.digitize(qualities, bins)

    train_persons, val_persons, test_persons = [], [], []

    for stratum in np.unique(strata):
        sp = persons[strata == stratum]
        rng.shuffle(sp)
        n       = len(sp)
        n_test  = max(1, int((1 - train_ratio - val_ratio) * n))
        n_val   = max(1, int(val_ratio * n))
        n_train = n - n_val - n_test

        train_persons.extend(sp[:n_train])
        val_persons.extend(sp[n_train:n_train + n_val])
        test_persons.extend(sp[n_train + n_val:])

    train_df = df[df['person'].isin(train_persons)].reset_index(drop=True)
    val_df   = df[df['person'].isin(val_persons)].reset_index(drop=True)
    test_df  = df[df['person'].isin(test_persons)].reset_index(drop=True)

    print(f'\nPersons → train={len(train_persons)}, val={len(val_persons)}, test={len(test_persons)}')
    print(f'Samples → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
    for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        q = d['quality']
        print(f'  {name}: mean={q.mean():.3f} std={q.std():.3f} '
              f'min={q.min():.2f} max={q.max():.2f}')

    return train_df, val_df, test_df


def get_kfold_splits(df, n_folds=N_FOLDS, random_state=42):
    """
    [تحسين 3] K-Fold على مستوى الأشخاص — يضمن عدم تسرب البيانات.
    كل fold يعطي (train_df, val_df, test_df).
    """
    persons = df['person'].unique()
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds   = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(persons)):
        test_persons    = persons[test_idx]
        trainval_persons = persons[trainval_idx]

        # من trainval خذي 15% كـ val
        n_val      = max(1, int(0.15 * len(trainval_persons)))
        rng        = np.random.default_rng(random_state + fold_idx)
        rng.shuffle(trainval_persons)
        val_persons   = trainval_persons[:n_val]
        train_persons = trainval_persons[n_val:]

        folds.append({
            'fold'     : fold_idx + 1,
            'train_df' : df[df['person'].isin(train_persons)].reset_index(drop=True),
            'val_df'   : df[df['person'].isin(val_persons)].reset_index(drop=True),
            'test_df'  : df[df['person'].isin(test_persons)].reset_index(drop=True),
        })

        print(f'  Fold {fold_idx+1}: '
              f'train={len(folds[-1]["train_df"])} '
              f'val={len(folds[-1]["val_df"])} '
              f'test={len(folds[-1]["test_df"])}')

    return folds

print('✓ Split functions defined  [5-bin Stratification + K-Fold]')


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved → {path}')


def plot_loss_curves(history, save_dir, prefix=''):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_loss'],   label='Validation', color='darkorange')
    ax.plot(epochs, history['test_loss'],  label='Test',       color='green', linestyle='--')
    ax.set_title(f'{prefix}Regression Loss (Weighted SmoothL1)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, f'{prefix}loss_curve.png'))


def plot_rmse_mae(history, save_dir, prefix=''):
    epochs = range(1, len(history['val_rmse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{prefix}RMSE & MAE', fontsize=14, fontweight='bold')
    for ax, metric in zip(axes, ['rmse', 'mae']):
        ax.plot(epochs, history[f'train_{metric}'], label='Train',      color='steelblue')
        ax.plot(epochs, history[f'val_{metric}'],   label='Validation', color='darkorange')
        ax.plot(epochs, history[f'test_{metric}'],  label='Test',       color='green', linestyle='--')
        ax.set_title(metric.upper())
        ax.set_xlabel('Epoch'); ax.set_ylabel(metric.upper())
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, f'{prefix}rmse_mae.png'))


def plot_r2(history, save_dir, prefix=''):
    epochs = range(1, len(history['val_r2']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['train_r2'], label='Train',      color='steelblue')
    ax.plot(epochs, history['val_r2'],   label='Validation', color='darkorange')
    ax.plot(epochs, history['test_r2'],  label='Test',       color='green', linestyle='--')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Perfect R²=1')
    ax.axhline(0.0, color='red',  linestyle=':', linewidth=1, label='Baseline R²=0')
    ax.set_title(f'{prefix}R² Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('R²')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, f'{prefix}r2_curve.png'))


def plot_regression_scatter(q_true, q_pred, split_name='Test', save_dir=None, prefix=''):
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
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True Quality Score', fontsize=12)
    ax.set_ylabel('Predicted Quality Score', fontsize=12)
    ax.set_title(f'{prefix}{split_name} — True vs Predicted', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    textstr = f'R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    if save_dir:
        save_and_show(fig, os.path.join(save_dir,
                      f'{prefix}scatter_{split_name.lower()}.png'))
    else:
        plt.close(fig)


def plot_early_stop(history, stopped_epoch, best_epoch, save_dir, prefix=''):
    epochs = range(1, len(history['val_mae']) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['train_mae'], label='Train MAE', color='steelblue')
    ax.plot(epochs, history['val_mae'],   label='Val MAE',   color='darkorange')
    ax.plot(epochs, history['test_mae'],  label='Test MAE',  color='green', linestyle='--')
    ax.axvline(best_epoch,    color='purple', linestyle=':',  linewidth=2,
               label=f'Best ({best_epoch})')
    ax.axvline(stopped_epoch, color='red',    linestyle='--', linewidth=2,
               label=f'Stop ({stopped_epoch})')
    ax.set_title(f'{prefix}MAE + Early Stopping', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, f'{prefix}early_stopping.png'))


def plot_kfold_summary(fold_results, save_dir):
    """رسم نتائج كل الـ folds مع الـ mean ± std."""
    folds = [r['fold'] for r in fold_results]
    maes  = [r['test_mae']  for r in fold_results]
    rmses = [r['test_rmse'] for r in fold_results]
    r2s   = [r['test_r2']   for r in fold_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('K-Fold Cross Validation Results', fontsize=14, fontweight='bold')

    for ax, vals, name, color in zip(
        axes,
        [maes, rmses, r2s],
        ['Test MAE', 'Test RMSE', 'Test R²'],
        ['steelblue', 'darkorange', 'limegreen']
    ):
        ax.bar(folds, vals, color=color, alpha=0.7, edgecolor='black')
        mean_val = np.mean(vals)
        std_val  = np.std(vals)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean={mean_val:.3f}±{std_val:.3f}')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_xticks(folds)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, 'kfold_summary.png'))

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
# Cell 16 — دالة تدريب fold واحد
# ══════════════════════════════════════════════════════════════════════════

def save_model(model, best_epoch, best_mae, path):
    """[تحسين 7] حفظ الـ model بشكل كامل."""
    torch.save({
        'epoch'      : best_epoch,
        'model_state': model.state_dict(),
        'val_mae'    : best_mae,
        'config'     : {
            'camera'    : CAMERA_ID,
            'num_joints': NUM_JOINTS,
            'target_frames': TARGET_FRAMES,
        }
    }, path)
    print(f'  ✓ Model saved → {path}')


def train_one_fold(train_df, val_df, test_df,
                   fold_label='', plots_dir=PLOTS_DIR):
    """يدرّب النموذج على fold واحد ويرجع النتائج."""

    # [تحسين 8] num_workers=4 للسرعة
    num_workers = 4 if DEVICE == 'cuda' else 0

    train_loader = DataLoader(
        BZUDataset(train_df, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(DEVICE == 'cuda'))
    val_loader = DataLoader(
        BZUDataset(val_df, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=(DEVICE == 'cuda'))
    test_loader = DataLoader(
        BZUDataset(test_df, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=(DEVICE == 'cuda'))

    model     = STGCN_Regression(dropout=0.3).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

    prefix = f'{fold_label}_' if fold_label else ''

    log.info(f'{"="*60}')
    log.info(f'TRAINING {fold_label or "single model"}')
    log.info(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')

    print(f'\n{"═"*60}')
    print(f'  {fold_label or "Training"}  |  '
          f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')
    print(f'{"═"*60}')

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimiser, is_train=True)
        vl = run_epoch(model, val_loader,   optimiser, is_train=False)
        te = run_epoch(model, test_loader,  optimiser, is_train=False)
        scheduler.step()

        for split, res in [('train', tr), ('val', vl), ('test', te)]:
            for m in METRICS:
                history[f'{split}_{m}'].append(res[m])

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
            print(f'\n  ⏹  Early stop @ epoch {epoch}  (best={early_stop.best_epoch})')
            log.info(f'Early stop @ {epoch}  best={early_stop.best_epoch}')
            break

    # Restore best weights
    model.load_state_dict(early_stop.best_wts)
    best_epoch = early_stop.best_epoch

    # Final evaluation
    final_te = run_epoch(model, test_loader, optimiser, is_train=False)

    print(f'\n  ── Final Test ({fold_label}) ──────────────')
    print(f'  MAE={final_te["mae"]:.4f}  RMSE={final_te["rmse"]:.4f}  R²={final_te["r2"]:.4f}')

    # Collect predictions for scatter
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for skels, qualities in test_loader:
            skels = centre_and_scale(skels.to(DEVICE))
            preds = model(skels)
            all_true.extend(qualities.numpy())
            all_pred.extend(preds.cpu().numpy())

    # Save plots
    plot_loss_curves(history, plots_dir, prefix=prefix)
    plot_rmse_mae(history, plots_dir, prefix=prefix)
    plot_r2(history, plots_dir, prefix=prefix)
    plot_regression_scatter(all_true, all_pred, 'Test', plots_dir, prefix=prefix)
    plot_early_stop(history, stopped_epoch, best_epoch, plots_dir, prefix=prefix)

    # [تحسين 7] حفظ الـ model
    model_path = os.path.join(OUT_DIR, f'best_model_{prefix}C{CAMERA_ID}.pth')
    save_model(model, best_epoch, early_stop.best_mae, model_path)

    # Save history JSON
    json_path = os.path.join(LOGS_DIR, f'history_{prefix}.json')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'fold'        : fold_label,
        'best_epoch'  : best_epoch,
        'stopped_epoch': stopped_epoch,
        'val_mae'     : history['val_mae'][best_epoch - 1],
        'val_rmse'    : history['val_rmse'][best_epoch - 1],
        'val_r2'      : history['val_r2'][best_epoch - 1],
        'test_mae'    : final_te['mae'],
        'test_rmse'   : final_te['rmse'],
        'test_r2'     : final_te['r2'],
        'model_path'  : model_path,
    }

print('✓ train_one_fold() defined')


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — تشغيل التدريب (K-Fold أو Single Split)
# ══════════════════════════════════════════════════════════════════════════

all_results = []

if USE_KFOLD:
    # ── [تحسين 3] K-Fold Cross Validation ───────────────────────────────
    print(f'\n{"═"*60}')
    print(f'  K-FOLD CROSS VALIDATION  (K={N_FOLDS})')
    print(f'{"═"*60}')

    folds = get_kfold_splits(df_index, n_folds=N_FOLDS)

    for fold_info in folds:
        fold_num = fold_info['fold']
        print(f'\n{"─"*60}')
        print(f'  FOLD {fold_num}/{N_FOLDS}')
        print(f'{"─"*60}')

        result = train_one_fold(
            train_df   = fold_info['train_df'],
            val_df     = fold_info['val_df'],
            test_df    = fold_info['test_df'],
            fold_label = f'fold{fold_num}',
            plots_dir  = PLOTS_DIR,
        )
        all_results.append(result)

        log.info(f'Fold {fold_num} done: '
                 f'test_mae={result["test_mae"]:.4f} '
                 f'test_r2={result["test_r2"]:.4f}')

    # رسم ملخص K-Fold
    plot_kfold_summary(all_results, PLOTS_DIR)

    # طباعة الملخص النهائي
    maes  = [r['test_mae']  for r in all_results]
    rmses = [r['test_rmse'] for r in all_results]
    r2s   = [r['test_r2']   for r in all_results]

    print(f'\n{"═"*60}')
    print(f'  K-FOLD FINAL SUMMARY  (K={N_FOLDS})')
    print(f'{"═"*60}')
    for r in all_results:
        print(f'  Fold {r["fold"]:>8} | '
              f'MAE={r["test_mae"]:.4f}  '
              f'RMSE={r["test_rmse"]:.4f}  '
              f'R²={r["test_r2"]:.4f}')
    print(f'{"─"*60}')
    print(f'  Mean ± Std | '
          f'MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  '
          f'RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}  '
          f'R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}')
    print(f'{"═"*60}')

    log.info(f'K-Fold Summary: MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  '
             f'R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}')

else:
    # ── Single Split ─────────────────────────────────────────────────────
    train_df, val_df, test_df = get_trial_split(df_index)
    result = train_one_fold(
        train_df   = train_df,
        val_df     = val_df,
        test_df    = test_df,
        fold_label = '',
        plots_dir  = PLOTS_DIR,
    )
    all_results.append(result)

    print(f'\n{"═"*60}')
    print(f'  FINAL RESULTS')
    print(f'{"═"*60}')
    print(f'  Best Epoch  : {result["best_epoch"]} (stopped {result["stopped_epoch"]})')
    print(f'  Val  MAE={result["val_mae"]:.4f}  RMSE={result["val_rmse"]:.4f}  R²={result["val_r2"]:.4f}')
    print(f'  Test MAE={result["test_mae"]:.4f}  RMSE={result["test_rmse"]:.4f}  R²={result["test_r2"]:.4f}')
    print(f'{"═"*60}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 18 — حفظ ملخص CSV النهائي
# ══════════════════════════════════════════════════════════════════════════

summary_path = os.path.join(OUT_DIR, 'training_summary.csv')
pd.DataFrame(all_results).to_csv(summary_path, index=False)
print(f'\n✓ Summary CSV saved → {summary_path}')
log.info('✓ All done!')

sys.stdout.restore()
print('✓ Log file closed and saved.')
