# ══════════════════════════════════════════════════════════════════════════
# EARLY FUSION MULTI-VIEW — ST-GCN Regression
# BZU Physiotherapy Dataset
#
# Key difference from Single-View:
#   Single-View : S_k,i ∈ R^(T×17×3)  →  one camera, 3 XYZ channels
#   Early Fusion: S_early ∈ R^(T×17×9) →  3 cameras concatenated on the
#                 coordinate axis BEFORE any model sees the data.
#
#   The model receives all three viewpoints simultaneously, allowing it to
#   jointly reason over multi-view spatial information from the first layer.
#
# Changes vs. single-view script:
#   Cell 1  : CAMERA_ID removed / NUM_VIEWS = 3 / IN_FEATURES = 18 (pos+vel per view)
#   Cell 7  : build_index_from_split_multiview  — loads all 3 cameras per trial
#   Cell 10 : BZUDataset_EarlyFusion           — stacks 3 skeleton tensors on axis-2
#   Cell 11 : STGCN_Regression_EarlyFusion     — in_features=18 (3 views × 6 channels)
#   Cell 12 : centre_and_scale_multiview        — normalises each view independently
#   All else: identical to the single-view script
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════
import os

DATASET_DIR   = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_DIR     = os.path.join(DATASET_DIR, "by_person")
CSV_PATH      = "/mvdlph/label_events_20260129_155122_stats_short.csv"

NUM_VIEWS     = 3          # cameras 0, 1, 2
NUM_JOINTS    = 17
# Per view: 3 position + 3 velocity = 6 channels → × 3 views = 18
IN_FEATURES   = 6 * NUM_VIEWS   # 18

TARGET_FRAMES = 100
EPOCHS        = 300
LR            = 1e-4
BATCH_SIZE    = 48
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "/mvdlph/masa/GCN_EarlyFusion_Results"

# ── Early Stopping ────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
WARMUP_EPOCHS = 20

print("✓ Configuration loaded  (Early Fusion Multi-View)")
print(f"  DATASET_DIR : {DATASET_DIR}")
print(f"  SPLIT_DIR   : {SPLIT_DIR}")
print(f"  NUM_VIEWS   : {NUM_VIEWS}  (cameras 0–{NUM_VIEWS-1})")
print(f"  IN_FEATURES : {IN_FEATURES}  ({NUM_VIEWS} views × 6 channels)")
print(f"  EXISTS      : {os.path.exists(DATASET_DIR)}")
print(f"  SPLIT EXISTS: {os.path.exists(SPLIT_DIR)}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 2 — Imports & output folders
# ══════════════════════════════════════════════════════════════════════════
import re, glob, json, logging, datetime, copy, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
# Cell 4 — Load CSV labels   (unchanged)
# ══════════════════════════════════════════════════════════════════════════
df_csv = None
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "rb") as f:
        raw = f.read()
    for enc in ["utf-8", "utf-8-sig", "utf-16", "latin-1", "cp1252"]:
        try:
            text = raw.decode(enc)
            tmp  = pd.read_csv(io.StringIO(text))
            tmp.columns = tmp.columns.str.strip()
            if "exercise" in tmp.columns:
                df_csv = tmp
                print(f"✓ CSV loaded ({enc})")
                break
        except Exception:
            pass
else:
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

if df_csv is None:
    raise FileNotFoundError("CSV could not be parsed.")

print(f"  Columns : {df_csv.columns.tolist()}")
print(f"  Shape   : {df_csv.shape}")


# ══════════════════════════════════════════════════════════════════════════
# Cell 5 — Logging setup   (unchanged)
# ══════════════════════════════════════════════════════════════════════════
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(LOGS_DIR, f"training_{timestamp}.log")


class Tee:
    def __init__(self, console, filepath):
        self.console  = console
        self._logfile = open(filepath, "a", encoding="utf-8", buffering=1)

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
log = logging.getLogger("STGCN-EarlyFusion")
log.info("=" * 70)
log.info("ST-GCN Early-Fusion Multi-View Regression | BZU Physiotherapy")
log.info(f"IN_FEATURES={IN_FEATURES}  NUM_VIEWS={NUM_VIEWS}  EPOCHS={EPOCHS}  PATIENCE={PATIENCE}")
log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Cell 6 — Filename parser & skeleton loader   (unchanged)
# ══════════════════════════════════════════════════════════════════════════
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

NPZ_KEY = "keypoints_3d"


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


print("✓ parse_filename and load_skeleton defined")


# ══════════════════════════════════════════════════════════════════════════
# Cell 7 — Build MULTI-VIEW index from pre-split directories
#
# NEW logic vs single-view:
#   For each (exercise, person, trial, segment) tuple we need ONE file
#   per camera (C0, C1, C2).  We group files by their
#   (exercise, person, trial_id, segment) key and keep only groups that
#   have ALL 3 cameras available.  Missing-camera groups are discarded
#   and reported.
# ══════════════════════════════════════════════════════════════════════════

def build_multiview_index(split_name, df_csv, num_views=3):
    """
    Scan a pre-split folder and build a DataFrame where every row
    represents ONE complete multi-view sample (all cameras present).

    Columns:
        exercise, person, trial_num, trial_id, segment, trial_key, quality,
        filepath_C0, filepath_C1, filepath_C2
    """
    split_path = os.path.join(SPLIT_DIR, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    all_files = sorted(glob.glob(
        os.path.join(split_path, "**/*.npz"), recursive=True))
    print(f"\n[{split_name.upper()}] NPZ files found : {len(all_files)}")

    df_csv         = df_csv.copy()
    df_csv.columns = df_csv.columns.str.strip()

    # ── Parse every file and index by (exercise, person, trial_id, segment)
    from collections import defaultdict
    groups = defaultdict(dict)   # key → {camera: filepath}

    skipped = 0
    for fpath in all_files:
        meta = parse_filename(fpath)
        if meta is None:
            skipped += 1
            continue
        key = (meta["exercise"], meta["person"], meta["trial_id"], meta["segment"])
        groups[key][meta["camera"]] = fpath

    print(f"  Unique (E, P, T, seg) groups : {len(groups)}  |  skipped files : {skipped}")

    # ── Keep only groups with all cameras present
    required_cameras = list(range(num_views))   # [0, 1, 2]
    complete, incomplete = [], []
    for key, cam_map in groups.items():
        if all(c in cam_map for c in required_cameras):
            complete.append((key, cam_map))
        else:
            incomplete.append(key)

    if incomplete:
        print(f"  ⚠️  {len(incomplete)} groups dropped (missing ≥1 camera): "
              f"{incomplete[:5]}{'...' if len(incomplete)>5 else ''}")
    print(f"  Complete multi-view groups   : {len(complete)}")

    if not complete:
        print(f"  ⚠️  No complete groups for split='{split_name}'")
        return pd.DataFrame()

    # ── Build records
    records = []
    for (ex, person, trial_id, seg), cam_map in complete:
        trial_num = int(trial_id[1:])

        # Quality label from CSV
        row = df_csv[
            (df_csv["exercise"] == f"E{ex}") &
            (df_csv["person"]   == person)    &
            (df_csv["trial"]    == trial_id)
        ]
        quality = float(row.iloc[0]["mean"]) if len(row) > 0 else np.nan

        rec = {
            "exercise"  : ex,
            "person"    : person,
            "trial_num" : trial_num,
            "trial_id"  : trial_id,
            "segment"   : seg,
            "trial_key" : f"E{ex}_{person}_{trial_id}",
            "quality"   : quality,
            "split"     : split_name,
        }
        for c in required_cameras:
            rec[f"filepath_C{c}"] = cam_map[c]

        records.append(rec)

    df = pd.DataFrame(records)

    # ── Fill NaN quality
    correct_mean   = df.loc[df["trial_num"] <= 2, "quality"].mean()
    erroneous_mean = df.loc[df["trial_num"] >= 3, "quality"].mean()
    if np.isnan(correct_mean):   correct_mean   = 4.0
    if np.isnan(erroneous_mean): erroneous_mean = 2.5
    df.loc[df["quality"].isna() & (df["trial_num"] <= 2), "quality"] = correct_mean
    df.loc[df["quality"].isna() & (df["trial_num"] >= 3), "quality"] = erroneous_mean

    print(f"  Samples          : {len(df)}")
    print(f"  Unique trials    : {df['trial_key'].nunique()}")
    print(f"  Quality mean±std : {df['quality'].mean():.3f} ± {df['quality'].std():.3f}")

    return df


def remove_corrupted_multiview(df, label="", num_views=3):
    """Drop rows where ANY camera file cannot be loaded."""
    bad_rows = []
    for idx, row in df.iterrows():
        for c in range(num_views):
            if load_skeleton(row[f"filepath_C{c}"]) is None:
                bad_rows.append(idx)
                break
    if bad_rows:
        print(f"  [{label}] Removing {len(bad_rows)} corrupted row(s)")
        df = df.drop(index=bad_rows).reset_index(drop=True)
    print(f"  [{label}] Clean multi-view samples : {len(df)}")
    return df


# ── Build splits
train_df = build_multiview_index("train", df_csv, NUM_VIEWS)
val_df   = build_multiview_index("valid", df_csv, NUM_VIEWS)
test_df  = build_multiview_index("test",  df_csv, NUM_VIEWS)

print("\nChecking for corrupted files...")
train_df = remove_corrupted_multiview(train_df, "TRAIN", NUM_VIEWS)
val_df   = remove_corrupted_multiview(val_df,   "VALID", NUM_VIEWS)
test_df  = remove_corrupted_multiview(test_df,  "TEST",  NUM_VIEWS)

# ── Sanity: no trial leakage across splits
tr_keys = set(train_df["trial_key"])
vl_keys = set(val_df["trial_key"])
te_keys = set(test_df["trial_key"])
assert tr_keys.isdisjoint(vl_keys), "LEAK: train ∩ val"
assert tr_keys.isdisjoint(te_keys), "LEAK: train ∩ test"
assert vl_keys.isdisjoint(te_keys), "LEAK: val ∩ test"
print("\n✓ No data-leakage detected across splits")

print(f'\n{"═"*68}')
print(f'  {"Split":<8} {"Samples":>8} {"Correct":>9} {"Erroneous":>11} '
      f'{"Q mean":>8} {"Q std":>7}')
print(f'  {"─"*66}')
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    cor = (d["trial_num"] <= 2).sum()
    err = (d["trial_num"] >= 3).sum()
    q   = d["quality"]
    print(f'  {name:<8} {len(d):>8} {cor:>9} {err:>11} '
          f'{q.mean():>8.3f} {q.std():>7.3f}')
print(f'{"═"*68}')


# ══════════════════════════════════════════════════════════════════════════
# Cell 10 — BZUDataset_EarlyFusion
#
# Returns (skeleton_fused, quality, exercise_id)
#
# skeleton_fused shape: (T, J, 6*NUM_VIEWS) = (100, 17, 18)
#
# How the fusion works:
#   For each camera k:
#       skel_k        →  (T, 17, 3)   position
#       velocity_k    →  (T, 17, 3)   finite difference
#       skel_vel_k    →  (T, 17, 6)   concat along last axis
#   Stack all views:
#       skel_fused    →  (T, 17, 18)  concat along last axis
#                                     i.e. [pos_C0, vel_C0, pos_C1, vel_C1, pos_C2, vel_C2]
# ══════════════════════════════════════════════════════════════════════════

class BZUDataset_EarlyFusion(Dataset):

    def __init__(self, df, target_frames=TARGET_FRAMES,
                 num_views=NUM_VIEWS, augment=False):
        self.df            = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.num_views     = num_views
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        view_tensors = []
        for c in range(self.num_views):
            skel = load_skeleton(row[f"filepath_C{c}"])
            if skel is None:
                skel = np.zeros((self.target_frames, NUM_JOINTS, 3), dtype=np.float32)

            skel = self._normalise_length(skel)

            if self.augment:
                skel = self._augment(skel)

            # Velocity
            velocity      = np.zeros_like(skel)
            velocity[1:]  = skel[1:] - skel[:-1]

            # (T, J, 6)
            skel_vel = np.concatenate([skel, velocity], axis=-1)
            view_tensors.append(skel_vel)

        # Early fusion: concatenate views on the channel axis → (T, J, 18)
        fused = np.concatenate(view_tensors, axis=-1)   # (T, 17, 18)

        skel_tensor = torch.tensor(fused,          dtype=torch.float32)
        quality     = torch.tensor(row["quality"], dtype=torch.float32)
        exercise_id = torch.tensor(row["exercise"],dtype=torch.long)

        return skel_tensor, quality, exercise_id

    # ── helpers (same as single-view) ────────────────────────────────────
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
        T     = skel.shape[0]
        speed = np.random.uniform(0.8, 1.2)
        idxs  = np.linspace(0, T - 1, max(10, int(T * speed))).astype(int)
        skel  = self._normalise_length(skel[idxs])
        skel += np.random.randn(*skel.shape).astype(np.float32) * 0.005
        return skel


print("✓ BZUDataset_EarlyFusion defined")


# ══════════════════════════════════════════════════════════════════════════
# Cell 11 — ST-GCN model adapted for Early Fusion
#
# The ONLY structural change: in_features = 18  (was 6).
# Everything else — adjacency, block depth, regression head — is identical.
# The model receives (T, J, 18) and treats the 18 channels as a richer
# per-joint feature vector that encodes all three views simultaneously.
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
    A = np.zeros((3, num_joints, num_joints), dtype=np.float32)
    for i in range(num_joints):
        A[0, i, i] = 1.0
    for (i, j) in edges:
        if dist[j] < dist[i]:   A[1, i, j] = 1.0
        elif dist[j] > dist[i]: A[2, i, j] = 1.0
        else:                    A[1, i, j] = 1.0
        if dist[i] < dist[j]:   A[1, j, i] = 1.0
        elif dist[i] > dist[j]: A[2, j, i] = 1.0
        else:                    A[1, j, i] = 1.0
    for k in range(3):
        row_sum   = A[k].sum(axis=1)
        d_inv_sq  = np.where(row_sum > 0, np.power(row_sum, -0.5), 0.0)
        D_inv_sq  = np.diag(d_inv_sq)
        A[k]      = D_inv_sq @ A[k] @ D_inv_sq
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
        x = self.conv(x)
        x = x.view(B, self.K, -1, T, J)
        A_eff = A + self.M
        out = torch.einsum("bkctj,kjv->bctv", x, A_eff)
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


class STGCN_EarlyFusion(nn.Module):
    """
    ST-GCN adapted for Early Fusion.

    Input: (B, T, J, 18)  — 3 cameras × 6 channels (pos+vel) per joint
    The model sees all viewpoints from layer 0 via the enlarged channel dim.

    Architecture is identical to the single-view model; only the first
    layer input size changes (18 instead of 6).
    """
    def __init__(self, in_features=IN_FEATURES, K=3, dropout=0.5):
        super().__init__()

        A = build_stgcn_adjacency(NUM_JOINTS, SKELETON_EDGES, center_joint=0)
        self.register_buffer("A", A)

        # Data BN operates on (C * J) flattened features
        self.data_bn = nn.BatchNorm1d(in_features * NUM_JOINTS)

        # 9 ST-GCN blocks — only block-0 in_channels changes (18 vs 6)
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, exercise_id):
        """
        x           : (B, T, J, C)   C = 18 for early fusion
        exercise_id : (B,)
        returns     : (B,)  quality scores in [1, 5]
        """
        B, T, J, C = x.shape

        # Data BN
        x = x.permute(0, 3, 2, 1).reshape(B, C * J, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, J, T).permute(0, 1, 3, 2)   # (B, C, T, J)

        for block in self.blocks:
            x = block(x, self.A)

        x   = self.gap(x).squeeze(-1).squeeze(-1)         # (B, 256)
        ex  = self.ex_embed(exercise_id)                   # (B, 32)
        h   = torch.cat([x, ex], dim=1)                    # (B, 288)
        out = 3.0 + 2.0 * torch.tanh(self.reg_head(h).squeeze(1))
        return out


# ── Sanity check ──────────────────────────────────────────────────────────
_dummy_x  = torch.zeros(2, TARGET_FRAMES, NUM_JOINTS, IN_FEATURES)
_dummy_ex = torch.zeros(2, dtype=torch.long)
_model    = STGCN_EarlyFusion()
_out      = _model(_dummy_x, _dummy_ex)
assert _out.shape == (2,), f"Expected (2,), got {_out.shape}"
print(f"\n✓ STGCN_EarlyFusion sanity check passed — output: {_out.shape}")
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f"✓ Trainable parameters: {total_params:,}")
del _dummy_x, _dummy_ex, _model, _out


# ══════════════════════════════════════════════════════════════════════════
# Cell 12 — Device & normalisation
#
# centre_and_scale_multiview:
#   Input  : (B, T, J, 18) — 3 views × 6 channels
#   Action : normalise each view's position block independently
#            using that view's own hip centre and torso height.
#   This ensures each camera's coordinate system is centred and scaled
#   before they are used jointly by the model.
# ══════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def centre_and_scale_multiview(x, num_views=NUM_VIEWS):
    """
    x : (B, T, J, 6*num_views)
    Normalises each view's position+velocity block independently.
    Returns same shape.
    """
    channels_per_view = 6   # 3 pos + 3 vel
    out_parts = []

    for v in range(num_views):
        start = v * channels_per_view
        end   = start + channels_per_view

        view  = x[:, :, :, start:end]          # (B, T, J, 6)
        pos   = view[:, :, :, :3]
        vel   = view[:, :, :, 3:]

        # Hip centre (average of joints 1 and 4)
        hip      = (pos[:, :, 1:2, :] + pos[:, :, 4:5, :]) / 2.0
        pos      = pos - hip

        # Torso height (distance from hip to mid-shoulder)
        shoulder = (pos[:, :, 11:12, :] + pos[:, :, 14:15, :]) / 2.0
        torso_h  = shoulder[:, :, :, 1:2].abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        pos      = pos / torso_h
        vel      = vel / torso_h

        out_parts.append(torch.cat([pos, vel], dim=-1))   # (B, T, J, 6)

    return torch.cat(out_parts, dim=-1)   # (B, T, J, 18)


def run_epoch(model, loader, optimiser, reg_fn, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    q_true, q_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for skels, qualities, exercise_ids in loader:
            skels        = centre_and_scale_multiview(skels.to(DEVICE))
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
        "loss": total_loss / n,
        "rmse": float(np.sqrt(np.mean((qt - qp) ** 2))),
        "mae" : float(np.mean(np.abs(qt - qp))),
        "r2"  : float(r2_score(qt, qp)) if len(qt) > 1 else 0.0,
        "pcc" : pcc,
    }


print("✓ centre_and_scale_multiview and run_epoch defined")


# ══════════════════════════════════════════════════════════════════════════
# Cell 14 — Plotting helpers   (identical to single-view)
# ══════════════════════════════════════════════════════════════════════════

def save_and_show(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {path}")


def plot_loss_curves(history, save_dir, test_loss=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["train_loss"], label="Train",      color="steelblue")
    ax.plot(epochs, history["val_loss"],   label="Validation", color="darkorange")
    if test_loss is not None:
        ax.axhline(test_loss, color="green", linestyle="-.", linewidth=1.5,
                   label=f"Test Loss={test_loss:.4f}")
    ax.set_title("Regression Loss (SmoothL1) — Early Fusion", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "loss_curve.png"))


def plot_rmse_mae(history, save_dir, test_rmse=None, test_mae=None):
    epochs = range(1, len(history["val_rmse"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("RMSE & MAE — Early Fusion Multi-View", fontweight="bold")
    for ax, m, tv in [(axes[0], "rmse", test_rmse), (axes[1], "mae", test_mae)]:
        ax.plot(epochs, history[f"train_{m}"], label="Train",      color="steelblue")
        ax.plot(epochs, history[f"val_{m}"],   label="Validation", color="darkorange")
        if tv is not None:
            ax.axhline(tv, color="green", linestyle="-.", linewidth=1.5,
                       label=f"Test={tv:.4f}")
        ax.set_title(m.upper()); ax.set_xlabel("Epoch"); ax.set_ylabel(m.upper())
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "rmse_mae.png"))


def plot_r2_pcc(history, save_dir, test_r2=None, test_pcc=None):
    epochs = range(1, len(history["val_r2"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("R² & PCC — Early Fusion Multi-View", fontweight="bold")
    for ax, m, tv, title in [
        (axes[0], "r2",  test_r2,  "R² Score"),
        (axes[1], "pcc", test_pcc, "Pearson Correlation"),
    ]:
        ax.plot(epochs, history[f"train_{m}"], label="Train",      color="steelblue")
        ax.plot(epochs, history[f"val_{m}"],   label="Validation", color="darkorange")
        if tv is not None:
            ax.axhline(tv, color="green", linestyle="-.", linewidth=1.5,
                       label=f"Test={tv:.4f}")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Perfect=1")
        ax.axhline(0.0, color="red",  linestyle=":", linewidth=1, label="Baseline=0")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "r2_pcc.png"))


def plot_regression_scatter(q_true, q_pred, split_name="Test", save_dir=None):
    qt   = np.array(q_true); qp = np.array(q_pred)
    r2   = float(r2_score(qt, qp)) if len(qt) > 1 else 0.0
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp) ** 2)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(qt, qp, alpha=0.6, edgecolors="black", linewidths=0.4,
               color="steelblue", s=60)
    lo = min(qt.min(), qp.min()) - 0.2; hi = max(qt.max(), qp.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True Quality Score"); ax.set_ylabel("Predicted Quality Score")
    ax.set_title(f"{split_name} — Early Fusion Multi-View", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f"R²={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    if save_dir:
        save_and_show(fig, os.path.join(save_dir,
                      f"scatter_{split_name.lower()}.png"))
    else:
        plt.close(fig)


def plot_early_stop(history, stopped_epoch, best_epoch, save_dir):
    epochs = range(1, len(history["val_mae"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["train_mae"], label="Train MAE", color="steelblue")
    ax.plot(epochs, history["val_mae"],   label="Val MAE",   color="darkorange")
    ax.axvline(best_epoch,    color="purple", linestyle=":",  linewidth=2,
               label=f"Best epoch ({best_epoch})")
    ax.axvline(stopped_epoch, color="red",    linestyle="--", linewidth=2,
               label=f"Early stop ({stopped_epoch})")
    ax.set_title("MAE + Early Stopping — Early Fusion", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "early_stopping.png"))


print("✓ Plotting helpers defined")


# ══════════════════════════════════════════════════════════════════════════
# Cell 15 — Early Stopping   (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_mae   = float("inf")
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


print("✓ EarlyStopping defined")


# ══════════════════════════════════════════════════════════════════════════
# Cell 16 — Training Loop
# ══════════════════════════════════════════════════════════════════════════

reg_fn = nn.SmoothL1Loss(beta=1.0)

train_loader = DataLoader(
    BZUDataset_EarlyFusion(train_df, augment=True),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(DEVICE == "cuda"))

val_loader = DataLoader(
    BZUDataset_EarlyFusion(val_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE == "cuda"))

test_loader = DataLoader(
    BZUDataset_EarlyFusion(test_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE == "cuda"))

model = STGCN_EarlyFusion(in_features=IN_FEATURES, K=3, dropout=0.5).to(DEVICE)

optimiser = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=50,
    min_lr=1e-6, verbose=True)

early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
SPLITS        = ["train", "val"]
METRICS       = ["loss", "rmse", "mae", "r2", "pcc"]
history       = {f"{s}_{m}": [] for s in SPLITS for m in METRICS}
stopped_epoch = EPOCHS

log.info("=" * 70)
log.info("STARTING EARLY-FUSION MULTI-VIEW TRAINING")
log.info(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
log.info("=" * 70)

print(f'\n{"═"*68}')
print(f'  Early Fusion Multi-View  |  in_features={IN_FEATURES}  ({NUM_VIEWS} views × 6)')
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'  Patience: {PATIENCE}  |  LR: {LR}  |  Batch: {BATCH_SIZE}')
print(f'{"═"*68}')

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, optimiser, reg_fn, is_train=True)
    vl = run_epoch(model, val_loader,   optimiser, reg_fn, is_train=False)
    scheduler.step(vl["mae"])

    for split, res in [("train", tr), ("val", vl)]:
        for m in METRICS:
            history[f"{split}_{m}"].append(res[m])

    stop, improved = early_stop.step(vl["mae"], model, epoch)

    star = " ★" if improved else ""
    msg  = (f"  Ep {epoch:3d}/{EPOCHS} | "
            f"Tr loss={tr['loss']:.4f} mae={tr['mae']:.3f} "
            f"r2={tr['r2']:.3f} pcc={tr['pcc']:.3f} | "
            f"Vl loss={vl['loss']:.4f} mae={vl['mae']:.3f} "
            f"r2={vl['r2']:.3f} pcc={vl['pcc']:.3f} | "
            f"ES {early_stop.counter}/{PATIENCE}{star}")
    print(msg)
    log.info(msg)

    if improved:
        print(f"    ★ val_mae={early_stop.best_mae:.4f}  "
              f"rmse={vl['rmse']:.4f}  r2={vl['r2']:.4f}  pcc={vl['pcc']:.4f}")

    if stop:
        stopped_epoch = epoch
        print(f"\n  ⏹  Early stopping at epoch {epoch} (best={early_stop.best_epoch})")
        log.info(f"Early stopping at epoch {epoch}  best={early_stop.best_epoch}")
        break

print("\n✓ Training complete!")

# ── Restore best weights ──────────────────────────────────────────────────
model.load_state_dict(early_stop.best_wts)
best_epoch = early_stop.best_epoch

# ── Final test evaluation ─────────────────────────────────────────────────
final_te = run_epoch(model, test_loader, optimiser, reg_fn, is_train=False)

print(f"\n  ── Final Test Results (best epoch = {best_epoch}) ──────────────────")
print(f"  Loss : {final_te['loss']:.4f}")
print(f"  RMSE : {final_te['rmse']:.4f}")
print(f"  MAE  : {final_te['mae']:.4f}")
print(f"  R²   : {final_te['r2']:.4f}")
print(f"  PCC  : {final_te['pcc']:.4f}")

# ── Collect predictions ───────────────────────────────────────────────────
model.eval()
all_true_q, all_pred_q, all_exercise_ids = [], [], []
with torch.no_grad():
    for skels, qualities, exercise_ids in test_loader:
        skels        = centre_and_scale_multiview(skels.to(DEVICE))
        exercise_ids = exercise_ids.to(DEVICE)
        preds        = model(skels, exercise_ids)
        all_true_q.extend(qualities.numpy())
        all_pred_q.extend(preds.cpu().numpy())
        all_exercise_ids.extend(exercise_ids.cpu().numpy())

# ── Save plots ────────────────────────────────────────────────────────────
plot_loss_curves(history, PLOTS_DIR, test_loss=final_te["loss"])
plot_rmse_mae(history, PLOTS_DIR,
              test_rmse=final_te["rmse"], test_mae=final_te["mae"])
plot_r2_pcc(history, PLOTS_DIR,
            test_r2=final_te["r2"], test_pcc=final_te["pcc"])
plot_regression_scatter(all_true_q, all_pred_q,
                        split_name="Test", save_dir=PLOTS_DIR)
plot_early_stop(history, stopped_epoch, best_epoch, PLOTS_DIR)

# ── Save history ──────────────────────────────────────────────────────────
json_path = os.path.join(LOGS_DIR, "training_history.json")
with open(json_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"  ✓ History JSON → {json_path}")

np.savez(os.path.join(LOGS_DIR, "training_history.npz"),
         **{k: np.array(v) for k, v in history.items()})

np.savez(os.path.join(LOGS_DIR, "test_predictions.npz"),
         q_true=np.array(all_true_q),
         q_pred=np.array(all_pred_q),
         exercise_ids=np.array(all_exercise_ids))


# ══════════════════════════════════════════════════════════════════════════
# Cell 16.5 — Per-exercise test metrics   (unchanged)
# ══════════════════════════════════════════════════════════════════════════

all_true_q_arr = np.array(all_true_q)
all_pred_q_arr = np.array(all_pred_q)
all_ex_arr     = np.array(all_exercise_ids)
unique_ex      = sorted(np.unique(all_ex_arr))
per_ex_results = {}

print("=" * 72)
print(f'  {"Exercise":<12} {"n":>5} {"MAE":>8} {"RMSE":>8} {"R²":>8} {"PCC":>8}')
print("─" * 72)

for ex_id in unique_ex:
    mask = all_ex_arr == ex_id
    qt   = all_true_q_arr[mask]; qp = all_pred_q_arr[mask]; n = mask.sum()
    mae  = float(np.mean(np.abs(qt - qp)))
    rmse = float(np.sqrt(np.mean((qt - qp) ** 2)))
    r2   = float(r2_score(qt, qp))    if n > 1 else float("nan")
    pcc  = float(pearsonr(qt, qp)[0]) if n > 1 else float("nan")
    per_ex_results[ex_id] = dict(n=n, mae=mae, rmse=rmse, r2=r2, pcc=pcc)
    print(f'  E{ex_id:<11} {n:>5} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pcc:>8.4f}')

print("─" * 72)
print(f'  {"Overall":<12} {len(all_true_q_arr):>5} '
      f'{final_te["mae"]:>8.4f} {final_te["rmse"]:>8.4f} '
      f'{final_te["r2"]:>8.4f} {final_te["pcc"]:>8.4f}')
print("=" * 72)

pd.DataFrame([{"exercise": f"E{ex}", **vals}
               for ex, vals in per_ex_results.items()]).to_csv(
    os.path.join(LOGS_DIR, "per_exercise_metrics.csv"), index=False)
print(f"\n  ✓ Per-exercise metrics CSV saved")


# ══════════════════════════════════════════════════════════════════════════
# Cell 17 — Final Summary
# ══════════════════════════════════════════════════════════════════════════

bv            = best_epoch - 1
best_val_mae  = history["val_mae"][bv]
best_val_rmse = history["val_rmse"][bv]
best_val_r2   = history["val_r2"][bv]
best_val_pcc  = history["val_pcc"][bv]

print("=" * 60)
print("  TRAINING SUMMARY — ST-GCN Early Fusion Multi-View")
print("=" * 60)
print(f"  Best Epoch       : {best_epoch}  (stopped at {stopped_epoch})")
print(f"  Best Val MAE     : {best_val_mae:.4f}")
print(f"  Best Val RMSE    : {best_val_rmse:.4f}")
print(f"  Best Val R²      : {best_val_r2:.4f}")
print(f"  Best Val PCC     : {best_val_pcc:.4f}")
print("─" * 60)
print(f"  Test MAE         : {final_te['mae']:.4f}")
print(f"  Test RMSE        : {final_te['rmse']:.4f}")
print(f"  Test R²          : {final_te['r2']:.4f}")
print(f"  Test PCC         : {final_te['pcc']:.4f}")
print("=" * 60)

log.info(f"Best Epoch={best_epoch}  stopped_epoch={stopped_epoch}")
log.info(f"Test MAE={final_te['mae']:.4f}")
log.info(f"Test RMSE={final_te['rmse']:.4f}")
log.info(f"Test R²={final_te['r2']:.4f}")
log.info(f"Test PCC={final_te['pcc']:.4f}")

summary_path = os.path.join(OUT_DIR, "training_summary_early_fusion.csv")
rows = [{"split": "test_overall", "exercise": "all",
         "fusion": "early",
         "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
         "val_mae": best_val_mae, "val_rmse": best_val_rmse,
         "val_r2": best_val_r2,   "val_pcc": best_val_pcc,
         "test_mae": final_te["mae"], "test_rmse": final_te["rmse"],
         "test_r2":  final_te["r2"], "test_pcc":  final_te["pcc"]}]
for ex, vals in per_ex_results.items():
    rows.append({"split": "test_per_exercise", "exercise": f"E{ex}",
                 "fusion": "early", **vals})
pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f"\n✓ Summary CSV → {summary_path}")

sys.stdout.restore()
print("✓ Log file closed.")
