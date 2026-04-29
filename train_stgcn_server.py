import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 🔴 عدّل هذا المسار فقط
DATASET_DIR = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"

# ─────────────────────────────────────────────
# Load first NPZ file
# ─────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATASET_DIR, '**/*.npz'), recursive=True))

if len(files) == 0:
    raise Exception("❌ No NPZ files found")

print(f"✓ Found {len(files)} files")
print(f"✓ Using file: {files[0]}")

data = np.load(files[0], allow_pickle=True)

print("Keys:", list(data.keys()))

# حاول تلقائي يختار الكي الصحيح
if "keypoints_3d" in data:
    skel = data["keypoints_3d"]
else:
    skel = data[list(data.keys())[0]]

# ─────────────────────────────────────────────
# Fix shape
# ─────────────────────────────────────────────
skel = skel.astype(np.float32)

if skel.ndim == 2:
    skel = skel.reshape(skel.shape[0], 17, 3)
elif skel.ndim == 4:
    skel = skel.squeeze(0)

print("Shape:", skel.shape)  # (T,17,3)

# ─────────────────────────────────────────────
# Plot joint indices
# ─────────────────────────────────────────────
def plot_joints(skel, frame_idx=0):
    pts = skel[frame_idx]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle("Joint Order Debug", fontsize=14)

    # XY
    axes[0].scatter(x, y, c='red')
    for i in range(17):
        axes[0].text(x[i], y[i], str(i), color='yellow', fontsize=12)
    axes[0].set_title("Front (X-Y)")
    axes[0].invert_yaxis()
    axes[0].grid()

    # ZY
    axes[1].scatter(z, y, c='blue')
    for i in range(17):
        axes[1].text(z[i], y[i], str(i), color='black', fontsize=12)
    axes[1].set_title("Side (Z-Y)")
    axes[1].invert_yaxis()
    axes[1].grid()

    # XZ
    axes[2].scatter(x, -z, c='green')
    for i in range(17):
        axes[2].text(x[i], -z[i], str(i), color='black', fontsize=12)
    axes[2].set_title("Top (X-Z)")
    axes[2].grid()

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
plot_joints(skel)
