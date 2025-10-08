import os, glob, h5py, numpy as np
from tqdm.auto import tqdm

SRC_DIRS = ['easyvoxels', 'medvoxels', 'auxvoxels', 'hardvoxels']

def down2x2_sum(x):
    T, C, H, W = x.shape
    Ht, Wt = (H // 2) * 2, (W // 2) * 2
    x = x[:, :, :Ht, :Wt]
    return x.reshape(T, C, Ht // 2, 2, Wt // 2, 2).sum(axis=(3, 5))

print('byrja')
for src_dir in SRC_DIRS:
    print(src_dir)
    dst_dir = src_dir + "x"
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.h5")))
    for src in tqdm(files, desc=os.path.basename(src_dir) or src_dir, unit="file"):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            continue
        with h5py.File(src, "r") as f:
            key = "voxel" if "voxel" in f else "data"
            x = f[key][()]
        x = down2x2_sum(x).astype(np.float32)
        with h5py.File(dst, "w") as f:
            f.create_dataset(key, data=x)
