#!/usr/bin/env python3
import os, glob, numpy as np, open3d as o3d

IN_DIR  = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/colorized"
OUT_MERGED = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/merged_colorized.ply"
VOXEL = 0.02  # 2cm 다운샘플 (밀도에 맞게 0.01~0.05 조절)

def to_pcd(arr):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(arr[:,:3].astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector((arr[:,3:6]/255.0).astype(np.float64))
    return pc

def from_ply(path):
    p = o3d.io.read_point_cloud(path)
    pts = np.asarray(p.points)
    cols = np.asarray(p.colors)
    if cols.size==0:
        cols = np.ones_like(pts)*[1,1,1]
    arr = np.hstack([pts, (cols*255.0).astype(np.uint8)])
    return arr

files = sorted(glob.glob(os.path.join(IN_DIR, "*_colorized.ply")))
print(f"[INFO] merging {len(files)} files")
chunks = []
for i,f in enumerate(files,1):
    arr = from_ply(f)
    chunks.append(arr)
    if i%20==0: print(f"  loaded {i}/{len(files)}")

if not chunks:
    raise SystemExit("No colorized ply found.")
all_arr = np.vstack(chunks)
pc = to_pcd(all_arr)
if VOXEL>0:
    pc = pc.voxel_down_sample(VOXEL)
o3d.io.write_point_cloud(OUT_MERGED, pc, write_ascii=False)
print("[OK] wrote", OUT_MERGED, "with", np.asarray(pc.points).shape[0], "points")
