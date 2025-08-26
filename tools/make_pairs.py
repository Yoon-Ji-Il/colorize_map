#!/usr/bin/env python3
import os, glob
from datetime import datetime

IM_DIR = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/images"
PCD_DIR = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/clouds"
OUT_LIST = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/pairs.txt"

def parse_ts(name):
    # 파일명: YYYYMMDD_HHMMSS.NNNNNNNNN.ext  → float(sec)
    base = os.path.splitext(os.path.basename(name))[0]
    date, frac = base.split("_")[0], base.split("_")[1]
    hhmmss, nsec = frac.split(".")[0], frac.split(".")[1]
    dt = datetime.strptime(date + "_" + hhmmss, "%Y%m%d_%H%M%S")
    t = dt.timestamp() + int(nsec)/1e9
    return t

imgs = sorted(glob.glob(os.path.join(IM_DIR, "*.jpg"))) + \
       sorted(glob.glob(os.path.join(IM_DIR, "*.png")))
pcds = sorted(glob.glob(os.path.join(PCD_DIR, "*.pcd")))

if not imgs or not pcds:
    raise SystemExit("이미지/포인트클라우드가 비어있습니다.")

img_ts = [(parse_ts(f), f) for f in imgs]

def nearest_image(t):
    # 이진탐색 없이 간단히 선형(개수 많으면 개선 가능)
    best = None
    best_dt = 1e9
    for ti, fi in img_ts:
        dt = abs(t - ti)
        if dt < best_dt:
            best_dt = dt; best = fi
    return best, best_dt

pairs = []
for p in pcds:
    tp = parse_ts(p)
    fi, dt = nearest_image(tp)
    pairs.append((p, fi, dt))

# 저장: pcd_path|image_path|dt
with open(OUT_LIST, "w") as f:
    for p, fi, dt in pairs:
        f.write(f"{p}|{fi}|{dt:.6f}\n")

print(f"[OK] {len(pairs)} pairs -> {OUT_LIST}")
print("예시 5줄:")
for row in pairs[:5]:
    print(row)
