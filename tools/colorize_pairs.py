#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, json
import numpy as np
from PIL import Image

import lidarProjectionUtils
import pointcloudIO

PARAMS_JSON = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/params/camera-parameters.json"
PAIRS_TXT   = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/pairs.txt"
OUT_DIR     = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/colorized"
DEVICE_ID   = "CAM_MAIN"       # params JSON의 device_id와 동일해야 함
DT_MAX      = 0.20             # 이미지-포인트 시간차 허용 (초)

MAX_DEPTH = 9_000.0
DEPTH_THRESHOLD = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

raw_params = json.load(open(PARAMS_JSON))
cam_dict = {p["device_id"]: p for p in raw_params}
if DEVICE_ID not in cam_dict:
    raise SystemExit(f"device_id '{DEVICE_ID}' not found in {PARAMS_JSON}")
cam = cam_dict[DEVICE_ID]

# pairs.txt 읽기
pairs = []
for line in open(PAIRS_TXT):
    pcd, img, dt = line.strip().split("|")
    dt = float(dt)
    if dt <= DT_MAX:
        pairs.append((pcd, img))
print(f"[INFO] {len(pairs)} pairs within dt <= {DT_MAX}s")

for idx, (pcd_path, img_path) in enumerate(pairs, 1):
    # 입력 로드
    pts = pointcloudIO.read_pcd_file(pcd_path)    # Nx3 (x,y,z)
    if pts.size == 0:
        continue
    image = np.array(Image.open(img_path))
    H, W = image.shape[:2]

    point_rgb = np.full((pts.shape[0], 3), 255, dtype=np.uint8)
    point_depth = np.full(pts.shape[0], MAX_DEPTH, dtype=np.float32)
    pixel_point_map = np.full((H, W), 0, dtype=np.uint32)
    pixel_point_map_set = np.zeros((H, W), dtype=bool)

    for i, pt in enumerate(pts):
        # LiDAR → Camera
        cam_pt = lidarProjectionUtils.world2cam(cam["extrinsic"], pt[:3])
        # Camera → Image
        u, v = lidarProjectionUtils.cam2image(cam, cam_pt, cam["cameraModel"])

        x, y = int(u), int(v)
        z = float(cam_pt[2])
        if 0 <= x < W and 0 <= y < H and z > DEPTH_THRESHOLD:
            # 픽셀 경쟁: 미할당 또는 충분히 더 가까우면 갱신
            if (not pixel_point_map_set[y, x]) or (z < point_depth[pixel_point_map[y, x]] - 0.3):
                if pixel_point_map_set[y, x]:
                    old = pixel_point_map[y, x]
                    #if point_depth[old] > z and abs(point_depth[old] - z) > 0.5:
                        #point_rgb[old] = [255, 255, 255]

                pixel_point_map[y, x] = i
                pixel_point_map_set[y, x] = True
                if point_depth[i] > z:
                    point_depth[i] = z
                point_rgb[i] = image[y, x]

    rgb_points = np.hstack((pts[:, :3], np.array(point_rgb)))
    out_name = os.path.splitext(os.path.basename(pcd_path))[0] + "_colorized.ply"
    out_path = os.path.join(OUT_DIR, out_name)
    pointcloudIO.save_ply(rgb_points, out_path)
    if idx % 10 == 0:
        print(f"[{idx}/{len(pairs)}] {out_path}")

print("[DONE] colorized point clouds saved to:", OUT_DIR)
