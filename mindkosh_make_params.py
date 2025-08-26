# /home/jiil/mindkosh_make_params.py
import json, numpy as np, sys
from scipy.spatial.transform import Rotation as R

# === 입력값 설정 ===
CALIB_JSON    = "/home/jiil/pre_0823/calib.json"   # koide3 결과 파일 경로로 바꿔줘
OUT_JSON      = "/home/jiil/colorize_ws/colorize-lidar-pointcloudcamera_params.json"        # Mindkosh에서 읽을 파라미터 파일
DEVICE_ID     = "CAM_MAIN"                             # 임의의 카메라 ID(원하면 바꿔도 됨)

# pinhole intrinsics (반드시 너의 값으로 바꿔줘!)
fx, fy, cx, cy = 645.4690, 646.8439, 347.1880, 246.1243

# 왜곡계수(브라운: k1,k2,p1,p2) — 없으면 []로 두거나 아래 4개만 넣어줘
distortion = [0.1595, -0.4787, -0.001222, 0.001]  # 예시

# === koide3 extrinsic 로드 ===
data = json.load(open(CALIB_JSON))

# 보통 results 안에 들어있음. 키 이름이 다를 수도 있어 대비:
T_lidar_camera = None
candidates = [
    ("results", "T_lidar_camera"),
    ("results", "T_camera_lidar"),   # 혹시 반대 이름으로 저장된 경우 대비
    ("T_lidar_camera",),
    ("T_camera_lidar",),
]
for keypath in candidates:
    try:
        tmp = data
        for k in keypath:
            tmp = tmp[k]
        if isinstance(tmp, list) and len(tmp) == 7:
            T_lidar_camera = tmp
            break
    except Exception:
        pass

if T_lidar_camera is None:
    sys.exit("ERROR: calib.json에서 T_lidar_camera / T_camera_lidar 를 찾지 못했습니다.")

x,y,z,qx,qy,qz,qw = T_lidar_camera

# 주의: koide3의 이 값이 'Camera→LiDAR'인 경우가 많음
# Camera→LiDAR 4x4
T_lc = np.eye(4)
T_lc[:3,:3] = R.from_quat([qx,qy,qz,qw]).as_matrix()
T_lc[:3, 3] = [x,y,z]

# Mindkosh는 LiDAR→Camera가 필요하므로 역행렬 취함
T_cl = np.linalg.inv(T_lc)

# === Mindkosh 포맷으로 저장 (단일 카메라 예시: 리스트에 1개 dict) ===
out = [{
    "device_id": DEVICE_ID,
    "cameraModel": "PINHOLE",
    "intrinsic": [float(fx), float(fy), float(cx), float(cy)],
    # distortion이 없으면 이 키를 아예 빼도 됨
    "distortion": [float(d) for d in distortion],
    "extrinsic": T_cl.tolist()  # LiDAR→Camera 4x4
}]

json.dump(out, open(OUT_JSON, "w"), indent=2)
print(f"Wrote {OUT_JSON}")
