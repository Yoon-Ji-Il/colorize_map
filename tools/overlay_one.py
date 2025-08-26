#!/usr/bin/env python3
import sys, os, json, numpy as np, cv2
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lidarProjectionUtils, pointcloudIO

PARAMS="params/camera-parameters.json"; DEVICE="CAM_MAIN"
PCD=sys.argv[1]; IMG=sys.argv[2]; OUT=sys.argv[3] if len(sys.argv)>3 else "overlay.png"
DOT=2; DEPTH=0.05

cam={p["device_id"]:p for p in json.load(open(PARAMS))}[DEVICE]
pts=pointcloudIO.read_pcd_file(PCD); img=np.array(Image.open(IMG)); H,W=img.shape[:2]; vis=img.copy()
N=min(len(pts), 50000); idx=np.random.choice(len(pts), N, replace=False); pts=pts[idx]
ok=0
for p in pts:
    c=lidarProjectionUtils.world2cam(cam["extrinsic"], p[:3])
    if c[2]<=DEPTH: continue
    u,v=lidarProjectionUtils.cam2image(cam, c, cam["cameraModel"])
    x,y=int(u),int(v)
    if 0<=x<W and 0<=y<H: cv2.circle(vis,(x,y),DOT,(0,255,0),-1); ok+=1
cv2.imwrite(OUT, vis); print("[OK]", ok, "pts ->", OUT)
