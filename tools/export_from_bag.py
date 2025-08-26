#!/usr/bin/env python3
import os, cv2, numpy as np
from datetime import datetime
import rosbag
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

# ===== 사용자 설정 =====
BAG_PATH    = "/home/jiil/0823/0823_extrinsic.bag"
IMG_TOPIC   = "/camera/color/image_raw"
PCD_TOPIC   = "/livox/lidar"

OUT_IMG_DIR = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/images"
OUT_PCD_DIR = "/home/jiil/colorize_ws/colorize-lidar-pointcloud/exports/clouds"

SAVE_IMG_EXT = ".jpg"   # ".png"도 가능
JPG_QUALITY  = 95
# ======================

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_PCD_DIR, exist_ok=True)

bridge = CvBridge()

def stamp_to_str(stamp):
    # 파일명: YYYYMMDD_HHMMSS.NNNNNNNNN (UTC)
    try:
        sec = int(stamp.secs)      # rospy.Time
        nsec = int(stamp.nsecs)
    except AttributeError:
        # fallback: rosbag read_messages t 또는 naive float
        return datetime.utcfromtimestamp(float(stamp)).strftime("%Y%m%d_%H%M%S.%f")[:-3] + "000"
    return f"{datetime.utcfromtimestamp(sec).strftime('%Y%m%d_%H%M%S')}.{nsec:09d}"

def save_image(msg):
    # msg는 sensor_msgs/Image 타입 (동적 모듈 이름이라 isinstance 검사 안함)
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    ts = getattr(msg.header, "stamp", None)
    if ts is None:
        raise RuntimeError("Image msg has no header.stamp")
    fname = os.path.join(OUT_IMG_DIR, stamp_to_str(ts) + SAVE_IMG_EXT)
    if SAVE_IMG_EXT.lower() == ".jpg":
        cv2.imwrite(fname, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
    else:
        cv2.imwrite(fname, cv_img)
    return fname

def save_pointcloud(msg):
    # msg는 sensor_msgs/PointCloud2
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(arr.astype(np.float64))
    ts = getattr(msg.header, "stamp", None)
    if ts is None:
        raise RuntimeError("PointCloud2 msg has no header.stamp")
    fname = os.path.join(OUT_PCD_DIR, stamp_to_str(ts) + ".pcd")
    o3d.io.write_point_cloud(fname, pc, write_ascii=False, compressed=False)
    return fname

def main():
    print(f"[INFO] Reading bag: {BAG_PATH}")
    bag = rosbag.Bag(BAG_PATH, "r")
    img_count = 0
    pcd_count = 0

    for topic, msg, t in bag.read_messages(topics=[IMG_TOPIC, PCD_TOPIC]):
        try:
            if topic == IMG_TOPIC:
                out = save_image(msg)
                img_count += 1
                if img_count % 50 == 0:
                    print(f"[IMG] {img_count} saved, last: {out}")
            elif topic == PCD_TOPIC:
                out = save_pointcloud(msg)
                if out:
                    pcd_count += 1
                    if pcd_count % 10 == 0:
                        print(f"[PCD] {pcd_count} saved, last: {out}")
        except Exception as e:
            print(f"[WARN] topic={topic}: {e}")

    bag.close()
    print(f"[DONE] images: {img_count}, clouds: {pcd_count}")
    print(f"[OUT] {OUT_IMG_DIR}")
    print(f"[OUT] {OUT_PCD_DIR}")

if __name__ == "__main__":
    main()
