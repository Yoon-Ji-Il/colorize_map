#!/usr/bin/env python3
import os, sys, time, struct
import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def load_points(path):
    # .ply 또는 .pcd 모두 지원 (open3d)
    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points)           # (N,3) float
    cols = np.asarray(pc.colors)          # (N,3) float [0,1] or empty
    if cols.size == 0:
        cols = np.ones_like(pts)          # 없으면 흰색
    cols_u8 = (cols * 255.0).clip(0,255).astype(np.uint8)
    # pack rgb into single float32 (ROS convention)
    rgb_uint32 = (cols_u8[:,0].astype(np.uint32) << 16) | \
                 (cols_u8[:,1].astype(np.uint32) << 8)  | \
                  cols_u8[:,2].astype(np.uint32)
    rgb_float = rgb_uint32.view(np.float32)
    # Nx4: x,y,z,rgb(float32)
    arr = np.zeros((pts.shape[0], 4), dtype=np.float32)
    arr[:,0:3] = pts.astype(np.float32)
    arr[:,3] = rgb_float
    return arr

def to_pointcloud2(points, frame_id="map", stamp=None):
    """
    points: (N,4) float32 array [x,y,z,rgb(float32)]
    """
    fields = [
        PointField('x',   0, PointField.FLOAT32, 1),
        PointField('y',   4, PointField.FLOAT32, 1),
        PointField('z',   8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1),
    ]
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp if stamp is not None else rospy.Time.now()
    data = points.tobytes()  # already tightly packed float32
    msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=16,             # 4 floats * 4 bytes
        row_step=16*points.shape[0],
        data=data,
    )
    return msg

def main():
    if len(sys.argv) < 2:
        print("Usage: rosrun <pkg> pub_pointcloud.py /path/to/merged_colorized.(ply|pcd) [frame_id] [rate_hz]")
        sys.exit(1)
    path = sys.argv[1]
    frame_id = sys.argv[2] if len(sys.argv) >= 3 else "map"
    rate_hz = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0

    rospy.init_node("pub_pointcloud", anonymous=True)
    pub = rospy.Publisher("/colorized_cloud", PointCloud2, queue_size=1, latch=True)

    rospy.loginfo(f"Loading point cloud: {path}")
    arr = load_points(path)
    rospy.loginfo(f"Loaded {arr.shape[0]} points")
    msg = to_pointcloud2(arr, frame_id=frame_id)

    # 1회 발행 후 latched로 유지, 그래도 RViz 접속 시 확실히 보이게 몇 번 더 쏨
    r = rospy.Rate(rate_hz)
    for _ in range(5):
        pub.publish(msg)
        r.sleep()

    rospy.loginfo("Latched. Ctrl+C to exit.")
    # 노드 살아있게 유지 (RViz 재접속 시에도 latched로 전달됨)
    rospy.spin()

if __name__ == "__main__":
    main()
