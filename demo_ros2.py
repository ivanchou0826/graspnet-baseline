"""Standalone GraspNet validation script — bypasses graspnet_ros2 package.

Subscribes to camera topics, runs GraspNet inference on demand, and publishes
results to RViz2 for visual verification.

Usage (Isaac Sim defaults):
  python3 demo_ros2.py --checkpoint_path checkpoint-rs.tar

Trigger inference:
  ros2 service call /graspnet_demo/trigger std_srvs/srv/Trigger {}

RViz2 topics:
  /graspnet_demo/markers     — gripper LINE_LIST (red=low score, green=high)
  /graspnet_demo/pointcloud  — masked point cloud fed to the model
"""

import os
import sys
import argparse
import numpy as np
import torch

_ros_distro = os.environ.get('ROS_DISTRO', 'humble')
_ros_site = f'/opt/ros/{_ros_distro}/local/lib/python3.10/dist-packages'
if _ros_site not in sys.path:
    sys.path.insert(0, _ros_site)

GRASPNET_ROOT = os.environ.get(
    'GRASPNET_ROOT',
    os.path.dirname(os.path.abspath(__file__))
)
for _sub in ('models', 'utils'):
    _p = os.path.join(GRASPNET_ROOT, _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspCameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


# ── Image decoding ────────────────────────────────────────────────────────────

def _imgmsg_to_numpy(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ('16uc1', '16sc1'):
        dtype = np.uint16 if enc == '16uc1' else np.int16
        return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    elif enc == '32fc1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    elif enc == 'rgb8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif enc == 'bgr8':
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1].copy()
    elif enc in ('bgra8', 'rgba8'):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        rgb = arr[:, :, :3]
        return rgb[:, :, ::-1].copy() if enc == 'bgra8' else rgb.copy()
    else:
        raise ValueError(f'Unsupported encoding: {msg.encoding}')


# ── PointCloud2 builder ───────────────────────────────────────────────────────

def _build_pointcloud2(xyz: np.ndarray, rgb_f: np.ndarray, header) -> PointCloud2:
    """Nx3 xyz + Nx3 rgb [0,1] → sensor_msgs/PointCloud2 (XYZRGB packed)."""
    import struct
    n = len(xyz)
    rgb_u8 = (np.clip(rgb_f, 0, 1) * 255).astype(np.uint8)
    packed = np.zeros(n, dtype=np.float32)
    for i in range(n):
        r, g, b = int(rgb_u8[i, 0]), int(rgb_u8[i, 1]), int(rgb_u8[i, 2])
        packed[i] = struct.unpack('f', struct.pack('BBBB', b, g, r, 0))[0]

    pts = np.column_stack([xyz.astype(np.float32), packed]).tobytes()

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.data = pts
    msg.is_dense = True
    return msg


# ── Gripper marker builder (same geometry as graspnet_node.py) ────────────────

def _make_gripper_marker(marker_id, R, t, width, depth, score, score_max, header):
    half_w = width / 2.0
    pts_local = np.array([
        [-depth, -half_w, 0.0],       # left finger at palm
        [0.0,    -half_w, 0.0],       # left finger contact (at grasp center t)
        [-depth,  half_w, 0.0],       # right finger at palm
        [0.0,     half_w, 0.0],       # right finger contact
        [-depth, -half_w, 0.0],       # palm bar left
        [-depth,  half_w, 0.0],       # palm bar right
        [-depth - 0.04, 0.0, 0.0],   # approach arm start
        [-depth,  0.0,   0.0],        # approach arm end (palm centre)
    ], dtype=np.float32)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    frac = float(np.clip(score / max(score_max, 1e-6), 0.0, 1.0))

    m = Marker()
    m.header = header
    m.ns = 'graspnet_demo'
    m.id = marker_id
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.004
    m.color.r = 1.0 - frac
    m.color.g = frac
    m.color.b = 0.0
    m.color.a = 1.0
    m.pose.orientation.w = 1.0

    for i, j in pairs:
        for idx in (i, j):
            world = t + R @ pts_local[idx]
            p = Point()
            p.x, p.y, p.z = float(world[0]), float(world[1]), float(world[2])
            m.points.append(p)
    return m


# ── Main node ─────────────────────────────────────────────────────────────────

class DemoNode(Node):
    def __init__(self, args):
        super().__init__('graspnet_demo')
        self.args = args
        self._last_frame = None

        # Load model
        self.get_logger().info(f'Loading: {args.checkpoint_path}')
        self.net = GraspNet(
            input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        ckpt = torch.load(args.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.net.eval()
        self.get_logger().info(f'Checkpoint loaded (epoch {ckpt["epoch"]})')

        # Publishers
        self._pub_markers = self.create_publisher(MarkerArray, '/graspnet_demo/markers', 1)
        self._pub_cloud   = self.create_publisher(PointCloud2,  '/graspnet_demo/pointcloud', 1)

        # Trigger service
        self.create_service(Trigger, '/graspnet_demo/trigger', self._trigger_cb)

        # Synchronized subscribers
        rgb_sub   = message_filters.Subscriber(self, Image,      args.rgb_topic)
        depth_sub = message_filters.Subscriber(self, Image,      args.depth_topic)
        info_sub  = message_filters.Subscriber(self, CameraInfo, args.info_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self._sync.registerCallback(self._frame_cb)

        self.get_logger().info(
            'Ready.\n'
            f'  RGB:   {args.rgb_topic}\n'
            f'  Depth: {args.depth_topic}\n'
            f'  Info:  {args.info_topic}\n'
            'Trigger: ros2 service call /graspnet_demo/trigger std_srvs/srv/Trigger {}')

    def _frame_cb(self, rgb_msg, depth_msg, info_msg):
        self._last_frame = (rgb_msg, depth_msg, info_msg)

    def _trigger_cb(self, req, resp):
        if self._last_frame is None:
            resp.success = False
            resp.message = 'No camera frame received yet.'
            return resp
        msg = self._run(*self._last_frame)
        resp.success = True
        resp.message = msg
        return resp

    def _run(self, rgb_msg, depth_msg, info_msg):
        # ── Decode ──────────────────────────────────────────────────────────
        color_rgb = _imgmsg_to_numpy(rgb_msg)
        depth_raw = _imgmsg_to_numpy(depth_msg)
        enc = depth_msg.encoding.lower()
        factor_depth = 1000.0 if '16' in enc else 1.0

        K = info_msg.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        H, W = depth_raw.shape[:2]
        self.get_logger().info(
            f'Frame: {W}x{H}  encoding={depth_msg.encoding}  factor={factor_depth}'
            f'  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}')

        # ── Point cloud + depth mask ─────────────────────────────────────────
        cam = GraspCameraInfo(float(W), float(H), fx, fy, cx, cy, factor_depth)
        cloud_org = create_point_cloud_from_depth_image(depth_raw, cam, organized=True)

        depth_mask = (depth_raw > 0) & (depth_raw < self.args.max_depth * factor_depth)
        cloud_masked = cloud_org[depth_mask]
        color_masked = (color_rgb / 255.0)[depth_mask]

        self.get_logger().info(f'Valid points: {len(cloud_masked)} / {H * W}')
        self._pub_cloud.publish(_build_pointcloud2(cloud_masked, color_masked, rgb_msg.header))

        if len(cloud_masked) == 0:
            return f'No points within max_depth={self.args.max_depth}m'

        # ── Sample ──────────────────────────────────────────────────────────
        n = self.args.num_point
        if len(cloud_masked) >= n:
            idxs = np.random.choice(len(cloud_masked), n, replace=False)
        else:
            idxs = np.concatenate([
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), n - len(cloud_masked), replace=True)
            ])
        cloud_sampled = cloud_masked[idxs]

        # ── Inference ───────────────────────────────────────────────────────
        end_points = {
            'point_clouds': torch.from_numpy(
                cloud_sampled[np.newaxis].astype(np.float32)).to(self.device),
            'cloud_colors': (color_masked[idxs]),
        }
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        self.get_logger().info(f'Raw candidates: {len(gg)}')

        # ── Collision filter ─────────────────────────────────────────────────
        if self.args.collision_thresh > 0:
            mfc = ModelFreeCollisionDetector(cloud_masked, voxel_size=self.args.voxel_size)
            col_mask = mfc.detect(gg, approach_dist=self.args.approach_dist,
                                  collision_thresh=self.args.collision_thresh)
            gg = gg[~col_mask]
            self.get_logger().info(f'After collision filter: {len(gg)}')

        # ── NMS + sort ───────────────────────────────────────────────────────
        gg.nms()
        gg.sort_by_score()
        gg = gg[:self.args.top_k]

        # ── Score filter ─────────────────────────────────────────────────────
        if self.args.min_score > 0 and len(gg) > 0:
            keep = gg.grasp_group_array[:, 0] >= self.args.min_score
            gg = GraspGroup(gg.grasp_group_array[keep])
            self.get_logger().info(f'After score filter (>={self.args.min_score}): {len(gg)}')

        # ── Log results ──────────────────────────────────────────────────────
        arr = gg.grasp_group_array
        score_max = float(arr[0, 0]) if len(arr) > 0 else 1.0
        self.get_logger().info(f'Top-{len(gg)} grasps:')
        for i, g in enumerate(gg):
            p = g.translation
            self.get_logger().info(
                f'  [{i:2d}] score={g.score:.3f}  '
                f'pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})  '
                f'width={g.width:.3f}  depth={g.depth:.3f}')

        # ── Publish markers ──────────────────────────────────────────────────
        ma = MarkerArray()
        del_m = Marker()
        del_m.header = rgb_msg.header
        del_m.ns = 'graspnet_demo'
        del_m.id = -1
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)

        for i, row in enumerate(arr):
            score = row[0]
            width = row[1]
            depth = row[3]
            R = row[4:13].reshape(3, 3)
            t = row[13:16]
            ma.markers.append(
                _make_gripper_marker(i, R, t, width, depth, score, score_max,
                                     rgb_msg.header))
        self._pub_markers.publish(ma)

        return f'Published {len(gg)} grasps. Best score={score_max:.3f}'


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoint-rs.tar')
    parser.add_argument('--rgb_topic',        default='/rgb/camera_1')
    parser.add_argument('--depth_topic',      default='/camera_1/depth/image_raw')
    parser.add_argument('--info_topic',       default='/camera_1/depth/camera_info')
    parser.add_argument('--num_point',        type=int,   default=20000)
    parser.add_argument('--max_depth',        type=float, default=3.0)
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision threshold in metres; -1 to disable')
    parser.add_argument('--voxel_size',       type=float, default=0.01,
                        help='Voxel size for collision detection point cloud')
    parser.add_argument('--approach_dist',    type=float, default=0.05,
                        help='Distance along approach direction to check for collisions')
    parser.add_argument('--top_k',            type=int,   default=50)
    parser.add_argument('--min_score',        type=float, default=0.0,
                        help='Filter out grasps below this score (0 = disabled)')
    args = parser.parse_args()

    rclpy.init()
    node = DemoNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
