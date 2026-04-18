"""GraspNet ROS2 node.

Subscribes to RGB, depth, and camera_info topics, runs GraspNet inference,
and publishes a visualization image with projected gripper rectangles.
"""

import os
import sys

# Allow running with venv while ROS2 packages live in the system Python path.
# Adjust if your ROS distro or Python version differs.
_ROS_PYTHON = f"/opt/ros/{os.environ.get('ROS_DISTRO', 'humble')}/lib/python3.10/site-packages"
if _ROS_PYTHON not in sys.path:
    sys.path.insert(0, _ROS_PYTHON)

import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspCameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


def _score_to_bgr(score: float, score_max: float = 1.2):
    """Map score [0, score_max] to BGR color (red=low, green=high)."""
    t = float(np.clip(score / score_max, 0.0, 1.0))
    r = int((1.0 - t) * 255)
    g = int(t * 255)
    return (0, g, r)


def _project(pt3d, fx, fy, cx, cy):
    """Project a 3-D point to pixel (u, v). Returns None if behind camera."""
    x, y, z = pt3d
    if z <= 1e-4:
        return None
    return (int(fx * x / z + cx), int(fy * y / z + cy))


def _draw_grasp(img, R, t, width, depth, score, fx, fy, cx, cy, score_max=1.2):
    """Draw a projected gripper rectangle onto img (BGR, in-place)."""
    # Gripper frame: X = depth axis (fingers point +X), Y = opening axis
    half_w = width / 2.0
    corners_local = np.array([
        [depth,  -half_w, 0.0],   # left  tip
        [depth,   half_w, 0.0],   # right tip
        [0.0,     half_w, 0.0],   # right base
        [0.0,    -half_w, 0.0],   # left  base
    ], dtype=np.float32)

    pixels = []
    for c in corners_local:
        world_pt = t + R @ c
        px = _project(world_pt, fx, fy, cx, cy)
        if px is None:
            return
        pixels.append(px)

    pts = np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))
    color = _score_to_bgr(score, score_max=score_max)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    # Mark approach direction: line from base-center toward tip-center
    base_mid = ((pixels[2][0] + pixels[3][0]) // 2,
                (pixels[2][1] + pixels[3][1]) // 2)
    tip_mid  = ((pixels[0][0] + pixels[1][0]) // 2,
                (pixels[0][1] + pixels[1][1]) // 2)
    cv2.arrowedLine(img, base_mid, tip_mid, color, 1, tipLength=0.3)


class GraspNetNode(Node):
    def __init__(self):
        super().__init__('graspnet_node')

        # ---------- parameters ----------
        self.declare_parameter('checkpoint_path', './checkpoint-rs.tar')
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        self.declare_parameter('collision_thresh', 0.01)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('top_k', 50)

        ckpt   = self.get_parameter('checkpoint_path').value
        n_view = self.get_parameter('num_view').value

        # ---------- model ----------
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        net = GraspNet(input_feature_dim=0, num_view=n_view, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02,
                       hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        net.to(device)
        checkpoint = torch.load(ckpt, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.get_logger().info(
            f"Loaded checkpoint '{ckpt}' (epoch {checkpoint['epoch']})")
        net.eval()
        self.net = net

        # ---------- misc ----------
        self.bridge = CvBridge()

        # ---------- subscriptions (synchronized) ----------
        qos = rclpy.qos.QoSProfile(depth=10)
        sub_rgb   = message_filters.Subscriber(self, Image, '/camera_2/image',   qos_profile=qos)
        sub_depth = message_filters.Subscriber(self, Image, '/camera_2/depth',   qos_profile=qos)
        sub_info  = message_filters.Subscriber(self, CameraInfo, '/camera_2/info', qos_profile=qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # ---------- publisher ----------
        self.pub = self.create_publisher(Image, '/graspnet/visualization', 10)
        self.get_logger().info('GraspNet node ready — waiting for camera topics...')

    # ------------------------------------------------------------------
    def callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        # --- decode ROS images ---
        color_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Auto-detect depth encoding: 16UC1 (mm) or 32FC1 (meters)
        enc = depth_msg.encoding.lower()
        if '16' in enc:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            factor_depth = 1000.0
        else:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            factor_depth = 1.0

        # --- camera intrinsics ---
        K = info_msg.k          # row-major 3×3 flattened
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        H, W   = depth_raw.shape[:2]

        # --- point cloud ---
        cam_info = GraspCameraInfo(float(W), float(H), fx, fy, cx, cy, factor_depth)
        cloud_organized = create_point_cloud_from_depth_image(
            depth_raw, cam_info, organized=True)

        mask = (depth_raw > 0)
        cloud_masked = cloud_organized[mask]            # (N, 3)
        color_masked = (color_rgb / 255.0)[mask]        # (N, 3)

        if len(cloud_masked) == 0:
            self.get_logger().warn('No valid depth points — skipping frame.')
            return

        # --- sample to num_point ---
        num_point = self.get_parameter('num_point').value
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
        else:
            idxs = np.concatenate([
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked),
                                 num_point - len(cloud_masked), replace=True)
            ])
        cloud_sampled = cloud_masked[idxs].astype(np.float32)

        end_points = {
            'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis]).to(self.device)
        }

        # --- inference ---
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        if len(gg_array) == 0:
            self.get_logger().warn('No grasps predicted.')
            self._publish_image(color_bgr, rgb_msg.header)
            return
        gg = GraspGroup(gg_array)

        # --- collision detection ---
        collision_thresh = self.get_parameter('collision_thresh').value
        voxel_size       = self.get_parameter('voxel_size').value
        if collision_thresh > 0:
            detector = ModelFreeCollisionDetector(cloud_masked, voxel_size=voxel_size)
            collision_mask = detector.detect(
                gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        gg = gg.nms().sort_by_score()
        top_k = self.get_parameter('top_k').value
        gg = gg[:top_k]
        self.get_logger().info(f'Visualizing {len(gg)} grasps (top {top_k}).')

        # --- draw grasps on BGR image ---
        vis = color_bgr.copy()
        translations     = gg.translations       # (M, 3)
        rotation_mats    = gg.rotation_matrices  # (M, 3, 3)
        scores           = gg.scores             # (M,)
        widths           = gg.widths             # (M,)
        depths           = gg.depths             # (M,)

        score_max = float(scores.max()) if len(scores) > 0 else 1.0
        for i in range(len(gg)):
            _draw_grasp(vis,
                        R=rotation_mats[i], t=translations[i],
                        width=widths[i], depth=depths[i],
                        score=scores[i], score_max=score_max,
                        fx=fx, fy=fy, cx=cx, cy=cy)

        self._publish_image(vis, rgb_msg.header)

    # ------------------------------------------------------------------
    def _publish_image(self, bgr_img, header):
        msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding='bgr8')
        msg.header = header
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GraspNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
