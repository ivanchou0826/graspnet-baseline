"""GraspNet ROS2 node — on-demand inference via /graspnet/trigger service.

Camera topics are cached continuously; inference runs only when the service
is called, reducing CPU/GPU load and eliminating visualization jitter.

Optional: subscribe to /detections_output (vision_msgs/Detection2DArray) from
upstream detectors (YOLOv8, RT-DETR, etc.) to restrict point cloud to ROI
bounding boxes before running GraspNet inference.
"""

import os
import sys

_ros_distro = os.environ.get('ROS_DISTRO', 'humble')
_ros_site = f'/opt/ros/{_ros_distro}/local/lib/python3.10/dist-packages'
if _ros_site not in sys.path:
    sys.path.insert(0, _ros_site)

GRASPNET_ROOT = os.environ.get('GRASPNET_ROOT', '')
if not GRASPNET_ROOT:
    raise RuntimeError(
        'Environment variable GRASPNET_ROOT is not set. '
        'Point it to the graspnet-baseline repository root, e.g.:\n'
        '  export GRASPNET_ROOT=/path/to/graspnet-baseline\n'
        'or use the provided launch file which sets it automatically.')

for _sub in ('models', 'utils', 'dataset'):
    _p = os.path.join(GRASPNET_ROOT, _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection2DArray

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspCameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


def _imgmsg_to_numpy(msg: Image):
    enc = msg.encoding.lower()
    if enc in ('16uc1', '16sc1'):
        dtype = np.uint16 if enc == '16uc1' else np.int16
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    elif enc in ('32fc1',):
        img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    elif enc in ('rgb8', 'bgr8', 'bgra8', 'rgba8'):
        channels = 4 if enc in ('bgra8', 'rgba8') else 3
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        if enc == 'rgb8':
            img = img[:, :, ::-1].copy()
    else:
        raise ValueError(f'Unsupported encoding: {msg.encoding}')
    return img


def _numpy_to_imgmsg(img: np.ndarray, encoding: str, header=None) -> Image:
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height = img.shape[0]
    msg.width  = img.shape[1]
    msg.encoding = encoding
    msg.step   = int(img.strides[0])
    msg.data   = img.tobytes()
    return msg


def _score_to_bgr(score: float, score_max: float = 1.2):
    t = float(np.clip(score / score_max, 0.0, 1.0))
    r = int((1.0 - t) * 255)
    g = int(t * 255)
    return (0, g, r)


def _project(pt3d, fx, fy, cx, cy):
    x, y, z = pt3d
    if z <= 1e-4:
        return None
    return (int(fx * x / z + cx), int(fy * y / z + cy))


def _draw_grasp(img, R, t, width, depth, score, fx, fy, cx, cy, score_max=1.2):
    half_w = width / 2.0
    corners_local = np.array([
        [depth,  -half_w, 0.0],
        [depth,   half_w, 0.0],
        [0.0,     half_w, 0.0],
        [0.0,    -half_w, 0.0],
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

    base_mid = ((pixels[2][0] + pixels[3][0]) // 2,
                (pixels[2][1] + pixels[3][1]) // 2)
    tip_mid  = ((pixels[0][0] + pixels[1][0]) // 2,
                (pixels[0][1] + pixels[1][1]) // 2)
    cv2.arrowedLine(img, base_mid, tip_mid, color, 1, tipLength=0.3)

    label = f'{score:.2f}'
    lx, ly = tip_mid[0] + 4, tip_mid[1] - 4
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,    1, cv2.LINE_AA)


def _build_pointcloud2(cloud_xyz: np.ndarray, cloud_rgb_f: np.ndarray, header) -> PointCloud2:
    """Build XYZRGB PointCloud2 from Nx3 float32 xyz and Nx3 float32 rgb [0,1]."""
    n = len(cloud_xyz)
    rgb_uint8 = (cloud_rgb_f * 255).clip(0, 255).astype(np.uint8)
    rgb_packed = (rgb_uint8[:, 0].astype(np.uint32) << 16 |
                  rgb_uint8[:, 1].astype(np.uint32) << 8  |
                  rgb_uint8[:, 2].astype(np.uint32))
    rgb_as_float = rgb_packed.view(np.float32)

    data = np.zeros((n, 4), dtype=np.float32)
    data[:, :3] = cloud_xyz.astype(np.float32)
    data[:, 3]  = rgb_as_float

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width  = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step   = 16 * n
    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data.tobytes()
    return msg


def _make_gripper_marker(marker_id, R, t, width, depth, score, score_max, header):
    """Build a LINE_LIST Marker showing the gripper shape in 3D."""
    half_w = width / 2.0
    pts_local = np.array([
        [0.0,     -half_w, 0.0],   # left  base
        [depth,   -half_w, 0.0],   # left  tip
        [0.0,      half_w, 0.0],   # right base
        [depth,    half_w, 0.0],   # right tip
        [0.0,     -half_w, 0.0],   # palm bar start
        [0.0,      half_w, 0.0],   # palm bar end
        [-0.04,    0.0,    0.0],   # wrist back
        [0.0,      0.0,    0.0],   # wrist front
    ], dtype=np.float32)

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    m = Marker()
    m.header = header
    m.ns = 'graspnet'
    m.id = marker_id
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.004

    frac = float(np.clip(score / score_max, 0.0, 1.0))
    m.color.r = float(1.0 - frac)
    m.color.g = float(frac)
    m.color.b = 0.0
    m.color.a = 1.0

    m.lifetime.sec = 0
    m.lifetime.nanosec = 0

    for i, j in pairs:
        for idx in (i, j):
            world = t + R @ pts_local[idx]
            p = Point()
            p.x, p.y, p.z = float(world[0]), float(world[1]), float(world[2])
            m.points.append(p)

    return m


def _make_score_text_marker(marker_id, t, score, header):
    """Floating score label above the grasp centre."""
    m = Marker()
    m.header = header
    m.ns = 'graspnet_text'
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(t[0])
    m.pose.position.y = float(t[1])
    m.pose.position.z = float(t[2]) + 0.04
    m.pose.orientation.w = 1.0
    m.scale.z = 0.025
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 1.0
    m.text = f'{score:.2f}'
    m.lifetime.sec = 0
    m.lifetime.nanosec = 0
    return m


def _rotation_to_quaternion(R):
    """Convert 3x3 rotation matrix to (x, y, z, w) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class GraspNetNode(Node):
    def __init__(self):
        super().__init__('graspnet_node')

        # ---- parameters ----
        self.declare_parameter('checkpoint_path', os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar'))
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        self.declare_parameter('collision_thresh', 0.01)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('top_k', 50)
        self.declare_parameter('max_depth', 2.0)
        self.declare_parameter('remove_plane', True)
        self.declare_parameter('plane_dist_thresh', 0.01)
        # detection ROI parameters
        self.declare_parameter('det_input_width',  0)    # 0 = same resolution as depth image
        self.declare_parameter('det_input_height', 0)
        self.declare_parameter('det_score_thresh', 0.5)
        self.declare_parameter('det_class_filter', '')   # comma-separated; empty = all classes
        self.declare_parameter('det_timeout_sec',  3.0)  # ignore stale detections

        ckpt   = self.get_parameter('checkpoint_path').value
        n_view = self.get_parameter('num_view').value

        # ---- model ----
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = GraspNet(input_feature_dim=0, num_view=n_view, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02,
                       hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        net.to(self.device)
        checkpoint = torch.load(ckpt, map_location=self.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.get_logger().info(f"Loaded checkpoint '{ckpt}' (epoch {checkpoint['epoch']})")
        net.eval()
        self.net = net

        # ---- cached frames (updated by subscribers, consumed by service) ----
        self._latest_rgb   = None
        self._latest_depth = None
        self._latest_info  = None

        # ---- detection cache (optional, from YOLOv8 / RT-DETR etc.) ----
        self._latest_detections       = None   # Detection2DArray msg
        self._latest_detections_stamp = None   # rclpy.time.Time when received

        # ---- synchronized camera subscriptions ----
        qos = rclpy.qos.QoSProfile(depth=10)
        sub_rgb   = message_filters.Subscriber(self, Image,       '/camera_2/image', qos_profile=qos)
        sub_depth = message_filters.Subscriber(self, Image,       '/camera_2/depth', qos_profile=qos)
        sub_info  = message_filters.Subscriber(self, CameraInfo,  '/camera_2/info',  qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info], queue_size=10, slop=0.1)
        self.sync.registerCallback(self._cache_frame)

        # ---- optional detection subscription (independent of camera sync) ----
        self.create_subscription(
            Detection2DArray, '/detections_output', self._cache_detections, 10)

        # ---- publishers ----
        self.pub_vis     = self.create_publisher(Image,        '/graspnet/visualization', 10)
        self.pub_grasp   = self.create_publisher(PoseStamped, '/graspnet/best_grasp',    10)
        self.pub_markers = self.create_publisher(MarkerArray, '/graspnet/markers',       10)
        self.pub_cloud   = self.create_publisher(PointCloud2, '/graspnet/pointcloud',    10)

        # ---- trigger service ----
        self.srv = self.create_service(Trigger, '/graspnet/trigger', self._trigger_cb)

        self.get_logger().info('GraspNet node ready — call /graspnet/trigger to run inference.')
        self.get_logger().info('rviz2: add MarkerArray display on /graspnet/markers')
        self.get_logger().info(
            'Optional: publish vision_msgs/Detection2DArray to /detections_output '
            'to restrict inference to detected object ROIs.')

    # ------------------------------------------------------------------
    def _cache_frame(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self._latest_rgb   = rgb_msg
        self._latest_depth = depth_msg
        self._latest_info  = info_msg

    # ------------------------------------------------------------------
    def _cache_detections(self, msg: Detection2DArray):
        self._latest_detections       = msg
        self._latest_detections_stamp = self.get_clock().now()

    # ------------------------------------------------------------------
    def _get_active_roi(self, img_H: int, img_W: int):
        """Return (detections_msg, roi_mask) if a fresh detection cache exists, else (None, None).

        roi_mask is a bool HxW array in depth-image pixel space (True = keep point).
        Bboxes are scaled from the detector's input resolution to depth image resolution.
        """
        if self._latest_detections is None:
            return None, None

        age = (self.get_clock().now() - self._latest_detections_stamp).nanoseconds / 1e9
        timeout = self.get_parameter('det_timeout_sec').value
        if age > timeout:
            self.get_logger().warn(
                f'Detection cache is {age:.1f}s old (timeout={timeout}s) — ignoring ROI.')
            return None, None

        det_w = self.get_parameter('det_input_width').value
        det_h = self.get_parameter('det_input_height').value
        scale_x = img_W / det_w if det_w > 0 else 1.0
        scale_y = img_H / det_h if det_h > 0 else 1.0

        score_thresh = self.get_parameter('det_score_thresh').value
        class_filter_str = self.get_parameter('det_class_filter').value
        class_filter = (set(c.strip() for c in class_filter_str.split(',') if c.strip())
                        if class_filter_str else set())

        roi_mask = np.zeros((img_H, img_W), dtype=bool)
        n_accepted = 0

        for det in self._latest_detections.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda r: r.hypothesis.score)
            if best.hypothesis.score < score_thresh:
                continue
            if class_filter and best.hypothesis.class_id not in class_filter:
                continue

            cx = det.bbox.center.position.x * scale_x
            cy = det.bbox.center.position.y * scale_y
            sw = det.bbox.size_x * scale_x
            sh = det.bbox.size_y * scale_y

            x1 = int(np.clip(cx - sw / 2, 0, img_W - 1))
            y1 = int(np.clip(cy - sh / 2, 0, img_H - 1))
            x2 = int(np.clip(cx + sw / 2, 0, img_W))
            y2 = int(np.clip(cy + sh / 2, 0, img_H))

            if x2 > x1 and y2 > y1:
                roi_mask[y1:y2, x1:x2] = True
                n_accepted += 1

        if n_accepted == 0:
            self.get_logger().warn('Detections received but none passed score/class filter.')
            return self._latest_detections, None

        return self._latest_detections, roi_mask

    # ------------------------------------------------------------------
    def _draw_detection_boxes(self, vis_img: np.ndarray, detections_msg: Detection2DArray,
                               img_H: int, img_W: int):
        """Overlay scaled detection bounding boxes on vis_img in-place (yellow)."""
        det_w = self.get_parameter('det_input_width').value
        det_h = self.get_parameter('det_input_height').value
        scale_x = img_W / det_w if det_w > 0 else 1.0
        scale_y = img_H / det_h if det_h > 0 else 1.0

        score_thresh = self.get_parameter('det_score_thresh').value

        for det in detections_msg.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda r: r.hypothesis.score)
            if best.hypothesis.score < score_thresh:
                continue

            cx = det.bbox.center.position.x * scale_x
            cy = det.bbox.center.position.y * scale_y
            sw = det.bbox.size_x * scale_x
            sh = det.bbox.size_y * scale_y

            x1 = int(np.clip(cx - sw / 2, 0, img_W - 1))
            y1 = int(np.clip(cy - sh / 2, 0, img_H - 1))
            x2 = int(np.clip(cx + sw / 2, 0, img_W))
            y2 = int(np.clip(cy + sh / 2, 0, img_H))

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            label = f'{best.hypothesis.class_id} {best.hypothesis.score:.2f}'
            lx, ly = x1, max(y1 - 6, 12)
            cv2.putText(vis_img, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_img, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _trigger_cb(self, request, response):
        if self._latest_rgb is None:
            response.success = False
            response.message = 'No camera frame received yet — check that camera topics are publishing.'
            return response

        try:
            result_msg = self._run_inference(
                self._latest_rgb, self._latest_depth, self._latest_info)
            response.success = True
            response.message = result_msg
        except Exception as exc:
            self.get_logger().error(f'Inference failed: {exc}')
            response.success = False
            response.message = str(exc)

        return response

    # ------------------------------------------------------------------
    def _run_inference(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> str:
        color_bgr = _imgmsg_to_numpy(rgb_msg)
        color_rgb = color_bgr[:, :, ::-1].copy()

        enc = depth_msg.encoding.lower()
        depth_raw = _imgmsg_to_numpy(depth_msg)
        factor_depth = 1000.0 if '16' in enc else 1.0

        K  = info_msg.k
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        H, W   = depth_raw.shape[:2]

        cam = GraspCameraInfo(float(W), float(H), fx, fy, cx, cy, factor_depth)
        cloud_org = create_point_cloud_from_depth_image(depth_raw, cam, organized=True)

        max_depth = self.get_parameter('max_depth').value
        mask = (depth_raw > 0) & (depth_raw < max_depth)

        # Apply detection ROI if available
        active_detections, roi_mask = self._get_active_roi(H, W)
        if roi_mask is not None:
            mask = mask & roi_mask
            self.get_logger().info(
                f'Detection ROI applied: {int(roi_mask.sum())} ROI pixels from '
                f'{len(active_detections.detections)} detection(s).')

        cloud_masked = cloud_org[mask]
        color_masked = (color_rgb / 255.0)[mask]

        self.get_logger().info(f'Valid points after depth filter: {mask.sum()} / {mask.size}')
        if len(cloud_masked) == 0:
            return f'No points within max_depth={max_depth}m — try increasing it.'

        if self.get_parameter('remove_plane').value:
            import open3d as o3d
            dist_thresh = self.get_parameter('plane_dist_thresh').value
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_masked)
            _, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                           ransac_n=3, num_iterations=100)
            keep = np.ones(len(cloud_masked), dtype=bool)
            keep[inliers] = False
            cloud_masked = cloud_masked[keep]
            color_masked = color_masked[keep]
            self.get_logger().info(
                f'After plane removal: {keep.sum()} points (removed {len(inliers)} plane inliers)')
            if len(cloud_masked) < 100:
                return 'Too few points after plane removal — skipping frame.'

        # Publish colored object point cloud (full resolution, plane removed)
        self.pub_cloud.publish(_build_pointcloud2(cloud_masked, color_masked, rgb_msg.header))

        num_point = self.get_parameter('num_point').value
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
        else:
            idxs = np.concatenate([
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True),
            ])
        cloud_sampled = cloud_masked[idxs].astype(np.float32)

        end_points = {
            'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis]).to(self.device)
        }

        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        if len(gg_array) == 0:
            vis = color_bgr.copy()
            if active_detections is not None:
                self._draw_detection_boxes(vis, active_detections, H, W)
            self._pub_vis(vis, rgb_msg.header)
            return 'No grasps predicted.'
        gg = GraspGroup(gg_array)

        collision_thresh = self.get_parameter('collision_thresh').value
        voxel_size       = self.get_parameter('voxel_size').value
        if collision_thresh > 0:
            detector = ModelFreeCollisionDetector(cloud_masked, voxel_size=voxel_size)
            collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        gg = gg.nms().sort_by_score()
        top_k = self.get_parameter('top_k').value
        gg = gg[:top_k]

        from sklearn.cluster import DBSCAN
        translations  = gg.translations
        rotations     = gg.rotation_matrices
        scores        = gg.scores
        widths        = gg.widths
        depths        = gg.depths

        if len(translations) > 0:
            labels = DBSCAN(eps=0.08, min_samples=1).fit_predict(translations)
            best_idxs = []
            for lbl in np.unique(labels):
                cluster_idxs = np.where(labels == lbl)[0]
                best = cluster_idxs[np.argmax(scores[cluster_idxs])]
                best_idxs.append(best)
            best_idxs = np.array(best_idxs)
            translations = translations[best_idxs]
            rotations    = rotations[best_idxs]
            scores       = scores[best_idxs]
            widths       = widths[best_idxs]
            depths       = depths[best_idxs]

        n_grasps = len(translations)
        self.get_logger().info(f'Found {n_grasps} grasps (1 per object cluster).')
        score_max = float(scores.max()) if n_grasps > 0 else 1.0

        # Publish visualization: detection boxes (yellow) underneath grasp overlays
        vis = color_bgr.copy()
        if active_detections is not None:
            self._draw_detection_boxes(vis, active_detections, H, W)
        for i in range(n_grasps):
            _draw_grasp(vis,
                        R=rotations[i], t=translations[i],
                        width=widths[i], depth=depths[i],
                        score=scores[i], score_max=score_max,
                        fx=fx, fy=fy, cx=cx, cy=cy)
        self._pub_vis(vis, rgb_msg.header)

        # Publish 3D gripper markers for rviz2
        ma = MarkerArray()
        del_marker = Marker()
        del_marker.header = rgb_msg.header
        del_marker.ns = 'graspnet'
        del_marker.action = Marker.DELETEALL
        ma.markers.append(del_marker)
        for i in range(n_grasps):
            ma.markers.append(_make_gripper_marker(
                i, rotations[i], translations[i],
                widths[i], depths[i], scores[i], score_max, rgb_msg.header))
            ma.markers.append(_make_score_text_marker(
                i, translations[i], scores[i], rgb_msg.header))
        self.pub_markers.publish(ma)

        # Publish best grasp as PoseStamped (camera frame)
        if n_grasps > 0:
            best_idx = int(np.argmax(scores))
            pose_msg = PoseStamped()
            pose_msg.header = rgb_msg.header
            t = translations[best_idx]
            pose_msg.pose.position.x = float(t[0])
            pose_msg.pose.position.y = float(t[1])
            pose_msg.pose.position.z = float(t[2])
            qx, qy, qz, qw = _rotation_to_quaternion(rotations[best_idx])
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            self.pub_grasp.publish(pose_msg)
            self.get_logger().info(
                f'Best grasp: score={scores[best_idx]:.3f} pos=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})')

        return f'Done: {n_grasps} grasps found, best score={score_max:.3f}'

    def _pub_vis(self, bgr_img, header):
        msg = _numpy_to_imgmsg(bgr_img, encoding='bgr8', header=header)
        self.pub_vis.publish(msg)


def main(args=None):
    rclpy.init(args=args)
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
