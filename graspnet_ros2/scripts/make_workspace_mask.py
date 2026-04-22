"""Interactive workspace mask creator for GraspNet ROS2.

Usage:
  python make_workspace_mask.py [--topic /camera_1/image] [--output workspace_mask.png]

Controls:
  Left click (1st)  — set first corner of rectangle
  Left click (2nd)  — set opposite corner and complete rectangle
  Right click       — reset rectangle
  Enter             — save mask and exit
  r                 — reset rectangle
  q                 — quit without saving
"""

import argparse
import os
import sys

_ros_distro = os.environ.get('ROS_DISTRO', 'humble')
_ros_site = f'/opt/ros/{_ros_distro}/local/lib/python3.10/dist-packages'
if _ros_site not in sys.path:
    sys.path.insert(0, _ros_site)

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _imgmsg_to_bgr(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == 'bgr8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    elif enc == 'rgb8':
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1].copy()
    elif enc in ('bgra8', 'rgba8'):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        bgr = arr[:, :, :3]
        if enc == 'rgba8':
            bgr = bgr[:, :, ::-1]
        return bgr.copy()
    else:
        raise ValueError(f'Unsupported encoding for mask tool: {msg.encoding}')


class MaskCreator(Node):
    def __init__(self, topic: str, output: str):
        super().__init__('mask_creator')
        self._output = output
        self._latest = None
        self._corners = []      # 0, 1, or 2 (x, y) tuples
        self._mouse_pos = None  # live cursor for preview
        self._done = False
        self._saved = False

        self.create_subscription(Image, topic, self._cb, 10)
        self.get_logger().info(f'Subscribed to {topic} — waiting for first frame...')

    def _cb(self, msg: Image):
        try:
            self._latest = _imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(str(e))

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(self._corners) < 2:
                self._corners.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._corners.clear()

    def _rect_from_corners(self, p1, p2):
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        return (x1, y1), (x2, y2)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()

        # Determine current rectangle to preview
        if len(self._corners) == 2:
            p1, p2 = self._corners
        elif len(self._corners) == 1 and self._mouse_pos:
            p1, p2 = self._corners[0], self._mouse_pos
        else:
            p1, p2 = None, None

        if p1 and p2:
            tl, br = self._rect_from_corners(p1, p2)
            fill = overlay.copy()
            cv2.rectangle(fill, tl, br, (0, 255, 0), -1)
            cv2.addWeighted(fill, 0.35, overlay, 0.65, 0, overlay)
            thickness = 2 if len(self._corners) < 2 else 3
            color = (0, 200, 255) if len(self._corners) < 2 else (0, 255, 255)
            cv2.rectangle(overlay, tl, br, color, thickness)
            # size label
            w_px = br[0] - tl[0]
            h_px = br[1] - tl[1]
            cv2.putText(overlay, f'{w_px}x{h_px}px',
                        (tl[0] + 4, tl[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, f'{w_px}x{h_px}px',
                        (tl[0] + 4, tl[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Corner dots
        for pt in self._corners:
            cv2.circle(overlay, pt, 6, (0, 255, 255), -1)
            cv2.circle(overlay, pt, 6, (0, 0, 0), 1)

        # HUD
        if len(self._corners) == 0:
            status = 'Step 1: click first corner'
        elif len(self._corners) == 1:
            status = 'Step 2: click opposite corner'
        else:
            status = 'Rectangle set — press Enter to save'

        lines = [
            status,
            'Right click / r: reset',
            'Enter: save  |  q: quit',
        ]
        for i, text in enumerate(lines):
            y_pos = 24 + i * 22
            cv2.putText(overlay, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return overlay

    def _save_mask(self, H: int, W: int):
        mask = np.zeros((H, W), dtype=np.uint8)
        tl, br = self._rect_from_corners(*self._corners)
        cv2.rectangle(mask, tl, br, 255, -1)
        cv2.imwrite(self._output, mask)
        white_px = int((mask > 0).sum())
        self.get_logger().info(
            f'Saved workspace mask → {self._output}  '
            f'rect=({tl[0]},{tl[1]})-({br[0]},{br[1]})  '
            f'{white_px} white px / {H * W} total  '
            f'({100.0 * white_px / (H * W):.1f}%)')
        self._saved = True

    def run(self):
        WIN = 'Workspace Mask Creator'
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._mouse_cb)

        print('Waiting for first camera frame…')
        while rclpy.ok() and self._latest is None:
            rclpy.spin_once(self, timeout_sec=0.05)

        print(f'Frame received ({self._latest.shape[1]}×{self._latest.shape[0]}). '
              'Click two opposite corners to define the workspace rectangle.')

        while rclpy.ok() and not self._done:
            rclpy.spin_once(self, timeout_sec=0.01)

            if self._latest is None:
                continue

            display = self._draw_overlay(self._latest)
            cv2.imshow(WIN, display)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, ord('\r')):   # Enter
                if len(self._corners) < 2:
                    print('Need 2 corners — click two opposite corners first.')
                else:
                    H, W = self._latest.shape[:2]
                    self._save_mask(H, W)
                    self._done = True
            elif key == ord('r'):
                self._corners.clear()
                print('Rectangle reset.')
            elif key == ord('q'):
                print('Quit without saving.')
                self._done = True

        cv2.destroyAllWindows()
        return self._saved


def main():
    parser = argparse.ArgumentParser(description='Interactive workspace mask creator (rectangle)')
    parser.add_argument('--topic',  default='/camera_1/image',
                        help='RGB image topic (default: /camera_1/image)')
    parser.add_argument('--output', default='workspace_mask.png',
                        help='Output PNG path (default: workspace_mask.png)')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args or None)
    node = MaskCreator(args.topic, args.output)
    saved = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if saved else 1)


if __name__ == '__main__':
    main()
