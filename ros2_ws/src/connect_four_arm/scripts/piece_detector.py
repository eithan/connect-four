#!/usr/bin/env python3
"""
piece_detector.py — Detect Connect Four pieces from the overhead RGBD camera.

HSV-segments red and yellow pieces in each RGB frame, back-projects their
centroids using depth, transforms to base_link via TF2, and publishes a JSON
string on /connect_four/piece_positions:
    {"red": [[x,y], ...], "yellow": [[x,y], ...]}

Positions are in base_link frame, sorted by y (ascending).
"""
import json
import numpy as np
import cv2
import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # registers PointStamped with the tf2 registry

# ── HSV thresholds (OpenCV: H 0-180, S 0-255, V 0-255) ─────────────────────
# Red Gazebo material: ambient R=0.8 G=0.1 B=0.1  → H≈0°,  S≈220, V≈200
RED_LO1 = np.array([  0, 130,  60], dtype=np.uint8)
RED_HI1 = np.array([ 12, 255, 255], dtype=np.uint8)
RED_LO2 = np.array([168, 130,  60], dtype=np.uint8)
RED_HI2 = np.array([180, 255, 255], dtype=np.uint8)
# Yellow Gazebo material: ambient R=0.9 G=0.8 B=0.1 → H≈25°, S≈220, V≈230
YEL_LO  = np.array([ 18, 100,  60], dtype=np.uint8)
YEL_HI  = np.array([ 42, 255, 255], dtype=np.uint8)

# Piece contour area filter (pixels²).  At 1.2 m depth, 60° FOV, 640×480:
#   fx ≈ 554 px/m  →  piece radius 16.5 mm → ~9 px radius → ~250 px² area.
MIN_AREA =  30
MAX_AREA = 1200


def _decode_image(msg: Image, dtype: np.dtype, channels: int) -> np.ndarray:
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    return arr.reshape(msg.height, msg.width) if channels == 1 \
        else arr.reshape(msg.height, msg.width, channels)


class PieceDetector(Node):
    def __init__(self):
        super().__init__('piece_detector')
        self._info: CameraInfo | None = None
        self._depth: np.ndarray | None = None

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        self.create_subscription(CameraInfo, '/overhead_camera/camera_info',  self._on_info,  10)
        self.create_subscription(Image,      '/overhead_camera/depth_image',  self._on_depth, 10)
        self.create_subscription(Image,      '/overhead_camera/image',        self._on_rgb,   10)
        self._pub = self.create_publisher(String, '/connect_four/piece_positions', 10)
        self.get_logger().info('PieceDetector started')

    def _on_info(self, msg: CameraInfo):
        self._info = msg

    def _on_depth(self, msg: Image):
        self._depth = _decode_image(msg, np.float32, 1)

    def _on_rgb(self, msg: Image):
        if self._info is None or self._depth is None:
            return

        bgr = _decode_image(msg, np.uint8, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        K = self._info.k  # row-major 3×3 intrinsic matrix
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        src_frame = self._info.header.frame_id or 'overhead_camera_link'

        result: dict[str, list] = {'red': [], 'yellow': []}
        masks = {
            'red':    cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                                     cv2.inRange(hsv, RED_LO2, RED_HI2)),
            'yellow': cv2.inRange(hsv, YEL_LO, YEL_HI),
        }
        kern = np.ones((3, 3), np.uint8)

        for color, mask in masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA < area < MAX_AREA):
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                u = int(M['m10'] / M['m00'])
                v = int(M['m01'] / M['m00'])

                # Median depth in a 3×3 patch around centroid
                h_img, w_img = self._depth.shape
                patch = self._depth[max(0, v-1):min(h_img, v+2),
                                    max(0, u-1):min(w_img, u+2)]
                valid = patch[np.isfinite(patch) & (patch > 0.05)]
                if valid.size == 0:
                    continue
                d = float(np.median(valid))

                # Back-project to camera optical frame
                x_c = (u - cx) * d / fx
                y_c = (v - cy) * d / fy
                z_c = d

                pt = PointStamped()
                pt.header.frame_id = src_frame
                pt.header.stamp    = msg.header.stamp
                pt.point.x = x_c
                pt.point.y = y_c
                pt.point.z = z_c

                try:
                    pt_bl = self._tf_buf.transform(
                        pt, 'base_link',
                        timeout=rclpy.duration.Duration(seconds=0.1),
                    )
                    result[color].append([
                        round(float(pt_bl.point.x), 4),
                        round(float(pt_bl.point.y), 4),
                    ])
                except Exception as exc:
                    self.get_logger().warn(
                        f'TF {src_frame}→base_link: {exc}',
                        throttle_duration_sec=5.0,
                    )

            result[color].sort(key=lambda p: p[1])

        self._pub.publish(String(data=json.dumps(result)))
        self.get_logger().debug(
            f'pieces red={len(result["red"])} yellow={len(result["yellow"])}',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PieceDetector()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
