#!/usr/bin/env bash
# =============================================================
#  teleop.sh
#  Teleoperate the SO-101 (leader drives follower) with both cameras
#  shown in the Rerun viewer. Use to verify hardware and to teach poses.
#
#  Uses the stable udev symlinks so the arms never swap on reboot:
#    /dev/so101_follower -> ttyACM1
#    /dev/so101_leader   -> ttyACM0
#
#  Camera key `front` (was `laptop`) matches record_first_dataset.sh and
#  dataset_schema.md. Keep these keys identical everywhere — at record
#  time they become observation.images.<key> and the policy keys off them.
# =============================================================

set -euo pipefail

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=my_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --robot.cameras='{
    front: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"},
    hand: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "YUYV", backend: "V4L2"}
  }' \
  --display_port=9090
