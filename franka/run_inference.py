#!/usr/bin/env python3

# inspired from https://github.com/hca-lab-UofAlberta/pi0-franka-robot/blob/main/inference_pi.py

import socket
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from openpi.training import config
from openpi.policies import policy_config

CHECKPOINT_PATH = "gs://openpi-assets/checkpoints/pi0_fast_droid"

# UDP CONFIG

ROBOT_IP = "192.168.1.161"      # machine running udp_to_franky_ros (docker)
ACTION_PORT = 9090              # robot receives <8d>
STATE_PORT = 9091               # this PC receives <14d>

ACTION_FMT = "<8d"
STATE_FMT = "<14d"
STATE_SIZE = struct.calcsize(STATE_FMT)

action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
state_sock.bind(("0.0.0.0", STATE_PORT))
state_sock.setblocking(False)

last_robot_state = np.zeros(14, dtype=np.float64)


# IMAGE PREPROCESSING
def preprocess_image(image_pil: Image.Image, crop_scale: float = 0.9, out_size=(224, 224)) -> Image.Image:
    img = np.array(image_pil)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image with shape HxWx3")

    H, W = img.shape[:2]
    s = float(crop_scale) ** 0.5

    crop_h = int(round(H * s))
    crop_w = int(round(W * s))

    y0 = max((H - crop_h) // 2, 0)
    x0 = max((W - crop_w) // 2, 0)

    img_c = img[y0:y0 + crop_h, x0:x0 + crop_w]
    img_r = cv2.resize(img_c, out_size, interpolation=cv2.INTER_AREA)

    return Image.fromarray(img_r, mode="RGB")


# UDP STATE RECEIVE
def poll_robot_state_nonblocking():
    global last_robot_state

    while True:
        try:
            data, _ = state_sock.recvfrom(2048)
        except BlockingIOError:
            break

        if len(data) >= STATE_SIZE:
            last_robot_state = np.array(
                struct.unpack(STATE_FMT, data[:STATE_SIZE]),
                dtype=np.float64,
            )

    return last_robot_state

# WRIST CAMERA (Realsense)
def start_realsense_wrist(serial: str | None = None):
    pipeline = rs.pipeline()
    cfg = rs.config()

    if serial is not None:
        cfg.enable_device(serial)

    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)

    return pipeline

def get_realsense_rgb(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    bgr = np.asanyarray(color_frame.get_data())
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb

# LEFT CAMERA (logitech C920)
def start_left_camera(device_index=0):
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open left camera at index {device_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap

def get_left_camera_rgb(cap):
    ok, frame = cap.read()

    if not ok:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return rgb


# MAIN

def main():
    # Camera setup

    # Set this if multiple RealSense devices are connected.
    WRIST_REALSENSE_SERIAL = None

    # Change this if /dev/video0 is not your left camera.
    LEFT_CAMERA_INDEX = 0

    wrist_pipe = start_realsense_wrist(WRIST_REALSENSE_SERIAL)
    left_cap = start_left_camera(LEFT_CAMERA_INDEX)

    print("[camera] Started RealSense wrist camera")
    print(f"[camera] Started left camera on index {LEFT_CAMERA_INDEX}")

    # ----------------------------
    # Policy setup
    # ----------------------------

    cfg = config.get_config("pi0_fast_droid")
    policy = policy_config.create_trained_policy(cfg, CHECKPOINT_PATH)

    print("[policy] Loaded OpenPI policy")

    # Loop config

    send_interval = 0.2  # 5 Hz
    last_send_time = 0.0

    prompt = "pick up the banana and place it on the white plate"

    try:
        while True:
            # Get camera frames

            wrist_rgb = get_realsense_rgb(wrist_pipe)
            left_rgb = get_left_camera_rgb(left_cap)

            if wrist_rgb is None or left_rgb is None:
                continue

            wrist_image = preprocess_image(Image.fromarray(wrist_rgb))
            front_image = preprocess_image(Image.fromarray(left_rgb))

            # Display
            wrist_disp = cv2.cvtColor(np.array(wrist_image), cv2.COLOR_RGB2BGR)
            front_disp = cv2.cvtColor(np.array(front_image), cv2.COLOR_RGB2BGR)

            cv2.imshow("Wrist RealSense Processed", wrist_disp)
            cv2.imshow("Left Camera Processed", front_disp)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting...")
                break
            # Policy + UDP action send

            now = time.time()

            if now - last_send_time >= send_interval:
                cur_robot_state = poll_robot_state_nonblocking().astype(np.float32)

                print("[state]", np.round(cur_robot_state, 3))

                example = {
                    "observation/exterior_image_1_left": front_image,
                    "observation/wrist_image_left": wrist_image,
                    "observation/joint_position": cur_robot_state[7:14],
                    "observation/gripper_position": np.array([cur_robot_state[6]], dtype=np.float32),
                    "prompt": prompt,
                }

                action_chunk = policy.infer(example)["actions"]
                action_prediction = np.asarray(action_chunk[0], dtype=np.float64)

                if action_prediction.shape[0] != 8:
                    raise RuntimeError(
                        f"Expected action shape (8,), got {action_prediction.shape}"
                    )

                # commenting out action sending
                packet = struct.pack(ACTION_FMT, *action_prediction)
                action_sock.sendto(packet, (ROBOT_IP, ACTION_PORT))

                print("[action]", np.round(action_prediction, 4))

                last_send_time = now

    finally:
        wrist_pipe.stop()
        left_cap.release()
        cv2.destroyAllWindows()
        action_sock.close()
        state_sock.close()

if __name__ == "__main__":
    main()