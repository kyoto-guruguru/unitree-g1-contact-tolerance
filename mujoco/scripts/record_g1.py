"""Run the Unitree RL Gym MuJoCo deployment and record an MP4 video.

The upstream repository is imported but not modified.  The on-screen viewer
remains interactive, while the recorded camera follows the pelvis in the
horizontal plane so that the robot stays visible during long walking trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


# Public repository layout: <repo>/mujoco/scripts/record_g1.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "unitree_rl_gym"
CONFIG_DIR = UPSTREAM_ROOT / "deploy" / "deploy_mujoco" / "configs"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "results" / "videos"


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Convert a MuJoCo wxyz quaternion to roll, pitch, yaw in radians."""
    qw, qx, qy, qz = quaternion
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(float(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def get_contact_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_foot_body_id: int,
    right_foot_body_id: int,
    pelvis_body_id: int,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "left_contact": 0,
        "left_contact_count": 0,
        "left_normal_force_n": 0.0,
        "right_contact": 0,
        "right_contact_count": 0,
        "right_normal_force_n": 0.0,
        "nonfoot_ground_contact": 0,
        "pelvis_ground_contact": 0,
    }
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        bodies = (body1, body2)

        mujoco.mj_contactForce(model, data, contact_id, contact_force)
        normal_force = abs(float(contact_force[0]))

        if left_foot_body_id in bodies:
            result["left_contact"] = 1
            result["left_contact_count"] += 1
            result["left_normal_force_n"] += normal_force
        if right_foot_body_id in bodies:
            result["right_contact"] = 1
            result["right_contact_count"] += 1
            result["right_normal_force_n"] += normal_force

        # The ground plane belongs to the world body (body id 0).
        if body1 == 0 and body2 != 0:
            ground_body = body2
        elif body2 == 0 and body1 != 0:
            ground_body = body1
        else:
            ground_body = -1
        if ground_body >= 0 and ground_body not in (left_foot_body_id, right_foot_body_id):
            result["nonfoot_ground_contact"] = 1
        if ground_body == pelvis_body_id:
            result["pelvis_ground_contact"] = 1

    return result


def get_joint_layout(model: mujoco.MjModel) -> list[dict[str, int | float | str]]:
    layout: list[dict[str, int | float | str]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        layout.append(
            {
                "actuator_id": actuator_id,
                "joint_id": joint_id,
                "name": joint_name,
                "qpos_adr": int(model.jnt_qposadr[joint_id]),
                "dof_adr": int(model.jnt_dofadr[joint_id]),
                "torque_limit_low_nm": float(model.jnt_actfrcrange[joint_id, 0]),
                "torque_limit_high_nm": float(model.jnt_actfrcrange[joint_id, 1]),
            }
        )
    return layout


def make_csv_header(joint_layout: list[dict[str, int | float | str]]) -> list[str]:
    header = [
        "time_s",
        "physics_step",
        "cmd_vx_m_s",
        "cmd_vy_m_s",
        "cmd_yaw_rate_rad_s",
        "base_x_m",
        "base_y_m",
        "base_z_m",
        "base_qw",
        "base_qx",
        "base_qy",
        "base_qz",
        "base_roll_rad",
        "base_pitch_rad",
        "base_yaw_rad",
        "base_world_wx_rad_s",
        "base_world_wy_rad_s",
        "base_world_wz_rad_s",
        "base_world_vx_m_s",
        "base_world_vy_m_s",
        "base_world_vz_m_s",
        "base_body_wx_rad_s",
        "base_body_wy_rad_s",
        "base_body_wz_rad_s",
        "base_body_vx_m_s",
        "base_body_vy_m_s",
        "base_body_vz_m_s",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "left_foot_x_m",
        "left_foot_y_m",
        "left_foot_z_m",
        "left_contact",
        "left_contact_count",
        "left_normal_force_n",
        "right_foot_x_m",
        "right_foot_y_m",
        "right_foot_z_m",
        "right_contact",
        "right_contact_count",
        "right_normal_force_n",
        "ncon_total",
        "nonfoot_ground_contact",
        "pelvis_ground_contact",
    ]
    for joint in joint_layout:
        name = str(joint["name"])
        header.extend(
            [
                f"{name}_pos_rad",
                f"{name}_vel_rad_s",
                f"{name}_action",
                f"{name}_target_pos_rad",
                f"{name}_tau_requested_nm",
                f"{name}_actuator_force_nm",
                f"{name}_tau_applied_nm",
            ]
        )
    return header


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    candidate = CONFIG_DIR / value
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Configuration file not found: {value}")


def start_encoder(ffmpeg: str, output: Path, width: int, height: int, fps: int):
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_file",
        nargs="?",
        default="g1.yaml",
        help="YAML file name in the upstream config directory, or a full path",
    )
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--csv-output", type=Path, help="Output CSV path")
    parser.add_argument("--duration", type=float, help="Override duration in simulation seconds")
    parser.add_argument("--cmd-vx", type=float, help="Override forward velocity command [m/s]")
    parser.add_argument("--cmd-vy", type=float, help="Override lateral velocity command [m/s]")
    parser.add_argument("--cmd-yaw", type=float, help="Override yaw-rate command [rad/s]")
    parser.add_argument(
        "--log-decimation",
        type=int,
        default=1,
        help="Write one CSV row every N physics steps (default: 1)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=4.0)
    parser.add_argument("--camera-azimuth", type=float, default=-140.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("Video width and height must be positive even numbers")
    if args.fps <= 0:
        raise ValueError("FPS must be positive")
    if args.log_decimation <= 0:
        raise ValueError("Log decimation must be positive")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    config_path = resolve_config(args.config_file)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    root_token = "{LEGGED_GYM_ROOT_DIR}"
    policy_path = Path(config["policy_path"].replace(root_token, str(UPSTREAM_ROOT))).resolve()
    xml_path = Path(config["xml_path"].replace(root_token, str(UPSTREAM_ROOT))).resolve()
    simulation_duration = float(
        args.duration if args.duration is not None else config["simulation_duration"]
    )
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])

    kps = np.asarray(config["kps"], dtype=np.float32)
    kds = np.asarray(config["kds"], dtype=np.float32)
    default_angles = np.asarray(config["default_angles"], dtype=np.float32)
    ang_vel_scale = float(config["ang_vel_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    action_scale = float(config["action_scale"])
    cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    cmd = np.asarray(config["cmd_init"], dtype=np.float32)
    if args.cmd_vx is not None:
        cmd[0] = args.cmd_vx
    if args.cmd_vy is not None:
        cmd[1] = args.cmd_vy
    if args.cmd_yaw is not None:
        cmd[2] = args.cmd_yaw
    gait_period = 0.8

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (DEFAULT_VIDEO_DIR / f"{config_path.stem}_{timestamp}.mp4")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = (args.csv_output or output.with_suffix(".csv")).resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".json")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    # The upstream scene defaults to a 640-pixel offscreen framebuffer.
    # Enlarge it in memory for HD recording without modifying the XML model.
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()

    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_foot_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
    )
    right_foot_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
    )
    if min(pelvis_body_id, left_foot_body_id, right_foot_body_id) < 0:
        raise RuntimeError("Required G1 body names were not found in the MuJoCo model")
    joint_layout = get_joint_layout(model)
    if len(joint_layout) != num_actions:
        raise RuntimeError(
            f"Model has {len(joint_layout)} actuators but config expects {num_actions} actions"
        )

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    next_frame_time = 0.0
    frame_count = 0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    record_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(record_camera)
    record_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    record_camera.distance = args.camera_distance
    record_camera.azimuth = args.camera_azimuth
    record_camera.elevation = args.camera_elevation
    record_camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.7]

    encoder = start_encoder(ffmpeg, output, args.width, args.height, args.fps)
    if encoder.stdin is None:
        raise RuntimeError("Failed to open the ffmpeg input pipe")
    csv_file = csv_output.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(make_csv_header(joint_layout))
    logged_row_count = 0

    interrupted = False
    wall_start = time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = args.camera_distance
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
            viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.7]

            while viewer.is_running() and data.time < simulation_duration:
                step_start = time.time()
                target_used = target_dof_pos.copy()
                action_used = action.copy()
                tau = pd_control(
                    target_used,
                    data.qpos[7:],
                    kps,
                    np.zeros_like(kds),
                    data.qvel[6:],
                    kds,
                )
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)
                counter += 1

                if counter % control_decimation == 0:
                    qj = (data.qpos[7:] - default_angles) * dof_pos_scale
                    dqj = data.qvel[6:] * dof_vel_scale
                    quat = data.qpos[3:7]
                    omega = data.qvel[3:6] * ang_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)

                    phase = (counter * simulation_dt) % gait_period / gait_period
                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = [
                        np.sin(2 * np.pi * phase),
                        np.cos(2 * np.pi * phase),
                    ]
                    with torch.inference_mode():
                        action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
                    target_dof_pos = action * action_scale + default_angles

                if counter % args.log_decimation == 0:
                    world_velocity = np.zeros(6, dtype=np.float64)
                    body_velocity = np.zeros(6, dtype=np.float64)
                    mujoco.mj_objectVelocity(
                        model,
                        data,
                        mujoco.mjtObj.mjOBJ_BODY,
                        pelvis_body_id,
                        world_velocity,
                        0,
                    )
                    mujoco.mj_objectVelocity(
                        model,
                        data,
                        mujoco.mjtObj.mjOBJ_BODY,
                        pelvis_body_id,
                        body_velocity,
                        1,
                    )
                    roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])
                    contacts = get_contact_summary(
                        model,
                        data,
                        left_foot_body_id,
                        right_foot_body_id,
                        pelvis_body_id,
                    )
                    row: list[float | int] = [
                        float(data.time),
                        counter,
                        float(cmd[0]),
                        float(cmd[1]),
                        float(cmd[2]),
                        *data.qpos[0:3].tolist(),
                        *data.qpos[3:7].tolist(),
                        roll,
                        pitch,
                        yaw,
                        *world_velocity[0:3].tolist(),
                        *world_velocity[3:6].tolist(),
                        *body_velocity[0:3].tolist(),
                        *body_velocity[3:6].tolist(),
                        *data.subtree_com[pelvis_body_id].tolist(),
                        *data.xpos[left_foot_body_id].tolist(),
                        int(contacts["left_contact"]),
                        int(contacts["left_contact_count"]),
                        float(contacts["left_normal_force_n"]),
                        *data.xpos[right_foot_body_id].tolist(),
                        int(contacts["right_contact"]),
                        int(contacts["right_contact_count"]),
                        float(contacts["right_normal_force_n"]),
                        int(data.ncon),
                        int(contacts["nonfoot_ground_contact"]),
                        int(contacts["pelvis_ground_contact"]),
                    ]
                    for joint in joint_layout:
                        actuator_id = int(joint["actuator_id"])
                        qpos_adr = int(joint["qpos_adr"])
                        dof_adr = int(joint["dof_adr"])
                        row.extend(
                            [
                                float(data.qpos[qpos_adr]),
                                float(data.qvel[dof_adr]),
                                float(action_used[actuator_id]),
                                float(target_used[actuator_id]),
                                float(tau[actuator_id]),
                                float(data.actuator_force[actuator_id]),
                                float(data.qfrc_actuator[dof_adr]),
                            ]
                        )
                    csv_writer.writerow(row)
                    logged_row_count += 1

                if data.time + 1e-12 >= next_frame_time:
                    record_camera.lookat[0] = data.qpos[0]
                    record_camera.lookat[1] = data.qpos[1]
                    renderer.update_scene(data, camera=record_camera)
                    encoder.stdin.write(renderer.render().tobytes())
                    frame_count += 1
                    next_frame_time += 1.0 / args.fps

                viewer.sync()
                remaining = simulation_dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

            interrupted = data.time < simulation_duration
    finally:
        csv_file.close()
        renderer.close()
        try:
            encoder.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        return_code = encoder.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {encoder_stderr}")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "xml_path": str(xml_path),
        "config_command": config["cmd_init"],
        "command": cmd.tolist(),
        "gait_period_s": gait_period,
        "config": config,
        "requested_duration_s": simulation_duration,
        "recorded_simulation_duration_s": float(data.time),
        "wall_duration_s": time.time() - wall_start,
        "interrupted": interrupted,
        "fps": args.fps,
        "frame_count": frame_count,
        "csv_path": str(csv_output),
        "csv_log_decimation": args.log_decimation,
        "csv_sample_rate_hz": 1.0 / (simulation_dt * args.log_decimation),
        "csv_row_count": logged_row_count,
        "joint_layout": joint_layout,
        "width": args.width,
        "height": args.height,
        "camera_distance": args.camera_distance,
        "camera_azimuth": args.camera_azimuth,
        "camera_elevation": args.camera_elevation,
        "mujoco_version": mujoco.__version__,
        "torch_version": torch.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Video: {output}")
    print(f"CSV: {csv_output}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
