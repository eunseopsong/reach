# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_conjugate

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm

# 로봇: UR10e + spindle (사용자 USD를 참조하는 ArticulationCfg 객체)
from isaaclab_assets import UR10E_W_SPINDLE_CFG


# =============================================================================
# TXT (x y z r p y [rad]) -> CommandManager 갱신 함수
#  - Event(mode="step")에서 호출되어 매 프레임 목표 포즈를 업데이트
#  - asset_cfg 인자 없이, env.cfg.commands.ee_pose.body_name(=EE 링크명) 사용
# =============================================================================

def _rpy_to_quat_torch(rpy: torch.Tensor) -> torch.Tensor:
    """RPY(rad) -> quat(w,x,y,z)"""
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = torch.cos(r * 0.5), torch.sin(r * 0.5)
    cp, sp = torch.cos(p * 0.5), torch.sin(p * 0.5)
    cy, sy = torch.cos(y * 0.5), torch.sin(y * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=-1)

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply unit quaternion q(wxyz) to vector v."""
    w, x, y, z = q.unbind(-1)
    qv = torch.stack([torch.zeros_like(w), v[:, 0], v[:, 1], v[:, 2]], dim=-1)
    qc = torch.stack([w, -x, -y, -z], dim=-1)
    return (quat_mul(quat_mul(q, qv), qc))[:, 1:]

def _world_to_body(pos_w: torch.Tensor, quat_w: torch.Tensor,
                   root_pos_w: torch.Tensor, root_quat_w: torch.Tensor):
    """World -> base(body) 변환."""
    q_inv = quat_conjugate(root_quat_w)
    p_rel = pos_w - root_pos_w
    pos_b = _quat_apply(q_inv, p_rel)
    quat_b = quat_mul(q_inv, quat_w)
    return pos_b, quat_b

def txt_pose_command(env, command_name: str, file_path: str,
                     coordinate: str = "world", lookahead: int = 20, loop: bool = True):
    """
    TXT 궤적을 읽어 목표 포즈를 CommandManager에 기록.
    - TXT 형식: 행마다 [x y z r p y] (rad), World 좌표 기준 가정
    - 현재 EE에서 가장 가까운 웨이포인트 + lookahead 를 목표로 설정
    """
    dev = env.device

    # (1) 경로 캐시 로드 1회
    if not hasattr(env, "_txt_traj_loaded"):
        data = np.loadtxt(file_path, dtype=np.float64)  # (M,6)
        env._traj_pos_w = torch.tensor(data[:, 0:3], device=dev, dtype=torch.float32)               # [M,3]
        env._traj_quat_w = _rpy_to_quat_torch(torch.tensor(data[:, 3:6], device=dev, dtype=torch.float32))  # [M,4]
        env._traj_M = env._traj_pos_w.shape[0]
        env._txt_traj_loaded = True

    # (2) 로봇/EE 링크 식별 (env.cfg에서 이름을 읽음)
    robot = env.scene["robot"]
    ee_name = env.cfg.commands.ee_pose.body_name  # 예: "spindle_link"
    body_id = robot.data.body_names.index(ee_name)

    # (3) 현재 EE(World)
    ee_pos_w = robot.data.body_pos_w[:, body_id]  # [N,3]

    # (4) 최근접 웨이포인트 + lookahead
    dists = torch.cdist(ee_pos_w, env._traj_pos_w)   # [N,M]
    nearest = torch.argmin(dists, dim=1)
    idx = torch.clamp(nearest + lookahead, 0, env._traj_M - 1)

    des_pos_w = env._traj_pos_w[idx]     # [N,3]
    des_quat_w = env._traj_quat_w[idx]   # [N,4]

    # (5) World -> base(body) 변환 후 command 설정
    if coordinate == "world":
        des_pos_b, des_quat_b = _world_to_body(
            des_pos_w, des_quat_w, robot.data.root_pos_w, robot.data.root_quat_w
        )
    else:
        des_pos_b, des_quat_b = des_pos_w, des_quat_w

    cmd = torch.cat([des_pos_b, des_quat_b], dim=-1)  # [N,7]
    env.command_manager.set_command(command_name, cmd)
    return cmd


# =============================================================================
# 환경 설정
# =============================================================================

@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    """UR10e(+spindle) Reach 환경: EE=spindle_link, TXT 궤적 추종"""

    def __post_init__(self):
        super().__post_init__()

        # 1) 로봇 교체
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2) EE 링크명 고정 (USD 상: wrist_3_link 아래의 spindle_link)
        tip = "spindle_link"

        # 3) 리셋 범위 조정(초기 안정)
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # 4) 보상 항목이 참조하는 EE 링크를 모두 spindle_link로 통일
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [tip]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [tip]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [tip]
        self.rewards.end_effector_orientation_tracking_shaping.params["asset_cfg"].body_names = [tip]
        self.rewards.success_bonus.params["asset_cfg"].body_names = [tip]

        # 5) 액션 (전 관절 제어)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # 6) 커맨드: 정책/관측에서 사용할 EE 링크 지정
        self.commands.ee_pose.body_name = tip
        # (원한다면) 피치 고정
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # 7) TXT 커맨드 업데이트 이벤트 (매 step 실행)
        self.events.update_txt_command = EventTerm(
            func=txt_pose_command,
            mode="step",
            params={
                "command_name": "ee_pose",
                "file_path": "/home/eunseop/nrs_ws/src/nrs_path2/data/geodesic_waypoints.txt",  # <-- 너의 TXT 경로로 변경
                "coordinate": "world",
                "lookahead": 20,
                "loop": True,
            },
        )


@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    """플레이/디버그용 (env 수 축소, 관측 오염 해제)"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
