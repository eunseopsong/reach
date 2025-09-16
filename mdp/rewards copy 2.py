# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TCP offset (로컬 z축 방향으로 0.185 m)
TCP_OFFSET = torch.tensor([0.0, 0.0, -0.185], dtype=torch.float32)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the TCP position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 목표 위치 (월드)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)

    # 현재 링크 pose
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]      # (N, 3)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]    # (N, 4)

    # (중요) TCP 오프셋을 배치 크기(N)에 맞게 확장 -> (N, 3)
    offset_local = TCP_OFFSET.to(curr_pos_w.device, dtype=curr_pos_w.dtype).unsqueeze(0).expand(curr_pos_w.shape[0], -1)
    # 월드로 회전 적용
    offset_w = quat_apply(curr_quat_w, offset_local)                   # (N, 3)
    tcp_pos_w = curr_pos_w + offset_w

    return torch.norm(tcp_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the TCP position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)

    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]

    # (중요) 배치 확장
    offset_local = TCP_OFFSET.to(curr_pos_w.device, dtype=curr_pos_w.dtype).unsqueeze(0).expand(curr_pos_w.shape[0], -1)
    offset_w = quat_apply(curr_quat_w, offset_local)
    tcp_pos_w = curr_pos_w + offset_w

    distance = torch.norm(tcp_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / (std + 1e-6))


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)

    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)
