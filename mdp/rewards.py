# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------------------
# (A) 기존: "명령 기반" 리워드 (원본 유지)
# --------------------------------------------------------------------------------------
def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


# --------------------------------------------------------------------------------------
# (B) 추가: "월드 고정 타깃" 리워드 (명령 없이 고정 포즈 유지)
# --------------------------------------------------------------------------------------
def _as_target_pos(env: ManagerBasedRLEnv, target_pos_xyz: Tuple[float, float, float]) -> torch.Tensor:
    """(x,y,z) -> [N,3] tensor on device"""
    return torch.tensor(target_pos_xyz, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

def _as_target_quat(env: ManagerBasedRLEnv, target_quat_wxyz: Tuple[float, float, float, float]) -> torch.Tensor:
    """(w,x,y,z) -> [N,4] tensor on device"""
    return torch.tensor(target_quat_wxyz, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

def position_fixed_error(
    env: ManagerBasedRLEnv,
    target_pos_xyz: Tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE 현재 위치 vs '월드 기준' 고정 목표 위치 간 L2 오차(벌점)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # [N,3]
    des_pos_w = _as_target_pos(env, target_pos_xyz)               # [N,3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_fixed_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    target_pos_xyz: Tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE 현재 위치 vs '월드 기준' 고정 목표 위치: tanh 커널 보상."""
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    des_pos_w = _as_target_pos(env, target_pos_xyz)
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_fixed_error(
    env: ManagerBasedRLEnv,
    target_quat_wxyz: Tuple[float, float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE 현재 자세 vs '월드 기준' 고정 목표 자세: 최단 경로 각도(벌점)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # [N,4 wxyz]
    des_quat_w = _as_target_quat(env, target_quat_wxyz)             # [N,4 wxyz]
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def small_error_bonus_fixed(
    env: ManagerBasedRLEnv,
    pos_tol: float,
    ang_tol: float,
    target_pos_xyz: Tuple[float, float, float],
    target_quat_wxyz: Tuple[float, float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """작은 오차 구간 유지 보너스(+): pos_tol[m], ang_tol[rad] 이내면 1, 아니면 0."""
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]      # [N,3]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]    # [N,4]

    des_pos_w = _as_target_pos(env, target_pos_xyz)
    des_quat_w = _as_target_quat(env, target_quat_wxyz)

    pos_err = torch.norm(curr_pos_w - des_pos_w, dim=1)               # [N]
    ang_err = quat_error_magnitude(curr_quat_w, des_quat_w)           # [N]
    ok = (pos_err < pos_tol) & (ang_err < ang_tol)
    return ok.float()
