# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Tuple

import torch
from isaaclab.utils import configclass

# ✔ 패키지 레벨에서 export 된 클래스로 임포트 (경로 중요!)
from isaaclab.envs.mdp.commands import (
    UniformPoseCommand,
    UniformPoseCommandCfg,
)


class UniformPoseWithTCPShift(UniformPoseCommand):
    """UniformPoseCommand 결과에 EE 로컬 프레임 기준 TCP 오프셋을 더해주는 커맨드."""

    def __init__(self, cfg: "UniformPoseWithTCPShiftCfg", env):
        super().__init__(cfg, env)
        # EE 로컬 프레임 기준 오프셋(m). 스핀들이 +X라면 (0.185, 0, 0)
        self._tcp_offset = torch.tensor(cfg.tcp_offset, device=env.device, dtype=torch.float32)

    def get_value(self) -> torch.Tensor:
        """[pos(3), quat(4)] 반환. pos에 tcp_offset을 더함."""
        goal = super().get_value()          # shape: (N, 7)
        goal[..., 0:3] = goal[..., 0:3] + self._tcp_offset
        return goal


@configclass
class UniformPoseWithTCPShiftCfg(UniformPoseCommandCfg):
    """TCP 오프셋을 적용하는 Pose 커맨드 설정(Cfg)."""

    class_type = UniformPoseWithTCPShift

    # 기본값: +X 방향으로 0.185 m (스핀들 축이 +X가 아니라면 이 값/축만 바꿔줘)
    tcp_offset: Tuple[float, float, float] = (0.185, 0.0, 0.0)
