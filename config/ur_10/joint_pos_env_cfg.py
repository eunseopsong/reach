# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# 로봇 교체: UR10e + spindle (로컬 USD)
from isaaclab_assets import UR10E_W_SPINDLE_CFG


@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    """UR10e(+spindle)로 교체한 Reach 환경"""

    def __post_init__(self):
        super().__post_init__()

        # 1) 로봇 교체 (장면 배치 경로만 지정)
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2) EE 링크명: 로봇 cfg가 제공하면 그걸 쓰고, 없으면 USD 기준으로 'wrist_3_link'
        tip = getattr(self.scene.robot, "ee_frame_name", None) or "wrist_3_link"

        # 3) 이벤트
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # 4) (중요) 고정 타깃 리워드의 asset_cfg에 EE 링크 반영
        #   - reach_env_cfg.RewardsCfg에서 body_names=MISSING로 둔 항목들을 여기서 채운다.
        self.rewards.position_fixed.params["asset_cfg"].body_names = [tip]
        self.rewards.position_fixed_tanh.params["asset_cfg"].body_names = [tip]
        self.rewards.orientation_fixed.params["asset_cfg"].body_names = [tip]

        # 5) 액션 (조인트 전부)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # 6) 커맨드(목표자세)에서도 EE 링크/피치 고정 (관측 파이프라인 호환 목적)
        self.commands.ee_pose.body_name = tip
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    """플레이 모드(시각화/디버깅용)"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # (추가 덮어쓰기는 필요 없음 — 부모에서 이미 EE 링크 처리 완료)
