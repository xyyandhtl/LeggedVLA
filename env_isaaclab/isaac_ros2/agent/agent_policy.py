import os
import torch
from pathlib import Path
import gymnasium as gym

from agent.ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
POLICY_WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent / 'ckpts')


class HIMLocoEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, history_len=6):
        super().__init__(env)
        self.history_len = history_len
        self.obs_history = None

    def reset(self):
        obs, info = super().reset()
        # 初始化历史观测，维度：[num_envs, history_len, obs_dim]
        self.obs_history = torch.stack([obs] * self.history_len, dim=1)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # 更新历史观测
        self.obs_history = torch.cat([obs.unsqueeze(1) ,self.obs_history[:, :-1]], dim=1)
        return self._get_stacked_obs(), reward, done, info

    def _get_stacked_obs(self):
        # 展平维度：[num_envs, history_len * obs_dim]
        return self.obs_history.reshape(self.obs_history.shape[0], -1)


class WMPObsEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wmp_obs_dim = 285
        self.wmp_obs_dim_deploy = 45
        self.non_privilege_start = 53
        self.non_privilege_end = self.non_privilege_start + self.wmp_obs_dim_deploy

        # 用于对特定观测段做通道置换
        self.reverse_index_list = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

        # 输出动作的维度反向映射（用于推理部署）
        self.forward_index_list = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

    def reset(self):
        obs, info = super().reset()
        new_obs = self._construct_padded_obs(obs)
        return new_obs, info

    def step(self, action):
        # 将 action 置换
        action = action[:, self.forward_index_list]

        obs, reward, done, info = super().step(action)
        new_obs = self._construct_padded_obs(obs)
        return new_obs, reward, done, info

    def _construct_padded_obs(self, orig_obs):
        num_envs = orig_obs.shape[0]
        new_obs = torch.zeros((num_envs, self.wmp_obs_dim), dtype=orig_obs.dtype, device=orig_obs.device)
        # 填入原始观测的非特权部分
        new_obs[:, self.non_privilege_start:self.non_privilege_end] = orig_obs
        # 做 reverse_index 置换的部分：joint_pos, joint_vel, last_action 各12维
        base_idx = self.non_privilege_start  # 对应45维起始位置
        # wmp cmd_vel is 6:9, while default is 0:3
        new_obs[:, base_idx: base_idx + 6], new_obs[:, base_idx + 6: base_idx + 9] = \
            new_obs[:, base_idx + 3: base_idx + 9].clone(), new_obs[:, base_idx: base_idx + 3].clone()
        # joint_pos: [9:21]
        new_obs[:, base_idx + 9:base_idx + 21] = new_obs[:, base_idx + 9:base_idx + 21][:, self.reverse_index_list]
        # joint_vel: [21:33]
        new_obs[:, base_idx + 21:base_idx + 33] = new_obs[:, base_idx + 21:base_idx + 33][:, self.reverse_index_list]
        # last_action: [33:45]
        new_obs[:, base_idx + 33:base_idx + 45] = new_obs[:, base_idx + 33:base_idx + 45][:, self.reverse_index_list]
        return new_obs


def get_rsl_flat_policy_go2(cfg):
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    # env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"),
                                    run_dir=agent_cfg["load_run"],
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    print('Loaded policy from: ', ckpt_path)
    return env, policy

def get_rsl_rough_policy_go2(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"),
                                    run_dir=agent_cfg["load_run"],
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    print('Loaded policy from: ', ckpt_path)
    return env, policy

def get_rsl_env_himloco(cfg, unitree_robot):
    cfg.observations.policy.height_scan = None
    env = gym.make(f"Isaac-Velocity-Flat-Unitree-{unitree_robot}-v0", cfg=cfg)
    env = HIMLocoEnvWrapper(env)
    return env

def get_rsl_env_wmp(cfg, unitree_robot):
    cfg.observations.policy.height_scan = None
    env = gym.make(f"Isaac-Velocity-Flat-Unitree-{unitree_robot}-v0", cfg=cfg)
    env = WMPObsEnvWrapper(env)
    return env

def load_policy_himloco(robot_name):
    path = f'{POLICY_WEIGHTS_PATH}/him_loco/policy_{robot_name}.pt'
    body = torch.jit.load(path)
    print('Loaded policy from: ', path)
    def policy(obs):
        return body.forward(obs.to('cpu'))
    return policy

def load_policy_wmp(robot_name):
    # todo: load jit simple
    import sys
    from pathlib import Path
    LEGGED_GYM_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / 'training_loco/WMP_loco')
    sys.path.append(LEGGED_GYM_ROOT_DIR)
    from wmp.wmp_policy import WMPPolicy
    wmp_policy = WMPPolicy(ckpts_path=f'{POLICY_WEIGHTS_PATH}/wmp_loco', robot_name=robot_name)
    return wmp_policy