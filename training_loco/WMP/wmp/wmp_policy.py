from pathlib import Path
import numpy as np
import torch
import os

# from legged_gym.envs import LEGGED_GYM_ROOT_DIR
from wmp.train_config.aliengo_amp_config import AlienGoAMPCfgPPO
# from legged_gym.utils.helpers import class_to_dict, get_load_path
from wmp.wmp_deployment_runner import WMPDeploymentRunner


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
        print(f'last_run {last_run}')
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path


class WMPPolicy:
    def __init__(self, device='cuda:0'):
        # self.train_cfg = train_cfg
        self.device = device
        # load policy
        train_cfg = AlienGoAMPCfgPPO()
        train_cfg.runner.amp_num_preload_transitions = 1

        # load policy
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = 'a1'
        # train_cfg.runner.load_run = 'go2'

        train_cfg.runner.checkpoint = -1    # 9000
        train_cfg_dict = class_to_dict(train_cfg)

        self.runner = WMPDeploymentRunner(train_cfg_dict, device=self.device)

        log_root = str(Path(__file__).resolve().parent / 'weights')
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
                                    checkpoint=train_cfg.runner.checkpoint)
        print(f"Loading model from: {resume_path}")
        self.runner.load(resume_path)
        self.policy = self.runner.get_inference_policy(device=self.device)

        # self.env = env
        # self.logger = Logger(0.02)
        self.num_critic_obs = self.runner.num_critic_obs
        self.num_actor_obs = self.runner.num_actor_obs
        self.privileged_dim = self.runner.privileged_dim
        self.height_dim = self.runner.height_dim
        self.num_actions = self.runner.num_actions
        self.num_envs = self.runner.num_envs
        self.step_dt = self.runner.step_dt
        self.use_camera = self.runner.use_camera
        self.depth_resized = self.runner.depth_resized
        self.update_interval = self.runner.update_interval

        self.prop_dim = self.runner.prop_dim
        self.device = self.device

        # export policy as a jit module (used to run it from C++)
        # if EXPORT_POLICY:
        #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        #     export_policy_as_jit(self.runner.alg.actor_critic, path)
        #     print('Exported policy as jit script to: ', path)

        history_length = 5
        self.trajectory_history = torch.zeros(size=(self.num_envs, history_length, self.num_actor_obs -
                                              self.privileged_dim - self.height_dim - 3), device=self.device)
        self.obs_without_command = None
        self.wm_latent = self.wm_action = None
        self.world_model = self.runner._world_model.to(self.device)
        self.wm_is_first = torch.ones(self.num_envs, device=self.device)
        self.wm_update_interval = self.update_interval
        self.wm_action_history = torch.zeros(size=(self.num_envs, self.wm_update_interval, self.num_actions),
                                        device=self.device)

        self.wm_obs = {}
        self.wm_feature = torch.zeros((self.num_envs, self.runner.wm_feature_dim), device=self.device)

        # self.total_reward = 0
        # self.not_dones = torch.ones((self.num_envs,), device=self.device)

        self.global_counter = 1
        self.infos = {}

    def init_wmp_policy(self, obs):
        self.obs_without_command = torch.concat((obs[:, self.privileged_dim:self.privileged_dim + 6],
                                            obs[:, self.privileged_dim + 9:-self.height_dim]), dim=1)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], self.obs_without_command.unsqueeze(1)), dim=1)

        self.wm_obs = {
            "prop": obs[:, self.privileged_dim: self.privileged_dim + self.prop_dim],
            "is_first": self.wm_is_first,
        }
        if self.use_camera:
            self.wm_obs["image"] = torch.zeros(((self.num_envs,) + self.depth_resized + (1,)),
                                          device=self.world_model.device)

    def inference_action(self, obs):
        history = self.trajectory_history.flatten(1).to(self.device)
        actions = self.policy(obs.detach(), history.detach(), self.wm_feature.detach())
        self.global_counter += 1
        return actions

    def update_wm(self, actions, obs, depth, reset_env_ids=None):
        if self.global_counter % self.wm_update_interval == 0:
            if self.use_camera:
                depth = torch.from_numpy(np.ones(self.depth_resized) * 0.5)   # for depth-zero input test
                self.wm_obs["image"][0] = depth.unsqueeze(-1).to(self.world_model.device)
                # print(f'depth {depth}')

            wm_embed = self.world_model.encoder(self.wm_obs)
            self.wm_latent, _ = self.world_model.dynamics.obs_step(self.wm_latent, self.wm_action, wm_embed, self.wm_obs["is_first"],
                                                         sample=True)
            self.wm_feature = self.world_model.dynamics.get_deter_feat(self.wm_latent)
            self.wm_is_first[:] = 0
        # obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # update world model input
        self.wm_action_history = torch.concat(
            (self.wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        self.wm_obs = {
            "prop": obs[:, self.privileged_dim: self.privileged_dim + self.prop_dim],
            "is_first": self.wm_is_first,
        }
        if self.use_camera:
            self.wm_obs["image"] = torch.zeros(((self.num_envs,) + self.depth_resized + (1,)),
                                          device=self.world_model.device)

        self.wm_action = self.wm_action_history.flatten(1)

        # when env need to be reset
        if reset_env_ids:
            self.wm_action_history[reset_env_ids, :] = 0
            self.wm_is_first[reset_env_ids] = 1
        # process trajectory history
        # env_ids = dones.nonzero(as_tuple=False).flatten()
        # trajectory_history[env_ids] = 0
        self.obs_without_command = torch.concat((obs[:, self.privileged_dim:self.privileged_dim + 6],
                                            obs[:, self.privileged_dim + 9:-self.height_dim]), dim=1)
        self.trajectory_history = torch.concat(
            (self.trajectory_history[:, 1:], self.obs_without_command.unsqueeze(1)), dim=1)

        # todo: if there is a way to log policy step reward info
        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        #         }
        #     )
        # if 0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes > 0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i == stop_rew_log:
        #     logger.print_rewards()
        return actions

    def obs_convert_from_lab_env(self, lab_obs):
        wmp_obs = torch.zeros(size=(self.num_envs, self.num_actor_obs), device=self.device)
        wmp_obs[:, self.privileged_dim:self.privileged_dim + self.prop_dim + self.num_actions] = lab_obs[:, 3:]
        wmp_obs[:, self.privileged_dim:self.privileged_dim + 3] *= 0.25                 # WMP normalization
        wmp_obs[:, self.privileged_dim + 21:self.privileged_dim + self.prop_dim] *= 0.05    # WMP normalization
        wmp_obs[:, [self.privileged_dim + 10, self.privileged_dim + 13, self.privileged_dim + 16, self.privileged_dim + 19]] -= 0.8    # WMP normalization
        wmp_obs[:, [self.privileged_dim + 11, self.privileged_dim + 14, self.privileged_dim + 17, self.privileged_dim + 20]] += 1.5    # WMP normalization
        # wmp_obs[:, [self.privileged_dim + 13, self.privileged_dim + 14, self.privileged_dim + 15, self.privileged_dim + 16]] -= 0.8  # WMP normalization
        # wmp_obs[:, [self.privileged_dim + 17, self.privileged_dim + 18, self.privileged_dim + 19, self.privileged_dim + 20]] += 1.5    # WMP normalization
        # wmp_obs[:, self.privileged_dim + self.prop_dim:self.privileged_dim + self.prop_dim + self.num_actions] *= 4    # WMP normalization
        return wmp_obs

    def obs_convert_from_wtw_env(self, control_obs):
        wmp_obs = torch.zeros(size=(self.num_envs, self.num_actor_obs), device=self.device)

        # observe_vel False
        # # gravity and command
        # wmp_obs[:, self.privileged_dim + 3:self.privileged_dim + 9] = control_obs['obs'][:6]
        # # dof_pos, dof_vel and actions
        # wmp_obs[:, self.privileged_dim + 9:self.privileged_dim + 45] = control_obs['obs'][18:54]

        # set walk-these-ways' observe_vel True to get ang_vel needed by WMP
        wmp_obs[:, self.privileged_dim:self.privileged_dim + 3] = control_obs['obs'][3:6]
        # gravity and command
        wmp_obs[:, self.privileged_dim + 3:self.privileged_dim + 9] = control_obs['obs'][6:12]
        # dof_pos, dof_vel and actions
        wmp_obs[:, self.privileged_dim + 9:self.privileged_dim + 45] = control_obs['obs'][24:60]
        return wmp_obs


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    # args = get_args()
    ppo_runner = WMPPolicy()
    # logic method_call code
    # obs = env.reset()
    # ppo_runner.init_wmp_policy(obs)
    # while True:
    #     actions = ppo_runner.inference_action(obs)
    #     obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())
    #     ppo_runner.update_wm(actions, obs)
