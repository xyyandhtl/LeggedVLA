from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import class_to_dict, get_load_path
from rsl_rl_wmp.runners import WMPRunner

from wmp_deployment_runner import WMPRunner
import torch


class WMPPolicy:
    def __init__(self, args):
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        # override some parameters for testing
        env_cfg.env.num_envs = 1

        # todo: delete useless on inference
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = True
        env_cfg.noise.add_noise = False
        # env_cfg.domain_rand.randomize_friction = False
        # env_cfg.domain_rand.randomize_restitution = False
        # env_cfg.commands.heading_command = True

        env_cfg.domain_rand.friction_range = [1.0, 1.0]
        env_cfg.domain_rand.restitution_range = [0.0, 0.0]
        env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
        env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
        env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
        env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

        env_cfg.domain_rand.randomize_action_latency = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_gains = True
        # env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_link_mass = False
        # env_cfg.domain_rand.randomize_com_pos = False
        env_cfg.domain_rand.randomize_motor_strength = False

        train_cfg.runner.amp_num_preload_transitions = 1

        env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
        env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]

        env_cfg.terrain.border_size = 5
        env_cfg.terrain.slope_treshold = 0.75
        env_cfg.env.episode_length_s = 1000

        # env_cfg.terrain.mesh_type = 'plane'
        if (env_cfg.terrain.mesh_type == 'plane'):
            env_cfg.rewards.scales.feet_edge = 0
            env_cfg.rewards.scales.feet_stumble = 0

        if (args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt', 'mixed']):
            print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
            args.terrain = 'climb'
        env_cfg.terrain.terrain_proportions = {
            'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
            'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
            'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
            'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
            'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
            'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
            # 'mixed': [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1, 0.2, 0.1]
            # 'mixed': [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
            # 'mixed': [0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0]
            'mixed': [0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.2, 0]
            # # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        }[args.terrain]

        env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8]
        env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        env_cfg.commands.ranges.heading = [0, 0]

        env_cfg.commands.ranges.flat_lin_vel_x = [0.8, 0.8]
        env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
        env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

        # env_cfg.commands.ranges.lin_vel_x = [0.6, 1.0]
        # env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
        # env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        # env_cfg.commands.ranges.heading = [-3.14, 3.14]

        # env_cfg.commands.ranges.flat_lin_vel_x = [-0.6, 0.6]
        # env_cfg.commands.ranges.flat_lin_vel_y = [-0.6, 0.6]
        # env_cfg.commands.ranges.flat_ang_vel_yaw = [-3.14, 3.14]

        env_cfg.depth.use_camera = True

        # load policy
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = 'WMP'

        train_cfg.runner.checkpoint = -1    # 9000
        train_cfg_dict = class_to_dict(train_cfg)

        self.runner = WMPRunner(train_cfg_dict, device=args.sim_device)

        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
                                    checkpoint=train_cfg.runner.checkpoint)
        print(f"Loading model from: {resume_path}")
        self.runner.load(resume_path)
        self.policy = self.runner.get_inference_policy(device=args.sim_device)

        self.env_cfg, self.train_cfg = env_cfg, train_cfg
        # self.env = env
        self.logger = Logger(0.02)
        self.num_critic_obs, self.num_actor_obs, self.privileged_dim, self.height_dim, self.num_actions, \
            self.num_envs, self.step_dt, self.use_camera, self.depth_resized, self.update_interval = \
            self.runner.num_critic_obs, self.runner.num_actor_obs, self.runner.privileged_dim, self.runner.height_dim, \
            self.runner.num_actions, self.runner.num_envs, self.runner.step_dt, self.runner.use_camera, \
            self.runner.depth_resized, self.runner.update_interval
        self.prop_dim = self.runner.prop_dim
        self.device = args.sim_device

        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(self.runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)

        history_length = 5
        self.trajectory_history = torch.zeros(size=(self.num_envs, history_length, self.num_actor_obs -
                                              self.privileged_dim - self.height_dim - 3), device=args.sim_device)
        self.obs_without_command = None
        self.wm_latent = self.wm_action = None
        self.world_model = self.runner._world_model.to(args.sim_device)
        self.wm_is_first = torch.ones(self.num_envs, device=args.sim_device)
        self.wm_update_interval = self.update_interval
        self.wm_action_history = torch.zeros(size=(self.num_envs, self.wm_update_interval, self.num_actions),
                                        device=args.sim_device)

        self.wm_obs = {}
        self.wm_feature = torch.zeros((self.num_envs, self.runner.wm_feature_dim), device=args.sim_device)

        # self.total_reward = 0
        # self.not_dones = torch.ones((self.num_envs,), device=args.sim_device)

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
        return actions

    def update_wm(self, actions, obs):
        if self.global_counter % self.wm_update_interval == 0:
            if self.use_camera:
                self.wm_obs["image"][0] = self.infos["depth"].unsqueeze(-1).to(self.world_model.device)

            wm_embed = self.world_model.encoder(self.wm_obs)
            self.wm_latent, _ = self.world_model.dynamics.obs_step(self.wm_latent, self.wm_action, wm_embed, self.wm_obs["is_first"],
                                                         sample=True)
            self.wm_feature = self.world_model.dynamics.get_deter_feat(self.wm_latent)
            self.wm_is_first[:] = 0
        # obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # update world model input
        self.wm_action_history = torch.concat(
            (self.wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        wm_obs = {
            "prop": obs[:, self.privileged_dim: self.privileged_dim + self.prop_dim],
            "is_first": self.wm_is_first,
        }
        if self.use_camera:
            wm_obs["image"] = torch.zeros(((self.num_envs,) + self.depth_resized + (1,)),
                                          device=self.world_model.device)

        # self.wm_action = self.wm_action_history.flatten(1)    # useless when only actor running

        # when env need to be reset
        # wm_action_history[reset_env_ids, :] = 0
        # wm_is_first[reset_env_ids] = 1
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


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    args.rl_device = args.sim_device
    ppo_runner = WMPPolicy(args)
    # logic method_call code
    # obs = env.reset()
    # ppo_runner.init_wmp_policy(obs)
    # while True:
    #     actions = ppo_runner.inference_action(obs)
    #     obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())
    #     ppo_runner.update_wm(actions, obs)
