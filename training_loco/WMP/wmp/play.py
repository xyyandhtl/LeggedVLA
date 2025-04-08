import isaacgym

from wmp_policy import WMPPolicy, LEGGED_GYM_ROOT_DIR
# from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_cols = 1
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
    if(env_cfg.terrain.mesh_type == 'plane'):
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0


    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt', 'mixed']):
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

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    obs_list = obs[:, 53:98].cpu().numpy().tolist()[0]
    obs_list = ["{:.2f}".format(v) for v in obs_list]
    print(f'obs: {obs_list}')
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'WMP'


    train_cfg.runner.checkpoint = -1
    policy = WMPPolicy()
    policy.init_wmp_policy(obs)

    print(f'sim dt {env.dt}')
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)
    for i in range(1*int(env.max_episode_length) + 3):
        obs[:, :53] = 0.0       # test remove the privileged obs
        obs[:, 98:] = 0.0

        actions = policy.inference_action(obs)

        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())
        obs[:, :53] = 0.0
        obs[:, 98:] = 0.0
        obs_list = obs[:, 53:98].cpu().numpy().tolist()[0]
        obs_list = ["{:.2f}".format(v) for v in obs_list]
        print(f'obs: {obs_list}')
        print(f'actions: {actions}')

        # policy.infos = infos

        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        reset_env_ids = reset_env_ids.cpu().numpy()
        # if (len(reset_env_ids) > 0):
        #     wm_action_history[reset_env_ids, :] = 0
        #     wm_is_first[reset_env_ids] = 1

        # print(f"depth {infos['depth']} global_counter {policy.global_counter}")
        policy.update_wm(actions, obs, infos['depth'], reset_env_ids)
        # policy.update_wm(actions, obs, torch.from_numpy(np.ones(policy.depth_resized) * 0.5), reset_env_ids)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                print(f'writing to file {filename}')
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            # camara_position = env.root_states[2, :3].detach().cpu().numpy() + [0.32, 0, 0.03]
            # lootat = camara_position + [1.0, 0, 0]
            lootat = env.root_states[0, :3]     # 0-4: stairs, gap, pit, tilt, crawl
            camara_position = lootat.detach().cpu().numpy() + [0, -2.5, 0]
            # camara_position = lootat.detach().cpu().numpy() + [-3.0, 0, 0.5]
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    print('total reward:', total_reward)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    # args.rl_device = args.sim_device
    play(args)
