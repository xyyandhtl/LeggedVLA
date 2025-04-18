from isaacsim import SimulationApp
import os
import hydra
import rclpy
import torch
import time
import math
import numpy as np
import cv2


FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # launch omniverse app
    simulation_app = SimulationApp({"headless": False, "anti_aliasing": cfg.sim_app.anti_aliasing,
                                    "width": cfg.sim_app.width, "height": cfg.sim_app.height, 
                                    "hide_ui": cfg.sim_app.hide_ui})
    # isaacsim extensions can only be import after app start?
    import omni
    import carb
    from ros2.agent_ros2_bridge import RobotDataManager, robot_prim_dict
    import env.sim_env as sim_env
    import env.matterport3d_env as matterport3d_env
    import agent.agent_sensors as agent_sensors
    import agent.agent_ctrl as agent_ctrl

    if cfg.robot_name == 'unitree_go2':
        from agent.scene_go2 import Go2RSLEnvCfg, camera_follow
        # Go2 Environment setup
        env_cfg = Go2RSLEnvCfg()
        print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.unitree_go2}')
        sm = agent_sensors.SensorManagerGo2(cfg.num_envs)
    elif cfg.robot_name == 'unitree_a1':
        from agent.scene_a1 import A1RSLEnvCfg, camera_follow
        # Go2 Environment setup
        env_cfg = A1RSLEnvCfg()
        print(f'{cfg.robot_name} env_cfg robot: {env_cfg.scene.unitree_a1}')
        sm = agent_sensors.SensorManagerA1(cfg.num_envs)
    else:
        raise NotImplementedError(f'[{cfg.robot_name}] env has not been implemented yet')
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.decimation = math.ceil(1. / env_cfg.sim.dt / cfg.freq)
    print(f'sim.dt {env_cfg.sim.dt}')
    print(f'decimation {env_cfg.decimation}')
    env_cfg.sim.render_interval = env_cfg.decimation
    agent_ctrl.init_base_vel_cmd(cfg.num_envs)
    print(f'{cfg.robot_name} env_cfg policy: {env_cfg.observations.policy}')
    print(f'{cfg.robot_name} env_cfg actuator: {env_cfg.actions}')

    if cfg.policy == 'default':
        from agent.agent_policy import get_rsl_rough_policy_go2
        # only have go2 checkpoints
        # env, policy = agent_ctrl.get_rsl_flat_policy_go2(go2_env_cfg)
        env, policy = get_rsl_rough_policy_go2(env_cfg)
    elif cfg.policy == "wmp_loco":
        from agent.agent_policy import load_policy_wmp, get_rsl_env_wmp
        wmp_policy = load_policy_wmp(robot_name=cfg.robot_name)
        env = get_rsl_env_wmp(env_cfg, robot_prim_dict[cfg.robot_name])
        cameras_depth = sm.add_camera_wmp(cfg.freq)
    elif cfg.policy == "him_loco":
        from agent.agent_policy import load_policy_himloco, get_rsl_env_himloco
        env = get_rsl_env_himloco(env_cfg, robot_prim_dict[cfg.robot_name])
        policy = load_policy_himloco(robot_name=cfg.robot_name)
    else:
        raise NotImplementedError(f'Policy {cfg.policy} not implemented')

    # Sensor setup
    cameras = sm.add_camera(cfg.freq)
    lidar_annotators = sm.add_rtx_lidar()

    # Simulation environment
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env() # obstacles dense
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env() # obstacles medium
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env() # obstacles sparse
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env() # warehouse
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env() # warehouse shelves
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env() # full warehouse
    elif (cfg.env_name == "office"):
        sim_env.create_office_env() #
    elif (cfg.env_name == "hospital"):
        sim_env.create_hospital_env() #
    elif (cfg.env_name == "matterport3d"):
        matterport3d_env.create_matterport3d_env(cfg.episode_idx) # matterport3d

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    system_input.subscribe_to_keyboard_events(keyboard, agent_ctrl.sub_keyboard_event)
    # reset control
    reset_ctrl = agent_ctrl.AgentResetControl()
    system_input.subscribe_to_keyboard_events(keyboard, reset_ctrl.sub_keyboard_event)
    
    # ROS2 Bridge
    rclpy.init()
    dm = RobotDataManager(env, lidar_annotators, cameras, cfg)

    # Run simulation
    sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    obs, _ = env.reset()
    obs_list = obs.cpu().numpy().tolist()[0]
    obs_list = ["{:.2f}".format(v) for v in obs_list]
    print(f'init obs shape {obs.shape}: {obs_list}')

    while simulation_app.is_running():
        # if reset_ctrl.reset_flag: # todo: how to reset?
        #     obs, _ = env.unwrapped().reset()
        #     simulation_app.update()
        #     print(f'env reset done')
        #     reset_ctrl.reset_flag = False
        start_time = time.time()
        with torch.inference_mode():
            if cfg.policy == "wmp_loco":
                actions = wmp_policy.inference_action(obs)
                obs, _, _, _ = env.step(actions)
                # step the environment
                depth_tensor = torch.zeros((cfg.num_envs, 64, 64), dtype=torch.float32)
                for i, camera in enumerate(cameras_depth):
                    depth = camera.get_depth()
                    if depth is not None:
                        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                        depth = depth / 2 - 0.5
                        depth[(depth <= -0.5) | (depth > 0.5)] = 0.5
                        depth_tensor[i] = torch.from_numpy(depth.copy())
                # wmp_policy.update_wm(actions, obs, torch.from_numpy(np.ones(wmp_policy.depth_resized) * 0.5))
                wmp_policy.update_wm(actions, obs, depth_tensor)
            else:
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            # # ROS2 data
            dm.pub_ros2_data()
            rclpy.spin_once(dm)

            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
    
    dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
    