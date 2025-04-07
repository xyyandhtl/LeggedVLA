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

    import omni
    import carb
    import go2.go2_ctrl as go2_ctrl
    import ros2.go2_ros2_bridge as go2_ros2_bridge
    from go2.go2_env import Go2RSLEnvCfg, camera_follow
    import env.sim_env as sim_env
    import env.matterport3d_env as matterport3d_env
    import go2.go2_sensors as go2_sensors

    from wmp.wmp_policy import WMPPolicy

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1. / go2_env_cfg.sim.dt / cfg.freq)
    print(f'sim.dt {go2_env_cfg.sim.dt}')
    print(f'decimation {go2_env_cfg.decimation}')
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    # env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

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


    wmp_policy = WMPPolicy()

    # Sensor setup
    sm = go2_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()
    cameras = sm.add_camera(cfg.freq)

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
    # ROS2 Bridge
    rclpy.init()
    dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)

    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    obs_list = obs.cpu().numpy().tolist()[0]
    obs_list = ["{:.2f}".format(v) for v in obs_list]
    print(f'init obs: {obs_list}')

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():            
            # control joints
            # actions = policy(obs)

            wmp_obs = wmp_policy.obs_convert_from_lab_env(obs)
            actions = wmp_policy.inference_action(wmp_obs)

            obs_list = obs.cpu().numpy().tolist()[0]
            obs_list = ["{:.2f}".format(v) for v in obs_list]
            print(f'obs: {obs_list}')
            print(f'[{start_time}] actions: {actions}')

            # step the environment
            obs, _, _, _ = env.step(actions)
            depth_tensor = torch.zeros((cfg.num_envs, 64, 64), dtype=torch.float32)

            # if wmp_policy.global_counter % wmp_policy.wm_update_interval == 0:
            # for i, camera in enumerate(cameras):
            #     depth = camera.get_depth()
            #     if depth is not None:
            #
            #         depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            #         depth = -(depth / 2 - 0.5)
            #         depth[(depth < -0.5) | (depth > 0.5)] = 0.5
            #         depth = np.flipud(depth)
            #
            #         cv2.imwrite(f'./logs/depth_{wmp_policy.global_counter}.png', depth + 0.5)
            #         # print(f'depth {depth}')
            #         depth_tensor[i] = torch.from_numpy(depth.copy())

            wmp_obs = wmp_policy.obs_convert_from_lab_env(obs)
            wmp_policy.update_wm(actions, wmp_obs, depth_tensor)

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
    