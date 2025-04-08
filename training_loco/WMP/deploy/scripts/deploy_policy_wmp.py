# import glob
# import pickle as pkl
import lcm
import sys
import json

from deploy.utils.deployment_runner import DeploymentRunner
from deploy.envs.lcm_agent import LCMAgent
from deploy.utils.cheetah_state_estimator import StateEstimator
from deploy.utils.command_profile import *

from wmp.wmp_policy import WMPPolicy

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(robot_name, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    with open(f'./agent_config_{robot_name}.json', 'r') as f:
        cfg = json.load(f)
    print('config:', json.dumps(cfg, indent=2))

    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    wmp_policy = WMPPolicy()

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(wmp_policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run_wmp_policy(max_steps=max_steps, logging=True)


if __name__ == '__main__':
    robot_name = "aliengo"

    experiment_name = "example_experiment"

    load_and_run_policy(robot_name, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
