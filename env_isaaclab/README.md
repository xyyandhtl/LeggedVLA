# Progress
- Goal: Based on isaac_ros2, entry is isaac_ros2.py. To integrate the USD scene, locomotion policy, local planner, and VLN/VLA into a unified environment for testing a complete cross-modal complex-terrain legged robot navigation system.
- Matterport3D scenes are now supported in isaac_ros environment. To enable, collision and physics properties need to be added in the USD file. In env_isaaclab/isaac_ros2/cfg/sim.yaml, set env_name to matterport3d. The corresponding import code is env_isaaclab/isaac_ros2/env/matterport3d_env.py.
- WMP locomotion has been successfully ported.
- The original Go2 support has been extended to configurable multiple quadruped robots, with the ROS2 root node name unified to /legged.


# Todo
- USD scene：Matterport3d, Nvidia Omniverse, ...
- Task extensions(Isaaclab): locomotion, navigation, vln(follow Isaac-VLNCE), ...
- locomotion: himloco(isaacgym), wmp(isaacgym), H-Infinity(not opensource yet), legged_loco, ...
- local_planner: NavRL, Viplanner, ...
- VLN/VLA: NaVILA, TagMap/VLMaps(pre-built-map&habitat-sim), InstructNav(habitat-sim), ... 

# Tips
+ 好像isaaclab1.4以前，rsl-rl的代码很多工程都是自己提供了一份，貌似是从官方源码安装-i rsl-rl拉取rsl-rl会有不兼容bug，也有项目作者自己需要的一些改动的原因。后续需要老版本isaaclab去相应原项目内获取其rsl-rl版本
+ 因为以上rsl-rl版本混杂的原因，所以暂只能用两个环境如[requirements_isaaclab4.1.txt](requirements_isaaclab4.1.txt)和[requirements_isaaclab4.2.txt](requirements_isaaclab4.2.txt)
+ 貌似isaacsim4.2往后可以用官方./isaaclab.sh -i rsl-rl直接安装了
***************
```shell
pip install isaacsim-rl==4.2.0.2 isaacsim-replicator==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-app==4.2.0.2 --extra-index-url https://pypi.nvidia.com
# 有的项目还需要其他组件，直接装全isaacsim
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
```
+ isaaclab: https://github.com/isaac-sim/IsaacLab.git
```shell
./isaaclab.sh -i none
./isaaclab.sh -i rsl-rl # for isaaclab1.4+(not sure)
./isaaclab.sh -p -m pip install -e {THIS_REPO_DIR}/rsl_rl # for project with rsl_rl source code
# 更多其他组件貌似这个领域不太用得上了
```
***************
