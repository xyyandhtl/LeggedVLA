from rsl_rl_wmp.algorithms import AMPPPO, PPO
from rsl_rl_wmp.modules import ActorCritic, ActorCriticWMP, ActorCriticRecurrent
from rsl_rl_wmp.env import VecEnv
from rsl_rl_wmp.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl_wmp.datasets.motion_loader import AMPLoader
from rsl_rl_wmp.utils.utils import Normalizer
from rsl_rl_wmp.modules import DepthPredictor
import torch.optim as optim

from dreamer.models import *
import ruamel.yaml as yaml
import argparse
from pathlib import Path
from dreamer import tools


class WMPDeploymentRunner:

    def __init__(self,
                 train_cfg,
                 device='cpu',
                 history_length=5,
                 ):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.history_length = history_length

        self.num_critic_obs = 285
        self.num_actor_obs = 285 # would only use part of it
        self.privileged_dim = 53
        self.height_dim = 187
        self.num_actions = 12
        self.num_envs = 1
        self.step_dt = 0.02
        self.use_camera = True
        self.depth_resized = (64, 64)
        self.update_interval = 5

        # build world model
        # self._build_world_model()
        # world model
        print('Begin construct world model')
        configs = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "dreamer/configs.yaml").read_text()
        )
        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value
        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=True)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        parser.add_argument("--resume", action="store_true", default=False)
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()
        # allow world model and rl env on different device
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device
        self.wm_config.num_actions = self.wm_config.num_actions * self.update_interval
        self.prop_dim = self.num_actor_obs - self.privileged_dim - self.height_dim - self.num_actions   # 33
        image_shape = self.depth_resized + (1,)
        obs_shape = {'prop': (self.prop_dim,), 'image': image_shape, }

        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        self.wm_feature_dim = self.wm_config.dyn_deter  # + self.wm_config.dyn_stoch * self.wm_config.dyn_discrete

        # build depth predictor
        self.depth_predictor = DepthPredictor().to(self._world_model.device)
        self.depth_predictor_opt = optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
                                              weight_decay=self.depth_predictor_cfg["weight_decay"])

        self.history_dim = history_length * (self.num_actor_obs - self.privileged_dim - self.height_dim - 3) # =42,exclude command
        actor_critic = ActorCriticWMP(num_actor_obs=self.num_actor_obs,
                                          num_critic_obs=self.num_critic_obs,
                                          num_actions=self.num_actions,
                                          height_dim=self.height_dim,
                                          privileged_dim=self.privileged_dim,
                                          history_dim=self.history_dim,
                                          wm_feature_dim=self.wm_feature_dim,
                                          **self.policy_cfg).to(self.device)

        print({f'wmp motion files {self.cfg["amp_motion_files"]}'})
        amp_data = AMPLoader(
            device, time_between_frames=self.step_dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"])  # AMPPPO
        min_std = (torch.tensor([0.0864, 0.0801, 0.0974, 0.0864, 0.0801, 0.0974, 0.0864, 0.0801, 0.0974,
                    0.0864, 0.0801, 0.0974], device=self.device))
        self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                  min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.num_envs, self.num_steps_per_env, [self.num_actor_obs],
                              [self.num_critic_obs], [self.num_actions], self.history_dim, self.wm_feature_dim)

        # Log
        # self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0


    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'world_model_dict': self._world_model.state_dict(),
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            'depth_predictor': self.depth_predictor.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer = False):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        print(f'iteration resume from: {self.current_learning_iteration}')
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
