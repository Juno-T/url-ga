import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from collections import OrderedDict
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from dm_env import specs
import wandb

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from optimizer.base import BaseSkillOptimizer

torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg, single_skill):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    agent = hydra.utils.instantiate(cfg)
    if single_skill:
        agent.update_skill_every_step = -1
    return agent


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, "record_skills", cfg.agent.name, cfg.task, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(
                project="urlb", 
                group=cfg.wandb_group, 
                name=exp_name, 
                tags=["record_skills"],
                config=OmegaConf.to_container(cfg))
        
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs

        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent,
                                cfg.single_skill)
        self.tmp_optim = BaseSkillOptimizer(self.agent)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = utils.load_snapshot(self.cfg)['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.task else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.task else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def record_skills(self):
        tmp = self.agent.update_skill_every_step
        self.agent.update_skill_every_step = -1
        for skill_num in range(self.agent.skill_dim):
            meta = OrderedDict()
            skill = np.zeros(self.agent.skill_dim, dtype=np.float32)
            skill[skill_num] = 1.0
            meta['skill'] = skill
            self.agent.meta_constant = meta
            self.eval()
        self.agent.meta_constant = None
        self.agent.update_skill_every_step = tmp

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)


@hydra.main(config_path='.', config_name='record_skills')
def main(cfg):
    from record_skills import Workspace as W
    print("\n\nMAIN\n\n")
    workspace = W(cfg)
    workspace.record_skills()


if __name__ == '__main__':
    main()
