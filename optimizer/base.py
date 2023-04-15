from collections import OrderedDict

import numpy as np
import torch

from agent.diayn import DIAYNAgent

class BaseSkillOptimizer():
    def __init__(self, agent: DIAYNAgent, name="Optimizer"):
        self.name = name
        self.agent = agent
        self.skill_spec = self.agent.get_meta_specs()[0]
        self.skill_dim = self.skill_spec.shape[0]
    
    def gen_meta_from_skills(self, skills):
        if isinstance(skills, torch.Tensor):
            skills = skills.cpu().numpy()
        meta_array = np.zeros((len(skills), self.skill_dim), dtype=np.float32)
        meta_array[np.arange(len(skills)), skills]=1
        meta_list = [OrderedDict(skill = ma) for ma in meta_array]
        return meta_list
    
    def gen_skills_from_meta(self, meta):
        if isinstance(meta, list):
            meta = np.array([m['skill'] for m in meta])
        skills = np.argmax(meta, axis=1)
        return skills
    
    def get_random_meta(self, length=1000):
        return self.gen_meta_from_skills(np.random.randint(self.skill_dim, size=(length,)))
    
    def ask(self, seed_step=False):
        pass

    def tell(self, step_fitness):
        pass

    def eval_log_wandb(self, global_step, global_frame):
        pass