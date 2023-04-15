import torch
import numpy as np

import wandb

from optimizer.base import BaseSkillOptimizer
from optimizer.vanilla_ga import VanillaGA
from agent.diayn import DIAYNAgent

class PeriodicGA(BaseSkillOptimizer):
    def __init__(self,
        agent: DIAYNAgent,
        pop_size = 100,
        elite_size = 10,
        mutation_prob = 0.05,
        ep_len = 1000,
        device = 'cuda',
        max_period = 500,
        min_period = 2,
        max_offset = 500,
        max_repeat = 10,
        **kwargs,
    ):
        super().__init__(agent, **kwargs)
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.ep_len = ep_len
        self.device = device
        self.max_period = max_period
        self.min_period = min_period
        self.max_offset = max_offset
        self.max_repeat = max_repeat
    

        self.pop_period, self.pop_offset, self.pop_repeat, self.pop_default = self.random_init_pop(self.pop_size)
        self._pop_top_performer = None
        self._top_performer = None
        self._top_fitness = -1e9

    @property
    def pop_top_performer(self):
        if self._pop_top_performer is None:
            return None
        pop_top_skills = self.gen_skills_from_pop(*self._pop_top_performer)
        return self.gen_meta_from_skills(pop_top_skills[0])
    
    @property
    def top_performer(self):
        if self._top_performer is None:
            return None
        top_skills = self.gen_skills_from_pop(*self._top_performer)
        return self.gen_meta_from_skills(top_skills[0])
    
    @property
    def top_fitness(self):
        if self._top_performer is None:
            return None
        return self._top_fitness

    def random_init_pop(self, size):
        pop_period = torch.exp(torch.rand(size, self.skill_dim)*np.log(self.max_period/self.min_period)+np.log(self.min_period)).to(torch.int64).to(self.device)
        pop_offset = (torch.exp(torch.rand(size, self.skill_dim)*np.log(self.max_period)).to(torch.int64)-1).to(self.device)
        pop_repeat = ((torch.rand(size, self.skill_dim).to(self.device)*pop_period).to(torch.int64)+1)
        cap_repeat = ((torch.rand(size, self.skill_dim).to(self.device)*self.max_repeat).to(torch.int64)+1)
        pop_repeat[pop_repeat>self.max_repeat]=cap_repeat[pop_repeat>self.max_repeat]
        pop_default = torch.randint(0, self.skill_dim, (size,)).to(self.device)
        pop_period[torch.arange(size), pop_default] = 1
        pop_offset[torch.arange(size), pop_default] = 0
        pop_repeat[torch.arange(size), pop_default] = 1

        return pop_period, pop_offset, pop_repeat, pop_default

    
    def generate_new_pop(self, pop_period, pop_offset, pop_repeat, pop_default, ep_fitness):
        _children_size = self.pop_size - self.elite_size
        elite_indices = torch.topk(ep_fitness, k=self.elite_size).indices

        elites_period = pop_period[elite_indices]
        elites_offset = pop_offset[elite_indices]
        elites_repeat = pop_repeat[elite_indices]
        elites_default = pop_default[elite_indices]

        # Parents selection
        p1 = torch.multinomial(ep_fitness, num_samples=_children_size, replacement=True)
        p2_dist = ep_fitness.repeat(_children_size, 1)
        p2_dist[torch.arange(_children_size), p1] = 0
        p2 = torch.multinomial(p2_dist, num_samples=1, replacement=True).squeeze(1)
        mut_period, mut_offset, mut_repeat, mut_default = self.random_init_pop(_children_size)
        # print(mut_default)

        cross_mask = (torch.rand((_children_size, self.skill_dim), device=self.device)<0.5).type(torch.int64)
        default_cross_mask = (torch.rand((_children_size,), device=self.device)<0.5).type(torch.int64)
        mutation_mask = (torch.rand((_children_size, self.skill_dim), device=self.device)<self.mutation_prob).type(torch.int64)
        default_mutation_mask = (torch.rand((_children_size,), device=self.device)<self.mutation_prob).type(torch.int64)

        children_period, children_offset, children_repeat, children_default = self.random_init_pop(_children_size)
        children_period = pop_period[p1] * cross_mask + pop_period[p2] * (1-cross_mask)
        children_offset = pop_offset[p1] * cross_mask + pop_offset[p2] * (1-cross_mask)
        children_repeat = pop_repeat[p1] * cross_mask + pop_repeat[p2] * (1-cross_mask)
        children_default = pop_default[p1] * default_cross_mask + pop_default[p2] * (1 - default_cross_mask)

        children_period[mutation_mask] = mut_period[mutation_mask]
        children_offset[mutation_mask] = mut_offset[mutation_mask]
        children_repeat[mutation_mask] = mut_repeat[mutation_mask]
        children_default[default_mutation_mask] = mut_default[default_mutation_mask]

        new_period = torch.vstack((elites_period, children_period))
        new_offset = torch.vstack((elites_offset, children_offset))
        new_repeat = torch.vstack((elites_repeat, children_repeat))
        new_default = torch.concat((elites_default, children_default))
        new_period[torch.arange(self.pop_size), new_default] = 1
        new_offset[torch.arange(self.pop_size), new_default] = 0
        new_repeat[torch.arange(self.pop_size), new_default] = 1
        return new_period, new_offset, new_repeat, new_default

    def gen_skills_from_pop(self, pop_period, pop_offset, pop_repeat, pop_default):
        size = pop_period.shape[0]
        pop_period[torch.arange(size), pop_default] = 1
        pop_offset[torch.arange(size), pop_default] = 0
        pop_repeat[torch.arange(size), pop_default] = 1
        pop_period = pop_period.cpu().numpy()
        pop_offset = pop_offset.cpu().numpy()
        pop_repeat = pop_repeat.cpu().numpy()
        pop_skills = []
        for pop_idx in range(size):
            skills = np.zeros(self.ep_len, dtype=int)
            skill_idx_order = np.argsort(pop_period[pop_idx])
            for skill_idx in skill_idx_order:
                indices = np.arange(pop_offset[pop_idx, skill_idx], self.ep_len, pop_period[pop_idx, skill_idx])
                repetition = np.repeat(np.arange(0, pop_repeat[pop_idx, skill_idx]), len(indices)).reshape(-1, len(indices))
                indices = (indices+repetition).T.flatten()
                indices[indices>=self.ep_len]=self.ep_len-1
                skills[indices] = skill_idx
            pop_skills.append(skills)
        return pop_skills

    def ask(self, seed_step=False):
        metas = []
        pop_skills = self.gen_skills_from_pop(self.pop_period, self.pop_offset, self.pop_repeat, self.pop_default)
        for i in range(self.pop_size):
            metas.append(self.gen_meta_from_skills(pop_skills[i]))
        return metas

    def tell(self, step_fitness: torch.Tensor):
        if not isinstance(step_fitness, torch.Tensor):
            step_fitness = torch.Tensor(step_fitness).to(self.device)
        assert step_fitness.shape==(self.pop_size, self.ep_len)
        self.step_fitness = step_fitness.to(self.device)
        self.ep_fitness = self.step_fitness.sum(axis=1)
        top1 = torch.topk(self.ep_fitness, k=1)
        top_fitness, top_index = float(top1.values), top1.indices
        # self._pop_top_performer = self._pop[top_index].squeeze(0)
        self._pop_top_performer = \
            self.pop_period[top_index], self.pop_offset[top_index], self.pop_repeat[top_index], self.pop_default[top_index]
        if top_fitness> self._top_fitness:
            self._top_fitness = top_fitness
            self._top_performer = self._pop_top_performer

        self.pop_period, self.pop_offset, self.pop_repeat, self.pop_default \
            = self.generate_new_pop(self.pop_period, self.pop_offset, self.pop_repeat, self.pop_default, self.ep_fitness)
    
    def eval_log_wandb(self, global_step, global_frame):
        wandb.log({
            'optim/pop_repeat': wandb.Histogram(
                list(self.pop_repeat.cpu().numpy().flatten()),
                num_bins=50,
            ),
            'optim/pop_offset': wandb.Histogram(
                list(self.pop_offset.cpu().numpy().flatten()),
                num_bins=50,
            ),
            'optim/pop_default': wandb.Histogram(
                list(self.pop_default.cpu().numpy().flatten()),
                num_bins=50,
            ),
            'optim/pop_period': wandb.Histogram(
                list(self.pop_period.cpu().numpy().flatten()),
                num_bins=50,
            ),
            'optim/pop_log10_period': wandb.Histogram(
                list(np.log10(self.pop_period.cpu().numpy().flatten())),
                num_bins=50,
            ),
            'optim/pop_top_period': wandb.Histogram(
                list(self._top_performer[0].cpu().numpy().flatten()),
                num_bins=50,
            ),
            'global_step': global_step,
            'global_frame': global_frame,
            })