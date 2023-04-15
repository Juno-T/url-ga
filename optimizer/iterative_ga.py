from typing import Union

import numpy as np
import torch

from optimizer.base import BaseSkillOptimizer
from agent.diayn import DIAYNAgent

class IterativeGA(BaseSkillOptimizer):
    def __init__(self,
        agent: DIAYNAgent,
        pop_size = 100,
        elite_size = 10,
        mutation_prob = 0.05,
        ep_len = 1000,
        strategy = 'one_cover',
        next_window_fitness_threshold = 0.05,
        num_window = 10,
        gamma = 0.7,
        device = 'cuda',
        **kwargs,
    ):
        super().__init__(agent, **kwargs)
        self.device = device
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.ep_len = ep_len

        self._children_size = pop_size-elite_size
        self._tril = torch.tril(torch.ones((ep_len,ep_len), dtype=torch.int64, device=self.device))


        self.next_window_fitness_threshold = next_window_fitness_threshold # If the fitness of top performer is less than 10% improve, iterate the window
        self.num_window = num_window
        assert self.ep_len%self.num_window==0
        self.window_step = self.ep_len//self.num_window
        self.gamma = gamma
        self.current_window = 1
        self.multinomial_dist = torch.tensor([(1-self.gamma)**i*self.gamma for i in range(self.num_window)][::-1], device=self.device)
        self.multinomial_dist[0] /= self.gamma

        self.generation = 0
        self._pop = self.random_init_pop()
        self._pop_top_performer = None
        self._top_performer = None
        self._top_fitness = -1e9
    
    @property
    def pop_top_performer(self):
        if self._pop_top_performer is None:
            return None
        return self.gen_meta_from_skills(self._pop_top_performer)
    
    @property
    def top_performer(self):
        if self._top_performer is None:
            return None
        return self.gen_meta_from_skills(self._top_performer)
    
    @property
    def top_fitness(self):
        if self._top_performer is None:
            return None
        return self._top_fitness
    
    @property
    def children_size(self):
        return self._children_size

    def random_init_pop(self):
        return torch.randint(self.skill_dim, (self.pop_size, self.ep_len), device=self.device)

    def generate_new_pop(self, pop, ep_fitness):
        elites = pop[torch.topk(ep_fitness, k=self.elite_size).indices]

        # Parents selection
        p1 = torch.multinomial(ep_fitness, num_samples=self._children_size, replacement=True)
        p2 = torch.multinomial(ep_fitness, num_samples=self._children_size, replacement=True)
        p1 = pop[p1]
        p2 = pop[p2]

        # Crossing over
        cross_window = torch.multinomial(self.multinomial_dist[:self.current_window], num_samples=self._children_size, replacement=True).to(self.device)
        cross_positions = cross_window*self.window_step + torch.randint(self.window_step, (self._children_size,)).to(self.device)
        crossing_over_mask = self._tril[cross_positions]
        children = p1*crossing_over_mask + p2*(1-crossing_over_mask)

        # Mutation
        mutation = torch.randint(self.skill_dim, children.shape, device=self.device)
        mutation_prob_scale = self.multinomial_dist.clone()
        mutation_prob_scale[self.current_window:] = 0
        mutation_prob_scale /= self.multinomial_dist[self.current_window-1]
        mutation_prob_scale = torch.repeat_interleave(mutation_prob_scale, self.window_step)
        mutation_mask = (torch.rand(children.shape, device=self.device)<self.mutation_prob*mutation_prob_scale).type(torch.int64)
        children = children*(1-mutation_mask) + mutation*mutation_mask
        new_pop = torch.vstack((elites, children))
        return new_pop

    def ask(self, seed_step=False):
        metas = []
        max_step = self.ep_len
        if not seed_step:
            max_step = self.current_window*self.window_step
        for i in range(len(self._pop)):
            metas.append(self.gen_meta_from_skills(self._pop[i][:max_step]))
        return metas

    def tell(self, step_fitness: torch.Tensor):
        if not isinstance(step_fitness, torch.Tensor):
            step_fitness = torch.Tensor(step_fitness).to(self.device)
        # assert step_fitness.shape==(self.pop_size, self.current_window*self.window_step)
        step_fitness = step_fitness[:, :self.current_window*self.window_step]
        self.step_fitness = step_fitness.to(self.device)
        self.ep_fitness = self.step_fitness.sum(axis=1)
        top1 = torch.topk(self.ep_fitness, k=1)
        pop_top_fitness, top_index = float(top1.values), top1.indices
        self._pop_top_performer = self._pop[top_index].squeeze(0)
        if pop_top_fitness < self._top_fitness*(1+self.next_window_fitness_threshold):
            self.current_window += 1
            self.current_window = min(self.current_window, self.num_window)

        if pop_top_fitness> self._top_fitness:
            self._top_fitness = pop_top_fitness
            self._top_performer = self._pop_top_performer

        self._pop = self.generate_new_pop(self._pop, self.ep_fitness)
        self.generation += 1

        data = {
            'optim/iterated_step': self.window_step*self.current_window,
            'optim/pop_top_fitness': pop_top_fitness,
            'optim/generation': self.generation
        }
        return data