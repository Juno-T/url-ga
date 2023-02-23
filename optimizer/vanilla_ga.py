from typing import Union

import numpy as np
import torch

from optimizer.base import BaseSkillOptimizer
from agent.diayn import DIAYNAgent

class VanillaGA(BaseSkillOptimizer):
    def __init__(self,
        agent: DIAYNAgent,
        pop_size = 100,
        elite_size = 10,
        mutation_prob = 0.05,
        ep_len = 1000,
        strategy = 'one_cover',
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
        cross_positions = torch.randint(self.ep_len, (self._children_size,))
        crossing_over_mask = self._tril[cross_positions]
        children = p1*crossing_over_mask + p2*(1-crossing_over_mask)

        # Mutation
        mutation = torch.randint(self.skill_dim, children.shape, device=self.device)
        mutation_mask = (torch.rand(children.shape, device=self.device)<self.mutation_prob).type(torch.int64)
        children = children*(1-mutation_mask) + mutation*mutation_mask
        new_pop = torch.vstack((elites, children))
        return new_pop

    def ask(self):
        metas = []
        for i in range(len(self._pop)):
            metas.append(self.gen_meta_from_skills(self._pop[i]))
        return metas

    def tell(self, step_fitness: torch.Tensor):
        if not isinstance(step_fitness, torch.Tensor):
            step_fitness = torch.Tensor(step_fitness).to(self.device)
        assert step_fitness.shape==(self.pop_size, self.ep_len)
        self.step_fitness = step_fitness.to(self.device)
        self.ep_fitness = self.step_fitness.sum(axis=1)
        top1 = torch.topk(self.ep_fitness, k=1)
        top_fitness, top_index = float(top1.values), top1.indices
        self._pop_top_performer = self._pop[top_index].squeeze(0)
        if top_fitness> self._top_fitness:
            self._top_fitness = top_fitness
            self._top_performer = self._pop_top_performer

        self._pop = self.generate_new_pop(self._pop, self.ep_fitness)