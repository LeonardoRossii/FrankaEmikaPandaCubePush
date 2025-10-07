import numpy as np
from pathlib import Path

class CEM:
    def __init__(
        self,
        agent,
        llm,
        reward_gen: bool = False,
        n_its: int = 30,
        n_steps: int = 250,
        randoms: int = 1,
        gamma: float = 0.99,
        pop_size: int = 30,
        elite_frac: float = 0.2,
        top_frac: float = 0.2,
        sigma: float = 0.5,
        alpha: float = 0.6,
        beta: float = 0.2,
        pop_decay_rate: float = 0.97,
        pop_min: int = 8,
        elite_min: int = 2,
    ):
        self.agent = agent
        self.llm = llm
        self.grf = reward_gen
        self.n_its = n_its
        self.n_steps = n_steps
        self.randoms = randoms
        self.gamma = gamma
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.top_frac = top_frac
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.pop_decay_rate = pop_decay_rate
        self.pop_min = pop_min
        self.elite_min = elite_min
        self.weight_dim = None
        self.mean_weight = None
        self.best_weight = None
        self.top_weights = None
        self.n_top = None
        
    def init(self):
        self.llm.generate_ie_with_image()
        if self.grf: self.llm.generate_reward()
        self.weight_dim = self.agent.get_weights_dim()
        init_best_weight = 0 * np.random.randn(self.weight_dim)
        self.mean_weight = init_best_weight
        self.best_weight = self.mean_weight
        n_elite_init = max(int(self.pop_size * self.elite_frac), self.elite_min)
        self.n_top = max(int(max(n_elite_init, self.elite_min) * self.top_frac), 1)
        self.top_weights = [self.mean_weight + self.sigma * np.random.randn(self.weight_dim) for _ in range(self.n_top)]
    
    def populate(self):
        n_elite = max(int(self.pop_size * self.elite_frac), self.elite_min)
        weights_pop = [
            self.mean_weight + self.sigma * np.random.randn(self.weight_dim)
            for _ in range(self.pop_size - len(self.top_weights))
        ]
        for w in self.top_weights:
            weights_pop.append(w)
        return n_elite, weights_pop

    def evaluate(self, weights_pop):
        returns = []
        for i, weight in enumerate(weights_pop):
            _return= self.agent.evaluate(weight, self.n_steps, self.gamma)
            returns.append(_return)
            print(f"return {i}: {(_return)}")
        returns = np.asarray(returns, dtype=float)
        return returns

    def elitism(self, returns, weights_pop, n_elite):
        elite_idxs = np.argpartition(returns, -n_elite)[-n_elite:]
        elite_weights = np.array([weights_pop[i] for i in elite_idxs])
        elite_scores = returns[elite_idxs]
        order = np.argsort(elite_scores)
        elite_weights = elite_weights[order]
        elite_scores = elite_scores[order]
        n_top_iter = min(self.n_top, n_elite)
        self.top_weights = [elite_weights[-j - 1].copy() for j in range(n_top_iter)]
        self.best_weight = elite_weights[-1].copy()
        return elite_weights, elite_scores

    def update(self, elite_weights):
        elite_mean = np.mean(elite_weights, axis=0)
        self.mean_weight = self.alpha * elite_mean + (1 - self.alpha) * self.mean_weight        
        elite_std = np.std(elite_weights, axis=0)
        self.sigma = self.beta * elite_std + (1 - self.beta) * self.sigma

    def decay(self):
        self.pop_size = max(self.pop_min, int(round(self.pop_size * self.pop_decay_rate)))

    def save(self):
        np.savetxt(Path("weights") / "weights.txt", self.best_weight)

    def train(self):
        self.init()
        for i in range(self.n_its):
            n_elite, weights_pop = self.populate()
            returns = self.evaluate(weights_pop)
            elite_weights,_ = self.elitism(returns, weights_pop, n_elite)
            self.update(elite_weights)
            self.decay()
            self.save()