import utils
import json
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
        pop_size: int = 40,
        elite_frac: float = 0.2,
        top_frac: float = 0.2,
        sigma: float = 0.5,
        alpha: float = 0.6,
        beta: float = 0.2,
        pop_decay_rate: float = 0.97,
        pop_min: int = 8,
        elite_min: int = 2,
        n_lambdas = 3,
        init_lambda = 0.5,
        drop = 0,
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
        self.n_lambdas = n_lambdas
        self.lambdas = None
        self.init_lambda = init_lambda
        self._lambda = init_lambda 
        self.drop = drop
        
    def init(self):
        #self.llm.generate_irreversible_events()
        #if self.grf: self.llm.generate_reward()
        #self.llm.generate_preference_setup()
        self._lambda = self.init_lambda
        self.lambdas = utils.sample_params(self._lambda , self.n_lambdas)
        self.weight_dim = self.agent.get_weights_dim()
        init_best_weight = 0 * np.random.randn(self.weight_dim)
        self.mean_weight = init_best_weight
        self.best_weight = self.mean_weight
        n_elite_init = max(int(self.pop_size * self.elite_frac), self.elite_min)
        self.n_top = max(int(max(n_elite_init, self.elite_min) * self.top_frac), 1)
        self.top_weights = [
            self.mean_weight + self.sigma * np.random.randn(self.weight_dim)
            for _ in range(self.n_top)
        ]
    
    def log(self, iter):
        print(f"Episode: {iter}")
        print(f"Lambdas: {self.lambdas}")
        print(f"Drops: {self.drop}")
    
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
        _return = []
        best_returns = [-np.inf] * self.n_lambdas
        best_weights = np.stack([np.zeros(self.weight_dim).copy() for _ in range(self.n_lambdas)])
        for i, weight in enumerate(weights_pop):
            returns,_, drop= self.agent.evaluate(weight, self.n_steps, self.lambdas)
            if drop:
                self.save_drop(weight)
                self.drop += 1
            for n, ret in enumerate(returns):
                if ret > best_returns[n]:
                    best_returns[n] = ret
                    best_weights[n] = weight
            _return.append(returns[-1])
            print(f"return {i}: {(returns[-1])}")
        _return = np.asarray(_return, dtype=float)
        return _return, best_weights

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

    def feedback(self, weights):
        episode_metrics = []
        for w, weight in enumerate(weights):
            _, met, _ = self.agent.evaluate(weight, self.n_steps, [self.lambdas[w]], render = True, video_i=w)
            episode_metrics.append(met)
        metrics_text = "\n\n".join(f"Trajectory {i}:\n{json.dumps(t, indent=self.n_lambdas)}" for i, t in enumerate(episode_metrics))

        best_idx = self.llm.generate_preference(metrics_text)
        best_lambda = self.lambdas[best_idx]
        self._lambda = self._lambda + 0.15 * (best_lambda-self._lambda)
        self.lambdas = utils.sample_params(self._lambda, self.n_lambdas)

    def decay(self):
        self.pop_size = max(self.pop_min, int(round(self.pop_size * self.pop_decay_rate)))

    def save(self):
        np.savetxt(Path("weights") / "weights.txt", self.best_weight)
    
    def save_drop(self, weight):
        np.savetxt(Path("weights") / "weights_drop.txt", weight)

    def train(self):
        self.init()
        for i in range(self.n_its):
            self.log(i)
            n_elite, weights_pop = self.populate()
            returns, best_weights = self.evaluate(weights_pop)
            elite_weights,_ = self.elitism(returns, weights_pop, n_elite)
            self.update(elite_weights)
            #if i%2==0:
            #    if not utils.same_best_weight(best_weights):
            #        self.feedback(best_weights)
            self.decay()
            self.save()