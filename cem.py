import llm
import utils
import numpy as np
from pathlib import Path

def cem(
    agent,
    n_training_iterations: int = 25,
    max_n_timesteps: int = 250,
    gamma: float = 0.99,
    pop_size: int = 50,
    elite_frac: float = 0.25,
    top_frac: float = 0.2,
    sigma: float = 0.05,
    alpha: float = 0.5,
    beta: float = 0.2,
    sigma_min: float = 1e-3,
    pop_decay: float = 0.95,
    pop_min: int = 8,
    elite_min: int = 2,
    init_param: float = 0.5,
    n_params: int = 3,
    update_factor: float = 0.1,
    pref_freq = 2,
):  
    current_dir = Path(__file__).parent
    file_task_description_path = current_dir / "pmptpref.txt"
    with open(file_task_description_path, "r") as file:
        prompt = file.read().strip()

    c_param = init_param
    best_param = c_param
    params = utils.sample_params(c_param, n_params)

    weight_dim = agent.get_weights_dim()
    init_best_weight = 0 * np.random.randn(weight_dim)
    mean_weight = init_best_weight
    best_weight = mean_weight

    n_elite_init = max(int(pop_size * elite_frac), elite_min)
    n_top = max(int(max(n_elite_init, elite_min) * top_frac), 1)

    top_weights = [
        mean_weight + sigma * np.random.randn(weight_dim)
        for _ in range(n_top)
    ]

    drops = 0

    for i_iteration in range(0, n_training_iterations):
        print(f"- Episode: {i_iteration}")
        print(f"- Params:  {params}")
        
        returns = []
        best_returns_param = [-np.inf] * n_params
        best_weights_param = np.stack([init_best_weight.copy() for _ in range(n_params)])

        n_elite = max(int(pop_size * elite_frac), elite_min)
        
        weights_pop = [
            mean_weight + sigma * np.random.randn(weight_dim)
            for _ in range(pop_size - len(top_weights))
        ]

        for w in top_weights:
            weights_pop.append(w)

        for i, weight in enumerate(weights_pop):

            k_returns, drop = agent.evaluate(weight, params, max_n_timesteps, gamma)
            drops += int(drop)
            
            for j, k_ret in enumerate(k_returns):
                if k_ret > best_returns_param[j]:
                    best_returns_param[j] = k_ret
                    best_weights_param[j] = weight
            
            returns.append(k_returns[-1])
            print(f"return {i}: {(k_returns[-1])}")

        returns = np.asarray(returns, dtype=float)
        
        elite_idxs = np.argpartition(returns, -n_elite)[-n_elite:]
        elite_weights = np.array([weights_pop[i] for i in elite_idxs])
        elite_scores = returns[elite_idxs]

        order = np.argsort(elite_scores)
        elite_weights = elite_weights[order]
        elite_scores = elite_scores[order]

        n_top_iter = min(n_top, n_elite)
        top_weights = [elite_weights[-j - 1].copy() for j in range(n_top_iter)]
        best_weight = elite_weights[-1].copy()

        elite_mean = utils.weighted_mean(elite_scores, elite_weights, softmax_temp=1.0)
        mean_weight = alpha * elite_mean + (1 - alpha) * mean_weight
        elite_std = np.std(elite_weights, axis=0)
        sigma = beta * elite_std + (1 - beta) * sigma
        sigma = np.maximum(sigma, sigma_min)
    
        pop_size = max(pop_min, int(round(pop_size * pop_decay)))

        """if i_iteration%pref_freq==0 and not utils.same_best_weight(best_weights_param):
            best_index = llm.get_preference(agent, best_weights_param, 250, prompt)
            best_param = params[best_index]"""

        #c_param = c_param + update_factor*(best_param-c_param)
        # params = utils.sample_params(c_param, n_params)

        print(f"drops: {(drops)}")
        if i_iteration == n_training_iterations-1:
            np.savetxt('theta.txt', best_weight)
    
    return drops