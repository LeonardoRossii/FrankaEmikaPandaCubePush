import numpy as np

def cem(
    agent,
    n_training_iterations: int = 30,
    max_n_timesteps: int = 250,
    gamma: float = 0.99,
    pop_size: int = 30,
    elite_frac: float = 0.2,
    top_frac: float = 0.2,
    sigma: float = 0.20,
    alpha: float = 0.90,
    beta: float = 0.25,
    softmax_temp: float = 1.0,
    sigma_min: float = 1e-3,
    pop_decay = 0.95,
    pop_min= 8,
    elite_min = 2,
):
    
    weight_dim = agent.get_weights_dim()
    mean_weight = 0 * np.random.randn(weight_dim)
    best_weight = mean_weight

    n_elite_init = max(int(pop_size * elite_frac), elite_min)
    n_top = max(int(max(n_elite_init, elite_min) * top_frac), 1)

    top_weights = [
        mean_weight + sigma * np.random.randn(weight_dim)
        for _ in range(n_top)
    ]

    best_rewards = []

    for i_iteration in range(1, n_training_iterations + 1):
        
        print(f"- Episode: {i_iteration}")

        n_elite = max(int(pop_size * elite_frac), elite_min)
        weights_pop = [
            mean_weight + sigma * np.random.randn(weight_dim)
            for _ in range(pop_size - len(top_weights))
        ]

        for w in top_weights:
            weights_pop.append(w)

        returns = []

        for i, weight in enumerate(weights_pop):
            return_ = agent.evaluate(weight, max_n_timesteps, gamma)
            returns.append(return_)
            print(f"return {i}: {return_}")

        returns = np.asarray(returns, dtype=float)
        elite_idxs = np.argsort(returns)[-n_elite:]
        print(elite_idxs)
        elite_weights = np.array([weights_pop[i] for i in elite_idxs])
        elite_scores = returns[elite_idxs]

        shift = elite_scores.max()
        logits = (elite_scores - shift) / max(1e-8, softmax_temp)
        w = np.exp(logits)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            w = np.ones_like(elite_scores) / len(elite_scores)
        else:
            w /= w_sum
        weighted_mean_score = (elite_weights * w[:, None]).sum(axis=0)

        n_top_iter = min(n_top, n_elite)
        top_weights = elite_weights[-n_top_iter:]
        best_weight = elite_weights[-1]
        best_reward = elite_scores[-1]

        mean_weight = alpha * weighted_mean_score + (1 - alpha) * mean_weight
        elite_std = np.std(elite_weights, axis=0)
        sigma = beta * elite_std + (1 - beta) * sigma

        sigma = np.maximum(sigma, sigma_min)
        best_rewards.append(best_reward)
    
        pop_size = max(pop_min, int(round(pop_size * pop_decay)))

        if i_iteration == n_training_iterations:
            np.savetxt('policy.txt', best_weight)
    return best_rewards