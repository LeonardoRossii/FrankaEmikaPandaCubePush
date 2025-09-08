import numpy as np
import math

def weighted_mean(elite_weights, elite_weights_scores):
    elite_weights = np.array(elite_weights) 
    elite_weights_scores = np.array(elite_weights_scores)
    elite_weights_scores_normalized = elite_weights_scores / np.sum(elite_weights_scores)
    best_weight_new = np.average(elite_weights, axis=0, weights=elite_weights_scores_normalized)
    return best_weight_new

def cem(agent, n_training_iterations=30, max_n_timesteps=250, gamma=0.99, pop_size=30, n_elite=4, sigma=0.05, alpha= 0.85, beta= 0.2):
    
    mean_weight = np.zeros(agent.get_weights_dim())
    best_weight = mean_weight

    for i_iteration in range(1, n_training_iterations+1):
        
        rewards = []

        weights_pop = [mean_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size-1)]        
        weights_pop.append(best_weight)
        
        for i in range(len(weights_pop)):
            reward = agent.evaluate(weights_pop[i], max_n_timesteps, gamma)
            rewards.append(reward)
            print(f"return {i}: {reward}")

        elite_idxs = np.array(rewards).argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        elite_weights_scores = [np.array(rewards)[i] for i in elite_idxs]
        best_weight = elite_weights[n_elite-1]
        best_reward = elite_weights_scores[n_elite-1]

        mean_weight_new = weighted_mean(elite_weights, elite_weights_scores)
        mean_weight= alpha*mean_weight_new + (1-alpha)*mean_weight
        sigma_new = np.array(elite_weights).std(axis=0)
        sigma= beta*sigma_new + (1-beta)*sigma
        
        mean_reward = agent.evaluate(mean_weight,max_n_timesteps, gamma=1.0)
        if mean_reward >= best_reward:
            best_weight = mean_weight
        
        pop_size = max(8, pop_size-1)
        n_elite  = max(2, math.ceil(pop_size/5))

        if i_iteration == n_training_iterations:
            np.savetxt('policy.txt', best_weight)
    return None