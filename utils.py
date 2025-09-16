import numpy as np
from robosuite.environments import ALL_ENVIRONMENTS

def register_environment(env, name):
    if name not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS[name] = env

def strip_code(code_str):
    lines = code_str.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

def weighted_mean(elite_scores, elite_weights, softmax_temp, w_max = 0.5):
    shift = elite_scores.max()
    logits = (elite_scores - shift) / max(1e-8, softmax_temp)
    w = np.exp(logits)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        w = np.ones_like(elite_scores) / len(elite_scores)
    else:
        w /= w_sum
    
    if w.max() > w_max:
        w = np.clip(w, None, w_max)
        w /= w.sum()  # re-normalize to sum to 1
    weighted_mean_score = (elite_weights * w[:, None]).sum(axis=0)
    return weighted_mean_score

def mean(elite_weights):
    return np.mean(elite_weights, axis=0)

def sample_params(c_param, n_params):
    params = []
    if c_param != 0.0:
        while True:
            params = np.abs(np.random.normal(c_param, 0.25, size=n_params-1)).tolist()
            if ((params[0] < c_param and params[1] > c_param) or
                (params[0] > c_param and params[1] < c_param)) and all(x <= 1 for x in params):
                break
    else:
        params = np.abs(np.random.normal(c_param, 0.25, size=n_params-1)).tolist()
    params.append(c_param)
    return params

def same_best_weight(weights, rtol = 1e-02, atol= 1e-02):
    return all(
        np.allclose(weights[i], weights[i + 1], rtol, atol)
        for i in range(len(weights) - 1)
        )

import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)