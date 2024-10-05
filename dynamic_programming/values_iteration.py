import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for _ in range(max_iter):
        prev_val = values.copy()
        for state in range(mdp.observation_space.n):
            action_values = []
            for action in range(mdp.action_space.n):
                next_state, reward, done = mdp.P[state][action]
                total = reward + gamma * prev_val[next_state]
                action_values.append(total)
            values[state] = max(action_values)
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-8,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    # BEGIN SOLUTION
    values = np.zeros((env.height, env.width))
    delta = float("inf")
    for t in range(max_iter):
        prev_val = values.copy()
        delta = 0
        for row in range(env.height):
            for col in range(env.width):
                env.current_position = (row, col)
                action_values = []
                for action in range(env.action_space.n):
                    next_state, reward, _, _ = env.step(action, make_move=False)
                    next_row, next_col = next_state
                    total = (reward + gamma * prev_val[next_row, next_col]) * env.moving_prob[row, col, action]
                    action_values.append(total)
                values[row, col] = max(action_values)
                delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
        if delta < theta:
            break
    return values
    


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        current_sum = 0
        next_states = env.get_next_states(action=action)
        for next_state, reward, probability, _, _ in next_states:
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-8,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    delta = float("inf")
    for _ in range(max_iter):
        prev_val = values.copy()
        delta = 0
        for row in range(env.height):
            for col in range(env.width):
                delta = value_iteration_per_state(env, values, gamma, prev_val, delta)
                env.current_position = (row, col)
        if delta < theta:
            break
    return values
