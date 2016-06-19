import random
import numpy as np
from constants import action_map, MAX_STATES, FINAL_STATES

def char_state(i):
    return chr(ord('A') + i)


def print_array(w):
    print map(lambda x: "%.3f" % x, w)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def execute_random_action(current_state):
    action = action_map.get(random.choice(['left', 'right']))
    new_state = current_state + action
    return new_state


def random_walk_generator():
    current_state = 3  # state D
    next_state = current_state
    yield next_state
    while next_state not in [0, MAX_STATES - 1]:
        next_state = execute_random_action(current_state)
        current_state = next_state
        yield next_state


def get_numpy_episode_and_reward(episode):
    reward = 0 if episode[-1] == 0 else 1

    np_episode = np.zeros((len(episode), MAX_STATES))

    for i, state in enumerate(episode):
        np_episode[i][state - 1] = 1

    rewards = np.zeros(len(episode))
    rewards[-1] = reward
    return np_episode, rewards


def get_new_episode():
    return get_numpy_episode_and_reward(list(random_walk_generator()))


def get_state_vector(current_state):
    pass


def td_lambda(X, z, w,  lambda_val, alpha, total_states=MAX_STATES):
    """
    Args:
        :param X: list of episode vectors
        :param z: rewards
        :param w: current weights
        :param lambda_val: lambda
        :param alpha: alpha value
        :param total_states: total number of states (always 5)
    Return:
        :return: updated weights
    """
    N = len(X)
    e = np.zeros((N, total_states))

    pt = w.dot(X[0])
    dw = np.zeros(total_states)
    e[0] = X[0]

    for i in range(N):
        p_i = z[-1] if i == N - 1 else w.dot(X[i])
        p_diff = (p_i - pt)
        pt = p_i
        e[i] = X[i] + lambda_val * e[i-1]
        dw += alpha * p_diff * e[i]

    return w + dw