import random
import numpy as np

action_map = {
    'left': -1,
    'right': 1
}

FINAL_STATES = [0, 6]

MAX_STATES = 7

MAX_STATES_ACTUAL = 5


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
    episode = episode[:-1]
    np_episode = np.zeros((len(episode), MAX_STATES - 2))

    for i, state in enumerate(episode):
        np_episode[i][state - 1] = 1

    rewards = np.ravel(np.zeros((1, len(episode))))
    rewards[-1] = reward
    return np_episode, rewards


def get_new_episode():
    return get_numpy_episode_and_reward(list(random_walk_generator()))

def get_state_vector(current_state):
    pass


def td_lambda(X, z, w,  alpha, lambda_val, total_states=5):
    """
    Args:
        :param X: list of episode vectors
        :param z: rewards
        :param w: current weights
        :param alpha: alpha value
        :param lambda_val: lambda
        :param total_states: total number of states (always 5)
    Return:
        :return: updated weights
    """

    episode_len = X.shape[0]
    e = np.zeros((episode_len, total_states))

    p = np.zeros(episode_len)
    p_prev = w.dot(X[0])

    wt_sum = np.zeros(total_states)
    w_old = w
    for i in xrange(episode_len):
        p[i] = w.dot(X[i])
        p_diff = (p[i] - p_prev)
        p_prev = p[i]

        # etrace calculation
        if i == 0:
            try:
                e[i] = X[i]
            except ValueError:
                import ipdb; ipdb.set_trace()
        else:
            e[i] = X[i] + lambda_val * e[i - 1]

        delta_wt = alpha * p_diff * e[i]
        w += delta_wt

    wt_sum += alpha * (z[-1] - p_prev) * e[episode_len - 1]
    return w_old + wt_sum


def experiment_1(alpha):

    actual_z = [0, 1/6., 1/3., 1/2., 2/3., 5/6., 1.0]

    lambda_choices = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    errors = []
    for _lambda in lambda_choices:
        for ti in range(100):
            w = np.zeros(MAX_STATES_ACTUAL)
            w_accumulator = np.zeros(MAX_STATES_ACTUAL)
            s_count = 0
            for si in range(10):
                s_count += 1
                X, z = get_new_episode()
                # print X, z
                wt_deltas = td_lambda(X, z, w, alpha, _lambda)
                w_accumulator += wt_deltas
                # print "B", w_accumulator
            w += w_accumulator  # / s_count
            print map(lambda x: "%.3f" % x, w)

        predicted = np.array(w)
        predicted = np.insert(predicted,0,0.0)
        predicted = np.append(predicted,1.0)
        print "Predicted"
        print_array(predicted)

        print "Actual"
        print_array(actual_z)

        error = rmse(predicted, actual_z)
        print "Error", error

        print "Final weights"
        print_array(w)
        errors.append(error)


def main():
    _lambda = 0.3
    alpha = 0.4
    experiment_1(alpha)


if __name__ == '__main__':
    main()