import mdptoolbox
import numpy as np
import scipy.stats

# number of states required for two decimal precision
STATE_COUNT = 100
terrain_types = [0, 1, 2, 3, 4]  # always fixed


def get_terrain_type_for_state(start_points, state):

    for i in range(len(start_points) - 1):
        if start_points[i] <= state < start_points[i + 1]:
            return terrain_types[i]
    # if terrain is not found in the for loop above then the
    # agent has reached the goal state
    return 4


def get_mean_to_state(current_state, mean):
    final_state = current_state + mean
    return final_state


def get_normalized_probability_dist(mean, std, samples):
    probs = scipy.stats.norm(mean, std).pdf(samples)
    # normalize the probabilities
    return probs / probs.sum()


def get_probabilities(action, s, movement_mean, movement_std,
                      terrain_start_points):
    """
    calculates probabilities for all the states s_prime
    :param action:
    :param s_index:
    :param movement_mean:
    :param movement_std:
    :param terrain_start_points:
    :return:
    """
    terrain = get_terrain_type_for_state(terrain_start_points, s)

    mean = get_mean_to_state(s, movement_mean[terrain][action])
    std = movement_std[terrain][action]

    s_primes = np.arange(0, STATE_COUNT)

    # normalize the probabilities
    probs = get_normalized_probability_dist(mean, std, s_primes)
    #import ipdb; ipdb.set_trace()
    return probs


def get_transition_probability_matrix(action_count, terrain_start_points,
                                      terrain_types, movement_mean,
                                      movement_std):
    """
    Transition probability is a 3d matrix with
    dimen1 representing action,
    dimen2 representing current state s
    dimen3 representing next state s'
    :return:
    """

    P = np.zeros((action_count, STATE_COUNT, STATE_COUNT))

    for action in xrange(action_count):
        for s in xrange(STATE_COUNT):
                P[action][s] = get_probabilities(action, s, movement_mean,
                                                 movement_std,
                                                 terrain_start_points)
    return P


def get_rewards():
    """
    Returns the reward matrix which is a one-d matrix
    It contains the reward for each state
    """
    # one last state results in the reward of 1000, -1 for every other state
    R = np.ones(STATE_COUNT) * -1
    R[-1] = 1000
    return R


def solve(movement_mean, movement_std, sample_locs):

    num_action = 8  # always fixed
    terrain_start_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1]  # always fixed

    terrain_start_pts = descretize(terrain_start_points)
    movement_mean = np.asarray(movement_mean, dtype=float) * 100
    movement_std = np.asarray(movement_std, dtype=float) * 100
    sample_locs = descretize(sample_locs)

    P = get_transition_probability_matrix(action_count=num_action,
                                          terrain_start_points=terrain_start_pts,
                                          terrain_types=[0, 1, 2, 3, 4],
                                          movement_mean=movement_mean,
                                          movement_std=movement_std)
    R = get_rewards()

    learner = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    learner.run()
    policy = np.asarray(learner.policy, dtype=int)
    sample_policy_str = ','.join([str(p) for p in policy[sample_locs].tolist()])

    print 'bestActions={%s}' % sample_policy_str


def descretize(x, precision=100):
    """ Convert a array of floats between 0-1 into a array of int between 0-100
    """
    return (np.asarray(x, dtype=float) * precision).astype(int)


def get_list_from_string(string):
    string = string.replace('{', '[')
    string = string.replace('}', ']')
    return eval(string)


def process_test_case(*args):
    return [get_list_from_string(arg) for arg in args]


def read_input_and_solve():
    f = open('input.txt')

    while True:
        movement_mean_str = f.readline()
        movement_std_str = f.readline()
        sample_location_str = f.readline()
        _  = f.readline() # blank line between test cases

        if movement_mean_str == '':
            return

        solve(*process_test_case(movement_mean_str, movement_std_str,
                                 sample_location_str))


def main():
    read_input_and_solve()


if __name__ == '__main__':
    main()


