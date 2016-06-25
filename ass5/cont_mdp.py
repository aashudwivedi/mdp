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

    mean = movement_mean[terrain][action]
    std = movement_std[terrain][action]

    s_primes = np.arange(0, STATE_COUNT)

    probs = scipy.stats.norm(mean, std).pdf(s_primes)
    # normalize the probabilities
    probs = probs / probs.sum()
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


def solve(num_action, terrain_start_points, movement_mean, movement_std,
          sample_locs):
    P = get_transition_probability_matrix(action_count=num_action,
                                          terrain_start_points=terrain_start_points,
                                          terrain_types=[0, 1, 2, 3, 4],
                                          movement_mean=movement_mean,
                                          movement_std=movement_std)
    R = get_rewards()

    learner = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    learner.run()
    return learner.policy


def descretize(x, precision=100):
    """ Convert a array of floats between 0-1 into a array of int between 0-100
    """
    return (np.asarray(x, dtype=float) * precision).astype(int)


def main():
    numActions = 8
    terrainStartPoint = [0.0, 0.2, 0.4, 0.6, 0.8, 1]

    movementMean=[
        [-0.002,0.049,0.076,0.008,0.161,0.175,0.124,0.159],
        [0.129,0.169,0.168,0.059,-0.047,0.142,-0.07,0.188],
        [-0.052,0.099,-0.067,0.172,0.081,-0.095,0.121,-0.095],
        [0.142,0.025,-0.055,0.187,-0.099,0.125,0.056,0.024],
        [-0.072,0.125,0.076,0.07,-0.069,0.032,-0.044,-0.094]]

    movementSD=[
        [0.077,0.062,0.064,0.042,0.057,0.07,0.063,0.055],
        [0.058,0.042,0.077,0.089,0.052,0.074,0.053,0.088],
        [0.057,0.071,0.051,0.063,0.072,0.075,0.056,0.042],
        [0.062,0.058,0.063,0.087,0.078,0.089,0.046,0.042],
        [0.064,0.053,0.051,0.075,0.087,0.091,0.048,0.047]]

    sampleLocations=[0.67,0.69,0.74,0.77,0.85,0.89]

    terrainStartPoint = descretize(terrainStartPoint)
    movementMean = descretize(movementMean, 1000)
    movementSD = descretize(movementSD, 1000)
    sampleLocations = descretize(sampleLocations)

    print solve(numActions, terrainStartPoint, movementMean, movementSD, sampleLocations)


if __name__ == '__main__':
    main()


