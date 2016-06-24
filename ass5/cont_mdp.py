import mdptoolbox
import numpy as np
import scipy.stats

# number of states required for two decimal precision
STATE_COUNT = 100


def index_to_state(index):
    return index / 100


def state_to_index(state):
    return state * 100


def get_terrain_type_for_state(start_points, state):
    start_points.append(1)
    terrain_types = [0, 1, 2, 3, 4] # always fixed

    for i in range(len(start_points) - 1):
        if start_points[i] <= state < start_points[i + 1]:
            return terrain_types[i]
    # if terrain is not found in the for loop above then the
    # agent has reached the goal state
    return 4


def get_probability(action, s_index, s_prime_index, movement_mean, movement_std,
                    terrain_start_points):

    s = index_to_state(s_index)
    terrain = get_terrain_type_for_state(terrain_start_points, s)

    mean = movement_mean[terrain][action]
    std = movement_std[terrain][action]
    #TODO: calculate probability from mean and std
    dist = (s_prime_index - s_index) / 100
    #z_score = (dist - mean) / std
    return scipy.stats.norm(mean, std).pdf(dist)


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
            for s_prime in xrange(STATE_COUNT):
                P[action][s][s_prime] = get_probability(action, s, s_prime,
                                                        movement_mean,
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
    print 'done with the probability matrix'

    R = get_rewards()

    print 'got the reward'
    #import ipdb; ipdb.set_trace()
    result = mdptoolbox.mdp.QLearning(P, R, 0.1)
    print result



def main():
    numActions = 8
    terrainStartPoint = [0.0, 0.2, 0.4, 0.6, 0.8]

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

    print solve(numActions, terrainStartPoint, movementMean, movementSD, sampleLocations)


if __name__ == '__main__':
    main()


