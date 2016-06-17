import json
import numpy as np

from collections import OrderedDict


def generate_mdp(T, R, gamma=0.75, num_states=30):
    """Generate an MDP from the provided arrays and gamma value."""
    act1 = {"id": 0, "transitions": []}
    act2 = {"id": 1, "transitions": []}
    names = ("id", "probability", "reward", "to")
    init_vals = (0, 0, 0, 0)
    ids = [i for i in range(num_states)]
    mdp = OrderedDict()
    mdp["gamma"] = gamma
    mdp["states"] = [{
        "id": id_num,
        "actions": [
            OrderedDict(sorted(act1.items(), key=lambda t: t[0])),
            OrderedDict(sorted(act2.items(), key=lambda t: t[0]))
        ]
    } for id_num in ids]
    for i in range(num_states):
        mdp["states"][i]["actions"][0]["transitions"] = [
            OrderedDict(zip(names, init_vals)) for _ in range(num_states)
        ]
        mdp["states"][i]["actions"][1]["transitions"] = [
            OrderedDict(zip(names, init_vals)) for _ in range(num_states)
        ]
        for j in range(num_states):
            mdp["states"][i]["actions"][0]["transitions"][j]["id"] = j
            mdp["states"][i]["actions"][1]["transitions"][j]["id"] = j
            # print i,j, T[0, i, j]
            mdp["states"][i]["actions"][0]["transitions"][j]["probability"] = round(
                T[0, i, j], 2)
            mdp["states"][i]["actions"][1]["transitions"][j]["probability"] = round(
                T[1, i, j], 2)
            mdp["states"][i]["actions"][0]["transitions"][j]["reward"] = R[0, i, j]
            mdp["states"][i]["actions"][1]["transitions"][j]["reward"] = R[1, i, j]
            mdp["states"][i]["actions"][0]["transitions"][j]["to"] = j
            mdp["states"][i]["actions"][1]["transitions"][j]["to"] = j

    ans = json.dumps(mdp, separators=(',', ':'))
    return ans

    
def get_solution_mdp():
    """Generate arrays to feed to the MDP parser."""
    states = 30
    actions = 2

    T = np.zeros(shape=(actions, states, states))
    R = np.zeros(shape=(actions, states, states))

    # actions from state 0
    T[0, 0, 0] = 0.5
    T[0, 0, 1] = 0.5

    for i in range(1, states - 1):
        T[0, i, i + 1] = 0.5
        T[0, i, i] = 0.5

        T[1, i, i] = 1

        R[0, i, i + 1] = 0
        R[0, i, i] = 0

        R[1, i, i] = 0.0000001 * i

    T[0, states - 1, states - 1] = 1
    T[1, states - 1, states - 1] = 1
    R[1, states - 1, states - 1] = 30
    R[0, states - 1, states - 1] = 30

    return generate_mdp(T, R, num_states=states)


def get_sample_mdp2():
    states = 2
    actions = 2

    T = np.zeros(shape=(actions, states, states))
    R = np.zeros(shape=(actions, states, states))

    T[0, 0, 0] = 0.5
    T[0, 0, 1] = 0.5
    T[0, 1, 1] = 1
    T[1, 1, 1] = 1

    R[0, 0, 0] = 0
    R[0, 0, 1] = 0
    R[0, 1, 1] = 0
    R[1, 1, 1] = 0


def get_sample_mdp():
    states = 2
    actions = 2

    T = np.zeros(shape=(actions, states, states))
    R = np.zeros(shape=(actions, states, states))

    T[0, 0, 0] = 0.5
    T[0, 0, 1] = 0.5
    T[0, 1, 1] = 1
    T[1, 1, 1] = 1

    R[0, 0, 0] = 0
    R[0, 0, 1] = 0
    R[0, 1, 1] = 0
    R[1, 1, 1] = 0

    return generate_mdp(T, R, num_states=states)


if __name__ == '__main__':
    # T, R = generate_arrays()

    print(get_solution_mdp())
