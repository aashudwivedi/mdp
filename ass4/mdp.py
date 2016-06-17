import numpy
import simplejson
import pprint


def three_state():
    gamma = 1
    states_count = 3
    rewards = [1, -1 , 2]

    actions = {
        0: [0, 1],
        1: [2],
        2: [0],
    }
    return gamma, states_count, rewards, actions


def six_state():
    gamma = 0.75
    states_count = 6
    rewards = [1, -1, -1, 10, -1, 100]

    actions = {
        0: [0, 1],
        1: [2],
        2: [3],
        3: [4, 0, 3],
        4: [5],
        5: [0],
    }
    return gamma, states_count, rewards, actions


def get_dict(gamma, states_count, rewards, actions):
    result = {
        'gamma': gamma,
        'states': [],
    }

    for i in range(states_count):
        state = {
            'id': i,
            'actions': [],
        }
        for j, action in enumerate(actions[i]):
            action_dict = {
                'id': j,
                'transitions': [
                    {
                      "id": 0,
                      "probability": 1,
                      "reward": rewards[action],
                      "to": action
                    },
                ],
            }
            state['actions'].append(action_dict)
        result['states'].append(state)
    return result


def get_json(gamma, states_count, rewards, actions):
    return simplejson.dumps(get_dict(gamma, states_count, rewards, actions))


def main():
    return get_json(*six_state())


if __name__ == '__main__':
    pprint.pprint(main())