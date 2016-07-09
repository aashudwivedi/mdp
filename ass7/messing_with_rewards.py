import numpy as np
import mdptoolbox


def process_test_case(state_count, action_count, T, R, gamma):
    mdp = mdptoolbox.mdp.ValueIteration(T, R, gamma, epsilon=0.0000000000000001)
    mdp.run()

    print mdp.iter
    print ', '.join(['%.3f' % x for x in mdp.V])


def string_to_np(string):
    string = string.replace('{', '[')
    string = string.replace('}', ']')
    input_list = eval(string)
    return np.asarray(input_list, dtype=float)


def read_input_and_solve():
    f = open('input2.txt')

    try:
        i = 0
        while True:
            state_count = int(f.readline())
            action_count = int(f.readline())
            T = string_to_np(f.readline())
            R = string_to_np(f.readline())
            gamma = string_to_np(f.readline())

            T = np.swapaxes(T, 0, 1)
            R = np.swapaxes(R, 0, 1)

            print '-' * 10 + 'test case ' + str(i) + '--' * 10

            print 'numStates = %s' % str(state_count)
            print 'numActions = %s' % str(action_count)
            process_test_case(state_count, action_count, T, R, gamma)

            i += 1
            f.readline()
    except ValueError:
        pass


if __name__ == '__main__':
    read_input_and_solve()