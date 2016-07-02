import numpy as np


def string_to_np(string):
    string = string.replace('{', '[')
    string = string.replace('}', ']')
    input_list = eval(string)
    return np.asarray(input_list, dtype=int)


def read_input_and_solve():
    f = open('input.txt')

    try:
        i  = 0
        while True:
            patron_count = int(f.readline())
            establishment = string_to_np(f.readline())
            fight_occured = string_to_np(f.readline())

            print '--' + str(i) + '--'
            process_test_case(patron_count, establishment, fight_occured)

            i += 1
            f.readline()
    except ValueError:
        pass


def get_h_space(patron_count):
    hypothesis = np.zeros((patron_count, patron_count))
    hypothesis[np.arange(patron_count), np.arange(patron_count)] = 1
    return hypothesis


def get_output_for_h(input_row, h):
    import ipdb; ipdb.set_trace()
    pass


def one_hot(location, size):
    a = np.zeros(size)
    a[location] = 1
    return a


def process_test_case(patron_count, establishments, fight_occured):
    hypothesises = [(x, y) for x in xrange(patron_count)
                    for y in range(patron_count) if x != y]

    results = []

    for e_index, e in enumerate(establishments):
        outputs = np.zeros(len(hypothesises))

        for h_index, h in enumerate(hypothesises):
            p_loc, i_loc = h
            p_present = (one_hot(p_loc, patron_count) * e.T).sum()
            i_present = ((one_hot(i_loc, patron_count)) * e.T).sum()

            output = 0
            if p_present:
                output = 0
            elif i_present:
                output = 1

            outputs[h_index] = output

        if np.all(outputs == 0):
            results.append('NO FIGHT')
        elif np.all(outputs == 1):
            results.append('FIGHT')
        else:
            results.append("I DON'T KNOW")
            wrong_hs = np.where(outputs != fight_occured[e_index])[0]
            hypothesises = [h for i, h in enumerate(hypothesises) if i not in wrong_hs]

    print ','.join(results)


def print_output(str):
    print '\n'.join(str.strip().split(','))


if __name__ == '__main__':
    read_input_and_solve()
