import numpy as np


def string_to_np(string):
    string = string.replace('{', '[')
    string = string.replace('}', ']')
    input_list = eval(string)
    return np.asarray(input_list, dtype=int)


def process_test_case(patron_count, establishment, fight_occured):
    pass


def read_input_and_solve():
    f = open('input.txt')

    try:
        while True:
            patron_count = int(f.readline())
            establishment = string_to_np(f.readline())
            fight_occured = string_to_np(f.readline())

            print patron_count
            print establishment
            print fight_occured
    except ValueError:
        pass

if __name__ == '__main__':
    read_input_and_solve()