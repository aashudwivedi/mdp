import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_walk import get_new_episode
from random_walk import td_lambda

MAX_STATE_LEN = 5

actual_z = [0, 1/6., 1/3., 1/2., 2/3., 5/6., 1.0]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def exp_1():
    _lambda = 0.3
    alpha = 0.2

    w = np.zeros(MAX_STATE_LEN)

    # generate training sets
    print "Generate trainsets"

    trainsets = []
    for ti in range(100):
        sequences = [get_new_episode() for _ in range(10)]
        trainsets.append(sequences)

    print "Run Experiment 1"

    lambda_choices = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    errors = []

    for _lambda in lambda_choices:
        for ti in range(100):
            sequences = trainsets[ti]
            w = np.zeros(MAX_STATE_LEN)
            w_accumulator = np.zeros(MAX_STATE_LEN)
            s_count = 0

            for si in range(10):
                s_count += 1
                sequence = sequences[si]
                # print "Sequence set", ti
                # print sequence
                X, z = sequence
                #print X, z
                wt_deltas = td_lambda(X, z, w, _lambda, alpha, MAX_STATE_LEN)
                w_accumulator += wt_deltas
                # print "B", w_accumulator
            w += w_accumulator  # / s_count
            print map(lambda x: "%.3f" % x, w)

        predicted = np.array(w)
        predicted = np.insert(predicted,0,0.0)
        predicted = np.append(predicted,1.0)
        print "Predicted"
        print predicted

        print "Actual"
        print actual_z

        error = rmse(predicted, actual_z)
        print "Error", error

        print "Final weights"
        print w
        errors.append(error)

    columns = ["Lambda", "ERROR"]
    df = pd.DataFrame({"Lambda":lambda_choices, "ERROR":errors}).set_index("Lambda")
    ax = df.plot(title="Figure 3. TD(Lambda)", fontsize=12)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("ERROR")
    fig = ax.get_figure()
    plt.text(lambda_choices[-1]-0.2, errors[-1], "Widrow-Hoff")
    fig.savefig("figure3.png")
    plt.show()


if __name__ == '__main__':
    exp_1()