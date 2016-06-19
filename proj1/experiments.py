import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_walk import get_new_episode
from random_walk import td_lambda

MAX_STATE_LEN = 7
MAX_ITERATION = 100

actual_z = [0, 1/6., 1/3., 1/2., 2/3., 5/6., 1.0]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_on_traning_set(sequences, w, _lambda, alpha):
    """offline training"""

    w_accumulator = np.zeros(MAX_STATE_LEN)

    for sequence in sequences:
        X, z = sequence
        dw = td_lambda(X, z, w, _lambda, alpha, MAX_STATE_LEN)
        #print dw
        w_accumulator += dw

    return w_accumulator #/ len(sequence[0])


def get_traning_sets(traning_set_count=100, sequence_count=10):
    trainsets = []
    for ti in range(traning_set_count):
        sequences = [get_new_episode() for _ in range(sequence_count)]
        trainsets.append(sequences)
    return trainsets


def exp_1():
    alpha = 0.1

    # generate training sets
    print "Generate trainsets"

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    errors = []
    trainsets = get_traning_sets()

    for _lambda in lambdas:
        for trainset in trainsets:
            w = np.zeros(MAX_STATE_LEN)
            w[:] = 0.5
            w[0] = 0
            w[1] = 1

            i = 0
            while np.linalg.norm(w, np.inf) > 0.1 and i < MAX_ITERATION:
                w = train_on_traning_set(trainset, w, _lambda, alpha)
                i += 1

        predicted = np.array(w)
        #predicted = np.insert(predicted, 0, 0.0)
        #predicted = np.append(predicted, 1.0)

        #print "Predicted:"
        #print predicted

        print "Actual"
        print actual_z

        error = rmse(predicted, actual_z)
        print "Error", error

        print "Final weights"
        print w

        errors.append(error)

    columns = ["Lambda", "ERROR"]
    df = pd.DataFrame({"Lambda":lambdas, "ERROR":errors}).set_index("Lambda")
    ax = df.plot(title="Figure 3. TD(Lambda)", fontsize=12)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("ERROR")
    fig = ax.get_figure()
    plt.text(lambdas[-1]-0.2, errors[-1], "Widrow-Hoff")
    fig.savefig("figure3.png")
    plt.show()


if __name__ == '__main__':
    exp_1()