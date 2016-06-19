import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_walk import get_new_episode
from random_walk import td_lambda
from constants import MAX_STATES


MAX_ITERATIONS = 1000

TOLERANCE = 0.0001

actual_z = [0, 1/6., 1/3., 1/2., 2/3., 5/6., 1.0]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_on_traning_set(sequences, w, _lambda, alpha):
    """offline training"""

    w_accumulator = np.zeros(MAX_STATES)
    w_current = w

    i = 0
    while i < MAX_ITERATIONS:
        for sequence in sequences:
            X, z = sequence
            dw = td_lambda(X, z, w, _lambda, alpha, MAX_STATES)
            w_accumulator += dw

        w_current = w_accumulator
        if np.linalg.norm(w_current - w_accumulator, np.inf) < TOLERANCE:
            break
        i += 1
    return w_current


def get_traning_sets(traning_set_count=100, sequence_count=10):
    trainsets = []
    for ti in range(traning_set_count):
        sequences = [get_new_episode() for _ in range(sequence_count)]
        trainsets.append(sequences)
    return trainsets


def plot_erros(lambdas, errors):
    columns = ["Lambda", "ERROR"]
    df = pd.DataFrame({"Lambda":lambdas, "ERROR":errors}).set_index("Lambda")
    ax = df.plot(title="Figure 3. TD(Lambda)", fontsize=12)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("ERROR")
    fig = ax.get_figure()
    plt.text(lambdas[-1]-0.2, errors[-1], "Widrow-Hoff")
    fig.savefig("figure3.png")
    plt.show()


def exp_1():
    alpha = 0.1

    # generate training sets
    print "Generate trainsets"

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    errors = []
    trainsets = get_traning_sets()

    w_init = np.zeros(MAX_STATES)
    w_init[:] = 0.5
    w_init[0] = 0
    w_init[1] = 1

    for _lambda in lambdas:

        for trainset in trainsets:

            w_precdicted = train_on_traning_set(trainset, w_init, _lambda, alpha)

        #predicted = np.array(w)
        # predicted = np.insert(predicted, 0, 0.0)
        # predicted = np.append(predicted, 1.0)

        #print "Predicted:"
        #print predicted

        print "Actual"
        print actual_z

        error = rmse(w_precdicted, actual_z)
        print "Error", error

        print "Final weights"
        print w_precdicted

        errors.append(error)
    return lambdas, errors


def exp2():
    pass


if __name__ == '__main__':
    plot_erros(*exp_1())