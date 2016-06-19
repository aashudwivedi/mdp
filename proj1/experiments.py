import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_walk import get_new_episode
from random_walk import td_lambda
from constants import MAX_STATES


MAX_ITERATIONS = 1000

TOLERANCE = 0.000001

actual_z = [0, 1/6., 1/3., 1/2., 2/3., 5/6., 1.0]


def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_till_convergence(sequences, w, _lambda, alpha):
    w_prev = w

    i = 0
    while i < MAX_ITERATIONS:
        w_new = train_on_traning_set(sequences, w, _lambda, alpha)

        if np.linalg.norm(w_new - w_prev, np.inf) < TOLERANCE:
            break

        w_prev = w_new

        i += 1
    return w_new


def train_on_traning_set(sequences, w, _lambda, alpha, online=False):
    """offline training"""

    w_accumulator = np.zeros(MAX_STATES)

    for sequence in sequences:
        X, z = sequence

        if online:
            dw = td_lambda(X, z, w_accumulator, _lambda, alpha, MAX_STATES)
        else:
            dw = td_lambda(X, z, w, _lambda, alpha, MAX_STATES)

        w_accumulator += dw

    return w_accumulator


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
    alpha = 0.3

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

        rmses = []
        for trainset in trainsets:

            w_precdicted = train_till_convergence(trainset, w_init, _lambda, alpha)

            w_precdicted[0] = 0.0
            w_precdicted[-1] = 1.0
            rmse = get_rmse(w_precdicted, actual_z)
            rmses.append(rmse)

        print "Actual"
        print actual_z

        avg = np.array(rmses).mean()
        #error = get_rmse(w_precdicted, actual_z)
        print "Error", avg

        print "Final weights"
        print w_precdicted

        errors.append(avg)
    return lambdas, errors


def exp2():
    alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    lambdas = [0, 0.3, 0.8, 1]

    w_init = np.zeros(MAX_STATES)
    w_init[:] = 0.5
    w_init[0] = 0
    w_init[1] = 1

    trainsets = get_traning_sets()

    for _lambda in lambdas:
        for alpha in alpha_values:
            for trainset in trainsets:
                w_predicted = train_till_convergence(trainset, w_init, _lambda,
                                                     alpha)
                error = rmse(w_predicted, actual_z)





if __name__ == '__main__':
    plot_erros(*exp_1())