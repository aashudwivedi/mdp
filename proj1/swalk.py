#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calc_error(w, vec_len, actual_values):
    actual = np.zeros(vec_len)
    predicted = np.zeros(vec_len)
    for i in range(vec_len):
        c = chr(ord('B') + i)
        x = vector_for(c, vec_len)
        # print c, x
        predicted[i] = w.dot(x)
        actual[i] = actual_values[c]

    # predicted[0] = 0.0
    # predicted[-1] = 1.0
    # print "Actual", actual
    # print "Predicted", predicted
    return rmse(predicted, actual)

def gen_sequence(states, start_state, terminal_states):
    current_state = start_state
    seq = []
    seq.append(current_state)
    while current_state not in terminal_states:
        # print "Pick next state"
        next_state = random.choice(states[current_state])
        current_state = next_state
        seq.append(current_state)

    return seq

def vector_for(c, vec_len):
    x = np.zeros(vec_len)
    state_index = ord(c) - ord('B')
    x[state_index] = 1
    return x


def generate_vectors(seq, terminal_states, rewards, vec_len):
    X = []
    reward = 0
    for c in seq:
        if c in terminal_states:
            # print z
            # print "Terminated"
            reward = rewards[c]
            break
        x = vector_for(c, vec_len)
        X.append(x)
        # print x
    return (X, reward)

"""
Pt = wT.xt = ∑w(i)xt(i)               ; where i = 1 to length of vector
Δwt = α(Pt+1 − Pt) . ∑∇wPk            ; where k = 1 to t
w ← w + ∑Δwt                          ; where t = 1 to m
Δwt = α (Pt+1−Pt) ∑ lambda^t-k * ∇wPk
"""
def td_lambda(X, z, w, _lambda, alpha, vec_len):
    N = len(X)
    e = [[] for i in range(N)]
    p = np.zeros(N)

    pt = w.dot(X[0])

    # review: weights should be initialized to 0.5 according to the paper
    wt_sum = np.zeros(vec_len)
    w_old = w
    for i in range(N):
        p[i] = w.dot(X[i])
        p_diff = (p[i] - pt)
        p[i] = w.dot(X[i])
        pt = p[i]

        # Calculate Eligibility Trace:
        # e(t+1) = partial_derivative(Pt+1) + lambda * e(t)
        #
        # here partial derivate of Pk is xk itself
        if i == 0:
            e[i] = X[i]
        else:
            e[i] = X[i] + _lambda * e[i-1]

        delta_wt = alpha * p_diff * e[i]
        w += delta_wt
        # wt_sum = wt_sum + delta_wt
        # print delta_wt

    # Pm+1
    # ez = (e[N-1] + _lambda * e[N-1])
    wt_sum += alpha * (z - pt) * e[N-1]
    return w_old + wt_sum


start_state = "D"
terminal_states = ["A", "G"]
states = {
    "D": ["C", "E"],
    "C": ["B", "D"],
    "B": ["A", "C"],
    "E": ["D", "F"],
    "F": ["E", "G"],
    "A": None,
    "G": None
}

rewards = {"A":0, "B":0, "C":0, "D":0, "E":0, "F":0, "G":1.0,}

actual_values = {"A":0, "B": 1/6., "C": 1/3., "D": 1/2., "E": 2/3., "F": 5/6., "G": 1.0}

actual_z = [0, 1/6., 1/3., 1/2., 2/3., 5/6., 1.0]

vec_len = 5

seq = ['D', 'C', 'B', 'A']
# seq = gen_sequence(states, start_state, terminal_states)

_lambda = 0.3
alpha = 0.4

w = np.zeros(vec_len)

# generate training sets
print "Generate trainsets"

def print_array(w):
    print map(lambda x: "%.3f" % x, w)

trainsets = []
for ti in range(100):
    sequences = [
        gen_sequence(states, start_state, terminal_states) for si in range(10)
    ]
    trainsets.append(sequences)

print "Run Experiment 1"

lambda_choices = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
errors = []
for _lambda in lambda_choices:
    for ti in range(100):
        sequences = trainsets[ti]
        w = np.zeros(vec_len)
        w_accumulator = np.zeros(vec_len)
        s_count = 0
        for si in range(10):
            s_count += 1
            sequence = sequences[si]
            # print "Sequence set", ti
            # print sequence
            X, z = generate_vectors(sequence, terminal_states, rewards, vec_len)
            # print X, z
            wt_deltas = td_lambda(X, z, w, _lambda, alpha, vec_len)
            w_accumulator += wt_deltas
            # print "B", w_accumulator
        w += w_accumulator  # / s_count
        print map(lambda x: "%.3f" % x, w)

    predicted = np.array(w)
    predicted = np.insert(predicted,0,0.0)
    predicted = np.append(predicted,1.0)
    print "Predicted"
    print_array(predicted)

    print "Actual"
    print_array(actual_z)

    error = rmse(predicted, actual_z)
    print "Error", error

    print "Final weights"
    print_array(w)
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