import matplotlib as mp
import numpy as np
import pickle
import tools
import utils

def evaluate_policy(env, V, pi, gamma, theta):
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def bellman_update(env, V, pi, s, gamma):
    """Mutate ``V`` according to the Bellman update equation."""
    v_ = 0
    for a in env.A:
        transitions = env.transitions(s, a)
        for s_, (r, p) in enumerate(transitions):
            v_ = v_ + pi[s, a]*p*(r + gamma*V[s_])
    V[s] = v_


num_spaces = 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1
gamma = 0.9
theta = 0.1
V = evaluate_policy(env, V, city_policy, gamma, theta)

#print(V)
#tools.plot(V, city_policy)

def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi


def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    ### START CODE HERE ###
    Q = np.zeros(len(env.A))
    top = float("-inf")
    ties = []
    for a in env.A:
        transitions = env.transitions(s, a)
        for s_, (r, p) in enumerate(transitions):
            Q[a] += p*(r + gamma*V[s_])
    best_a = utils.argmax(Q)
    pi_ = np.zeros(len(env.A))
    pi_[best_a] = 1
    pi[s] = np.array(pi_)
    ### END CODE HERE ###


env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = policy_iteration(env, gamma, theta)

def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi


def bellman_optimality_update(env, V, s, gamma):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    ### START CODE HERE ###
    v_ = np.zeros(len(env.A))
    best_a = 0
    for a in env.A:
        transitions = env.transitions(s, a)
        v_opt = 0
        for s_, (r, p) in enumerate(transitions):
            v_[a] += p*(r + gamma*V[s_])

    V[s] = np.max(v_)

    ### END CODE HERE ###

env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration(env, gamma, theta)