# Import necessary libraries
import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import utils
import greedy_agent
import epsilon_greedy_agent


# Test argmax implentation
from epsilon_greedy_agent_constStepSize import EpsilonGreedyAgentConstantStepsize

test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
assert utils.argmax(test_array) == 8, "Check your argmax implementation returns the index of the largest value"

test_array = [1, 0, 0, 1]
total = 0
for i in range(100):
    total += utils.argmax(test_array)


np.save("argmax_test", total)

assert total > 0, "Make sure your argmax implementation randomly choooses among the largest values. Make sure you are not setting a random seed (do not use np.random.seed)"
assert total != 300, "Make sure your argmax implementation randomly choooses among the largest values."

# Do not modify this cell
# Test for Greedy Agent Code
greedy_agent = greedy_agent.GreedyAgent()
greedy_agent.q_values = [0, 0, 1.0, 0, 0]
greedy_agent.arm_count = [0, 1, 0, 0, 0]
greedy_agent.last_action = 1
action = greedy_agent.greedy_agent_step(1, 0)
#print(greedy_agent.q_values)
#np.save("greedy_test", greedy_agent.q_values)
print("Output:")
print(greedy_agent.q_values)
print("Expected Output:")
print([0, 0.5, 1.0, 0, 0])

assert action == 2, "Check that you are using argmax to choose the action with the highest value."
assert greedy_agent.q_values == [0, 0.5, 1.0, 0, 0], "Check that you are updating q_values correctly."

# Do not modify this cell
# Test Code for Epsilon Greedy Agent
e_greedy_agent = epsilon_greedy_agent.EpsilonGreedyAgent()
e_greedy_agent.q_values = [0, 0, 1.0, 0, 0]
e_greedy_agent.arm_count = [0, 1, 0, 0, 0]
e_greedy_agent.num_actions = 5
e_greedy_agent.last_action = 1
e_greedy_agent.epsilon = 0.1
action = e_greedy_agent.agent_step(1, 0)
print("Output:")
print(e_greedy_agent.q_values)
#print("Expected Output:")
#print([0, 0.5, 1.0, 0, 0])

# assert action == 2, "Check that you are using argmax to choose the action with the highest value."
assert e_greedy_agent.q_values == [0, 0.5, 1.0, 0, 0], "Check that you are updating q_values correctly."

for step_size in [0.01, 0.1, 0.5, 1.0]:
    e_greedy_agent = EpsilonGreedyAgentConstantStepsize()
    e_greedy_agent.q_values = [0, 0, 1.0, 0, 0]
    # e_greedy_agent.arm_count = [0, 1, 0, 0, 0]
    e_greedy_agent.num_actions = 5
    e_greedy_agent.last_action = 1
    e_greedy_agent.epsilon = 0.0
    e_greedy_agent.step_size = step_size
    action = e_greedy_agent.agent_step(1, 0)
    print("Output for step size: {}".format(step_size))
    print(e_greedy_agent.q_values)
    print("Expected Output:")
    print([0, step_size, 1.0, 0, 0])
    assert e_greedy_agent.q_values == [0, step_size, 1.0, 0, 0], "Check that you are updating q_values correctly using the stepsize."