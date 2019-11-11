# Epsilon Greedy Agent here [Graded]

import greedy_agent
import random_agent
import utils
import numpy as np


class EpsilonGreedyAgent(greedy_agent.GreedyAgent, random_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this for this assignment
        as you will not use it until future lessons.
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################

        # Update action-values - this should be the same update as your greedy agent above
        # (~3-5 lines)
        ### START CODE HERE ###
        #self.arm_count[self.last_action] = self.arm_count[self.last_action] + 1
        #self.step_size = 1 / sum(self.arm_count)
        #self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size * (
        #        reward - self.q_values[self.last_action])
        ### END CODE HERE ###

        # Choose action using epsilon greedy
        # Randomly choose a number between 0 and 1 and see if it is less than self.epsilon
        # (Hint: look at np.random.random()). If it is, set current_action to a random action.
        # Otherwise choose current_action greedily as you did above.
        # (~4 lines)
        ### START CODE HERE ###
        rnd = np.random.random()
        if rnd > self.epsilon:
            current_action = self.random_agent_step(reward, observation)
        else:
            current_action = self.greedy_agent_step(reward, observation)
        ### END CODE HERE ###

        #self.last_action = current_action

        return current_action
