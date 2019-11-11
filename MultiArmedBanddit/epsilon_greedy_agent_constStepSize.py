import main_agent
import utils
import numpy as np


# Constant Step Size Agent Here [Graded]
# Greedy agent here
class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
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
        # self.step_size : A float which is the current step size for the agent.
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################

        # Update q_values for action taken at previous time step
        # using self.step_size intead of using self.arm_count
        # (~1-2 lines)
        ### START CODE HERE ###
        self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size * (
                reward - self.q_values[self.last_action])
        ### END CODE HERE ###

        # Choose action using epsilon greedy. This is the same as you implemented above.
        # (~4 lines)
        ### START CODE HERE ###
        rnd = np.random.random()
        if (rnd > self.epsilon):
            current_action = np.random.choice(4)
        else:
            current_action = utils.argmax(self.q_values)
            ### END CODE HERE ###

        self.last_action = current_action

        return current_action