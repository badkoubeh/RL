# Greedy agent here [Graded]

import main_agent
import utils


class GreedyAgent(main_agent.Agent):
    def greedy_agent_step(self, reward, observation):
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
        #######################

        # Update action values. Hint: Look at the algorithm in section 2.4 of the textbook.
        # Increment the counter in self.arm_count for the action from the previous time step
        # Update the step size using self.arm_count
        # Update self.q_values for the action from the previous time step
        # (~3-5 lines)
        ### START CODE HERE ###

        self.arm_count[self.last_action] = self.arm_count[self.last_action] + 1
        self.step_size = 1 / sum(self.arm_count)
        self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size*(reward - self.q_values[self.last_action])
        ### END CODE HERE ###

        # current action = ? # Use the argmax function you created above
        # (~2 lines)
        ### START CODE HERE ###
        current_action = utils.argmax(self.q_values)
        ### END CODE HERE ###

        #self.last_action = current_action

        return current_action
