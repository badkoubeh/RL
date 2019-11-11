import numpy as np

pi_ = np.zeros(4)
pi_[0] = 1
#print(pi_)

num_spaces = 4
num_prices = 4
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1

#print(city_policy)

city_policy[0] = np.array(pi_)

print(city_policy)