import numpy as np


# softmax function
def softmax(action_values, tau=1.0):
    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1).reshape((-1, 1))

    exp_perferences = np.exp(preferences - max_preference)
    sum_of_exp_preferences = np.sum(exp_perferences, axis=1)

    action_probs = exp_perferences / sum_of_exp_preferences.reshape((-1, 1))
    return action_probs.squeeze()
