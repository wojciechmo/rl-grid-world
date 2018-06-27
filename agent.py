import numpy as np
import random

# ------------------------------------------------------------------------------------------
# ------------------------------------- Base Agent -----------------------------------------
# ------------------------------------------------------------------------------------------

class BaseAgent:
	
	def __init__(self, alpha, epsilon, discount, action_space, state_space):
 
		self.action_space = action_space
		self.alpha = alpha
		self.epsilon = epsilon
		self.discount = discount
		self.qvalues = np.zeros((state_space, action_space), np.float32)
		
	def update(self, state, action, reward, next_state, next_state_possible_actions, done):

		# Q(s,a) = (1.0 - alpha) * Q(s,a) + alpha * (reward + discount * V(s'))

		if done==True:
			qval_dash = reward
		else:
			qval_dash = reward + self.discount * self.get_value(next_state, next_state_possible_actions)
			
		qval_old = self.qvalues[state][action]      
		qval = (1.0 - self.alpha)* qval_old + self.alpha * qval_dash
		self.qvalues[state][action] = qval
        
	def get_best_action(self, state, possible_actions):

		best_action = possible_actions[0]
		value = self.qvalues[state][possible_actions[0]]
        
		for action in possible_actions:
			q_val = self.qvalues[state][action]
			if q_val > value:
				value = q_val
				best_action = action

		return best_action

	def get_action(self, state, possible_actions):
         
		# with probability epsilon take random action, otherwise - the best policy action

		epsilon = self.epsilon

		if epsilon > np.random.uniform(0.0, 1.0):
			chosen_action = random.choice(possible_actions)
		else:
			chosen_action = self.get_best_action(state, possible_actions)

		return chosen_action
        
	def get_value(self, state, possible_actions):
		
		pass

# ------------------------------------------------------------------------------------------
# ---------------------------------- Q-Learning Agent --------------------------------------
# ------------------------------------------------------------------------------------------

class QLearningAgent(BaseAgent):

	def get_value(self, state, possible_actions):

		# estimate V(s) as maximum of Q(state,action) over possible actions

		value = self.qvalues[state][possible_actions[0]]
       
		for action in possible_actions:
			q_val = self.qvalues[state][action]
			if q_val > value:
				value = q_val

		return value

# ------------------------------------------------------------------------------------------
# ------------------------------ Expected Value SARSA Agent --------------------------------
# ------------------------------------------------------------------------------------------
    
class EVSarsaAgent(BaseAgent):
    
	def get_value(self, state, possible_actions):
		
		# estimate V(s) as expected value of Q(state,action) over possible actions assuming epsilon-greedy policy
		# V(s) = sum [ p(a|s) * Q(s,a) ]
          
		best_action = possible_actions[0]
		max_val = self.qvalues[state][possible_actions[0]]
		
		for action in possible_actions:
            
			q_val = self.qvalues[state][action]
			if q_val > max_val:
				max_val = q_val
				best_action = action
        
		state_value = 0.0
		n_actions = len(possible_actions)
		
		for action in possible_actions:
            
			if action == best_action:
				trans_prob = 1.0 - self.epsilon + self.epsilon/n_actions
			else:
				trans_prob = self.epsilon/n_actions
                   
			state_value = state_value + trans_prob * self.qvalues[state][action]

		return state_value
