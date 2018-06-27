import environment
import agent

# ------------------------------------ environment 1 -----------------------------------------
gridH, gridW = 4, 4
start_pos = None
end_positions = [(0, 3), (1, 3)]
end_rewards = [10.0, -60.0]
blocked_positions = [(1, 1), (2, 1)]
default_reward= -0.2
# ------------------------------------ environment 2 -----------------------------------------
'''
gridH, gridW = 8, 4
start_pos = (7, 0)
end_positions = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
end_rewards = [10.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0]
blocked_positions = []
default_reward= -0.2
'''
# ------------------------------------ environment 3 -----------------------------------------
'''
gridH, gridW = 8, 9
start_pos = None
end_positions = [(2, 2), (3, 5), (4, 5), (5, 5), (6, 5)]
end_rewards = [10.0, -30.0, -30.0, -30.0, -30.0]
blocked_positions = [(i, 1) for i in range(1, 7)]+ [(1, i) for i in range(1, 8)] + [(i, 7) for i in range(1, 7)]
default_reward= -0.5
'''
# ------------------------------------ environment 4 -----------------------------------------
'''
gridH, gridW = 9, 7
start_pos = None
end_positions = [(0, 3), (2, 4), (6, 2)]
end_rewards = [20.0, -50.0, -50.0]
blocked_positions = [(2, i) for i in range(3)] + [(6, i) for i in range(4, 7)]
default_reward = -0.1
'''
# --------------------------------------------------------------------------------------------

env = environment.Environment(gridH, gridW, end_positions, end_rewards, blocked_positions, start_pos, default_reward)

alpha = 0.2
epsilon = 0.5
discount = 0.99
action_space = env.action_space
state_space = env.state_space

#agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, state_space)
agent = agent.EVSarsaAgent(alpha, epsilon, discount, action_space, state_space)

env.render(agent.qvalues)
state = env.get_state()

while(True):

	possible_actions = env.get_possible_actions()
	action = agent.get_action(state, possible_actions)
	next_state, reward, done = env.step(action)
	env.render(agent.qvalues)

	next_state_possible_actions = env.get_possible_actions()
	agent.update(state, action, reward, next_state, next_state_possible_actions, done)
	state = next_state

	if done == True:	
		env.reset_state()
		env.render(agent.qvalues)
		state = env.get_state()
		continue
